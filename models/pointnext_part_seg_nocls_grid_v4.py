import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pointnet2_utils import PointNetSetAbstractionMsg, PointNetFeaturePropagation

# ==========================================
# 空間變換模組 (STN) - 用於對齊輸入點雲
# ==========================================
class STN3d(nn.Module):
    def __init__(self, channel):
        super(STN3d, self).__init__()
        self.conv1 = nn.Conv1d(channel, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

    def forward(self, x):
        batchsize = x.size()[0]
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        
        # 初始化為單位矩陣
        iden = torch.tensor([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=torch.float).view(1, 9).repeat(batchsize, 1).to(x.device)
        x = x + iden
        x = x.view(-1, 3, 3)
        return x

# ==========================================
# Inverted Residual MLP - 特徵提取核心
# ==========================================
class InvResMLP(nn.Module):
    def __init__(self, in_channel, mlp):
        super(InvResMLP, self).__init__()
        self.conv1 = nn.Conv1d(in_channel, mlp[0], 1)
        self.bn1 = nn.BatchNorm1d(mlp[0])
        self.conv2 = nn.Conv1d(mlp[0], mlp[1], 1)
        self.bn2 = nn.BatchNorm1d(mlp[1])
        self.conv3 = nn.Conv1d(mlp[1], in_channel, 1)
        self.bn3 = nn.BatchNorm1d(in_channel)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        return F.relu(x + residual)

# ==========================================
# 主模型架構 (Encoder-Decoder + Grid Head)
# ==========================================
class get_model(nn.Module):
    def __init__(self, num_part, normal_channel=True, grid_num=4):
        super(get_model, self).__init__()
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel
        self.grid_num = grid_num
        self.num_grid_out = grid_num ** 3 

        self.stn = STN3d(in_channel)

        # --- Encoder ---
        self.sa1 = PointNetSetAbstractionMsg(1024, [0.03, 0.06], [16, 32], in_channel, [[32, 32, 64], [64, 64, 128]])
        self.res1 = InvResMLP(192, [64, 128])
        self.sa2 = PointNetSetAbstractionMsg(256, [0.1, 0.2], [32, 64], 192, [[128, 128, 256], [128, 196, 256]])
        self.res2 = InvResMLP(512, [128, 256])
        self.sa3 = PointNetSetAbstractionMsg(64, [0.3, 0.6], [64, 128], 512, [[256, 512, 512], [256, 512, 512]])
        self.res3 = InvResMLP(1024, [256, 512])

        # --- 多尺度 Grid Head (用於粗粒度定位) ---
        self.grid_head = nn.Sequential(
            nn.Linear(1024 + 512, 512), 
            nn.LayerNorm(512), 
            nn.ReLU(inplace=True),
            nn.Dropout(0.5), 
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.num_grid_out)
        )

        # --- Decoder ---
        self.fp3 = PointNetFeaturePropagation(1536, [512, 512])
        self.fp2 = PointNetFeaturePropagation(704, [256, 256])
        # 關鍵修改：輸入通道增加 3，用於接收 Local Offset (局部座標)
        self.fp1 = PointNetFeaturePropagation(256 + in_channel + 3, [128, 128, 128])

        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_part, 1)

    def forward(self, xyz):
        B, C, N = xyz.shape
        l0_xyz = xyz[:, :3, :] 
        l0_points = xyz

        # 1. 空間校正 (STN)
        trans = self.stn(l0_points)
        l0_xyz = torch.bmm(l0_xyz.transpose(1, 2), trans).transpose(1, 2)
        l0_points = torch.cat([l0_xyz, xyz[:, 3:, :]], dim=1) if self.normal_channel else l0_xyz

        # 2. Encoder 提取特徵
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l1_points = self.res1(l1_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l2_points = self.res2(l2_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l3_points = self.res3(l3_points) 

        # 3. 預測格子機率 (Grid Task)
        feat_mid = torch.max(l2_points, dim=2)[0] 
        feat_deep = torch.max(l3_points, dim=2)[0] 
        grid_logits = self.grid_head(torch.cat([feat_mid, feat_deep], dim=1)) 
        grid_probs = torch.sigmoid(grid_logits) 

        # 4. 關鍵優化：計算「點對應格子」的幾何特徵 (Spatial Geometric Embedding)
        # 這一步是提升 Recall 的核心，讓模型知道點在格子裡的「相對位置」
        gn = self.grid_num
        norm_xyz = (l0_xyz.transpose(1, 2) + 1.0) / 2.0 
        grid_coords_f = norm_xyz * gn
        grid_coords_i = grid_coords_f.long().clamp(0, gn - 1)
        
        # 計算 Grid Center
        grid_centers = (grid_coords_i.float() + 0.5) / gn * 2.0 - 1.0
        
        # 計算 Local Offset (點 - 格子中心)
        local_offset = l0_xyz.transpose(1, 2) - grid_centers # [B, N, 3]

        # 取得每個點對應的格子信心分數 (Grid Probability Lookup)
        point_in_grid_idx = grid_coords_i[:,:,0]*(gn**2) + grid_coords_i[:,:,1]*gn + grid_coords_i[:,:,2]
        batch_idx = torch.arange(B).view(B, 1).expand(B, N).to(xyz.device)
        point_weights = grid_probs[batch_idx, point_in_grid_idx].unsqueeze(1) # [B, 1, N]

        # 5. Decoder 融合特徵
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        
        # 在 FP1 階段注入 Local Offset，這是 PointNext 原始架構沒有的
        l0_points_with_offset = torch.cat([l0_points, local_offset.transpose(1, 2)], dim=1)
        l0_points = self.fp1(l0_xyz, l1_xyz, l0_points_with_offset, l1_points)

        # 6. 注入格子注意力：軟過濾 (Soft Attention)
        # 使用 0.8 + 0.4 * weights 而不是乘 0，保證 Recall 不會被誤殺
        l0_points = l0_points * (0.8 + 0.4 * point_weights)

        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        x = x.transpose(2, 1).contiguous()

        return x, grid_logits

# ==========================================
# Loss Function 定義
# ==========================================
class FocalLossBCE(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, pos_weight=None):
        super(FocalLossBCE, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, inputs, targets):
        # 針對小樣本的 Focal Loss 計算
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none', pos_weight=self.pos_weight)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt)**self.gamma * BCE_loss
        return F_loss.mean()

class get_loss(nn.Module):
    def __init__(self, weight=None):
        super(get_loss, self).__init__()
        self.weight = weight
        # 針對不平衡數據，這裡使用極高的 pos_weight (8.0) 來強迫模型關注少數類別(澆口)
        self.grid_criterion = FocalLossBCE(pos_weight=torch.tensor([8.0]).cuda())

    def forward(self, pred, target, grid_pred, grid_target):
        # 1. 點分割損失 (Label Smoothing = 0.2 強制泛化)
        seg_loss = F.nll_loss(pred, target, weight=self.weight, label_smoothing=0.2)
        
        # 2. 格子損失 (使用 Focal Loss)
        grid_loss = self.grid_criterion(grid_pred, grid_target)
        
        # 3. 總損失權重
        return seg_loss + 5.0 * grid_loss