import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstractionMsg, PointNetFeaturePropagation

# PointNeXt 核心模組
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

class get_model(nn.Module):
    def __init__(self, num_part, normal_channel=True, grid_num=4):
        super(get_model, self).__init__()
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel
        self.grid_num = grid_num
        self.num_grid_out = grid_num ** 3 

        # --- Encoder ---
        self.sa1 = PointNetSetAbstractionMsg(1024, [0.03, 0.06], [16, 32], in_channel, [[32, 32, 64], [64, 64, 128]])
        self.res1 = InvResMLP(192, [64, 128])

        self.sa2 = PointNetSetAbstractionMsg(256, [0.1, 0.2], [32, 64], 192, [[128, 128, 256], [128, 196, 256]])
        self.res2 = InvResMLP(512, [128, 256])

        self.sa3 = PointNetSetAbstractionMsg(64, [0.3, 0.6], [64, 128], 512, [[256, 512, 512], [256, 512, 512]])
        self.res3 = InvResMLP(1024, [256, 512])

        # --- Grid機率預測分支 (Grid Head) ---
        self.grid_head = nn.Sequential(
            nn.Linear(1024, 512),
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
        self.fp1 = PointNetFeaturePropagation(256 + in_channel, [128, 128, 128])

        # 分類頭
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_part, 1)

    def forward(self, xyz):
        B, C, N = xyz.shape
        l0_xyz = xyz[:, :3, :] # [B, 3, N]
        l0_points = xyz

        # Encoder Stage
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l1_points = self.res1(l1_points)

        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l2_points = self.res2(l2_points)

        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l3_points = self.res3(l3_points) 

        # --- 1. 提取全局特徵預測格子 ---
        global_feat = torch.max(l3_points, dim=2)[0] 
        grid_logits = self.grid_head(global_feat) # [B, grid_num^3]
        grid_probs = torch.sigmoid(grid_logits)   # [B, grid_num^3]

        # --- 2. 核心：Gate Attention (軟導航) ---
        # 將 XYZ [-1, 1] 映射到格子索引 [0, grid_num-1]
        # 公式: idx = floor((x + 1) / 2 * grid_num)
        gn = self.grid_num
        # 使用點雲 l0_xyz 對應所有 N 個點
        norm_xyz = (l0_xyz.transpose(1, 2) + 1.0) / 2.0 # [B, N, 3]
        grid_coords = (norm_xyz * gn).long().clamp(0, gn - 1) # [B, N, 3]
        
        # 將 3D 座標轉為 1D 索引 (gi * n^2 + gj * n + gk)
        point_grid_idx = grid_coords[:, :, 0] * (gn**2) + \
                         grid_coords[:, :, 1] * gn + \
                         grid_coords[:, :, 2] # [B, N]

        # 從預測的格子機率中提取每個點對應的權重
        # batch_indices 用於多樣本索引
        batch_idx = torch.arange(B, device=xyz.device).view(B, 1).expand(B, N)
        point_weights = grid_probs[batch_idx, point_grid_idx] # [B, N]

        # Decoder Stage
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, l0_points, l1_points)

        # --- 3. 注入注意力：加強對應區域特徵 ---
        # 這裡採用加成機制，讓模型在定位區域有更強的信號
        # 特徵調整公式: f = f * (1 + SoftGate)
        l0_points = l0_points * (1.0 + point_weights.unsqueeze(1))

        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        x = x.transpose(2, 1).contiguous()

        return x, grid_logits

# 優化後的 Focal Loss：專門解決格子類別極度不平衡
class FocalLossBCE(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, pos_weight=None):
        super(FocalLossBCE, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none', pos_weight=self.pos_weight)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt)**self.gamma * BCE_loss
        return F_loss.mean()

class get_loss(nn.Module):
    def __init__(self, weight=None):
        super(get_loss, self).__init__()
        self.weight = weight
        # 為格子損失使用 Focal Loss 穩定訓練
        self.grid_criterion = FocalLossBCE(pos_weight=torch.tensor([10.0]).cuda())

    def forward(self, pred, target, grid_pred, grid_target):
        # 點分割損失 (Label Smoothing 有助於減少過擬合)
        seg_loss = F.nll_loss(pred, target, weight=self.weight, label_smoothing=0.1)
        
        # 格子機率損失
        grid_loss = self.grid_criterion(grid_pred, grid_target)
        
        # 由於現在有 Attention 機制，點分割的梯度也會流回 grid_head
        # 建議權重比例 1 : 1
        return seg_loss + 1.0 * grid_loss