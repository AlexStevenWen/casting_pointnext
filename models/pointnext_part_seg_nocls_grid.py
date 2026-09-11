import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstractionMsg, PointNetFeaturePropagation

# ==========================================
# 1. 空間變換模組 (STN) - 保持不變
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
        iden = torch.tensor([1,0,0,0,1,0,0,0,1], dtype=torch.float).view(1,9).repeat(batchsize,1).to(x.device)
        x = x + iden
        x = x.view(-1, 3, 3)
        return x

class InvResMLP(nn.Module):
    def __init__(self, in_channel, mlp):
        super(InvResMLP, self).__init__()
        self.conv1 = nn.Conv1d(in_channel, mlp[0], 1); self.bn1 = nn.BatchNorm1d(mlp[0])
        self.conv2 = nn.Conv1d(mlp[0], mlp[1], 1); self.bn2 = nn.BatchNorm1d(mlp[1])
        self.conv3 = nn.Conv1d(mlp[1], in_channel, 1); self.bn3 = nn.BatchNorm1d(in_channel)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        return F.relu(x + residual)

# ==========================================
# 2. 主模型 (Feature Injection 版)
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

        # --- Grid Head ---
        # 輸入特徵拼接: Layer2(512) + Layer3(1024) = 1536
        self.grid_head = nn.Sequential(
            nn.Linear(1536, 512), 
            nn.LayerNorm(512), 
            nn.ReLU(), 
            nn.Linear(512, self.num_grid_out)
        )

        # --- Decoder ---
        self.fp3 = PointNetFeaturePropagation(1536, [512, 512])
        self.fp2 = PointNetFeaturePropagation(704, [256, 256])
        
        # [關鍵修正] FP1 輸入通道計算：
        # 上一層特徵 (256) + 原始點特徵 (in_channel) + Local Offset (3) + Grid Confidence (1)
        self.fp1 = PointNetFeaturePropagation(256 + in_channel + 4, [128, 128, 128])

        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_part, 1)

    def forward(self, xyz):
        B, C, N = xyz.shape
        l0_xyz = xyz[:, :3, :]
        l0_points = xyz

        # 1. Spatial Transform
        trans = self.stn(l0_points)
        l0_xyz = torch.bmm(l0_xyz.transpose(1, 2), trans).transpose(1, 2)
        l0_points = torch.cat([l0_xyz, xyz[:, 3:, :]], dim=1) if self.normal_channel else l0_xyz

        # 2. Encoding
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l1_points = self.res1(l1_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l2_points = self.res2(l2_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l3_points = self.res3(l3_points) 

        # 3. Grid Prediction
        feat_global = torch.cat([torch.max(l2_points, 2)[0], torch.max(l3_points, 2)[0]], dim=1)
        grid_logits = self.grid_head(feat_global)
        grid_probs = torch.sigmoid(grid_logits) # [B, GridNum^3]

        # 4. Feature Injection Preparation (準備幾何注入特徵)
        gn = self.grid_num
        norm_xyz = (l0_xyz.transpose(1, 2) + 1.0) / 2.0 
        grid_coords = (norm_xyz * gn).long().clamp(0, gn - 1)
        
        # A. 計算 Local Offset (點相對於格子中心的偏移)
        grid_centers = (grid_coords.float() + 0.5) / gn * 2.0 - 1.0
        local_offset = l0_xyz.transpose(1, 2) - grid_centers # [B, N, 3]
        
        # B. 獲取每個點對應格子的 Confidence
        idx = grid_coords[:,:,0]*(gn**2) + grid_coords[:,:,1]*gn + grid_coords[:,:,2]
        batch_idx = torch.arange(B).view(B, 1).expand(B, N).to(xyz.device)
        pt_grid_conf = grid_probs[batch_idx, idx].unsqueeze(1) # [B, 1, N] - 注意這裡是 [B, 1, N] 以便 concat

        # 5. Decoding with Injection
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        
        # [核心操作] 將 Offset 和 Confidence 拼接到原始點特徵上
        # l0_points: [B, C, N]
        # local_offset.transpose: [B, 3, N]
        # pt_grid_conf: [B, 1, N]
        l0_augmented = torch.cat([l0_points, local_offset.transpose(1, 2), pt_grid_conf], dim=1)
        
        l0_points = self.fp1(l0_xyz, l1_xyz, l0_augmented, l1_points)

        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        x = self.conv2(x)
        
        # 輸出 [B, N, C]
        return F.log_softmax(x, dim=1).transpose(2, 1).contiguous(), grid_logits

# ==========================================
# 3. Loss Functions (內建於模型檔案)
# ==========================================
class get_loss(nn.Module):
    def __init__(self, weight=None):
        super(get_loss, self).__init__()
        self.weight = weight
        # Grid Loss 權重調高到 20，確保格子先學好
        self.grid_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([20.0]).cuda())

    def forward(self, pred, target, grid_pred, grid_target):
        """
        pred: [B, N, 2] (LogSoftmaxed)
        target: [B, N]
        grid_pred: [B, GridOut]
        grid_target: [B, GridOut]
        """
        # 1. NLL Loss (分類)
        # NLL需要 [B, C, N]，所以要轉置 pred
        seg_loss = F.nll_loss(pred.transpose(1, 2), target, weight=self.weight)
        
        # 2. Dice Loss (針對小目標 IoU)
        # pred 是 log_softmax，取 exp 變回機率
        # pred shape: [B, N, 2] -> 取 index 1 (澆口機率) -> [B, N]
        pred_prob = torch.exp(pred)[:, :, 1] 
        target_f = target.float()
        
        # 計算 Batch 內每個 Sample 的 Dice，再取平均
        intersection = (pred_prob * target_f).sum(dim=1)
        union = pred_prob.sum(dim=1) + target_f.sum(dim=1)
        
        # Smooth 設為 1e-5 避免除以 0
        dice_score = (2. * intersection + 1e-5) / (union + 1e-5)
        dice_loss = 1 - dice_score.mean()
        
        # 3. Grid Loss
        grid_loss = self.grid_criterion(grid_pred, grid_target)
        
        # 總 Loss：提高 Dice 權重 (3.0) 以拉高 Recall
        return seg_loss + 3.0 * dice_loss + 5.0 * grid_loss