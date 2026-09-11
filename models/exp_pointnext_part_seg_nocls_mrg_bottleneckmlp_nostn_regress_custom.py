import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction, PointNetFeaturePropagation

# ==========================================
# 0. STN3d (T-Net) 模組
# ==========================================
class STN3d(nn.Module):
    def __init__(self, channel):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        # 動態獲取當前 device 避免 CUDA 錯誤
        iden = torch.eye(3, dtype=torch.float32, device=x.device).view(1, 9).repeat(batchsize, 1)
        x = x + iden
        x = x.view(-1, 3, 3)
        return x

# ==========================================
# 1. MLP Block: BOTTLENECKMLP
# ==========================================
class BottleneckMLP(nn.Module):
    def __init__(self, in_channel, mlp):
        super(BottleneckMLP, self).__init__()
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
# 2. Main Model: MRG + BOTTLENECKMLP + NOSTN (REGRESS_CUSTOM)
# ==========================================
class get_model(nn.Module):
    def __init__(self, num_part, normal_channel=True, grid_num=4):
        super(get_model, self).__init__()
        
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel
        
        # No STN initialized

        # --- Encoder (MRG 變體) ---
        self.sa1 = PointNetSetAbstractionMsg(1024, [0.05, 0.1], [16, 32], in_channel, [[32, 32, 64], [64, 64, 128]])
        self.res1 = BottleneckMLP(192, [64, 128])
        self.sa2 = PointNetSetAbstractionMsg(256, [0.1, 0.2], [32, 64], 192, [[128, 128, 256], [128, 196, 256]])
        self.sa2_mrg = PointNetSetAbstractionMsg(256, [0.2, 0.4], [16, 32], in_channel, [[64, 64, 128], [64, 128, 256]])
        self.res2 = BottleneckMLP(896, [256, 512])
        self.sa3 = PointNetSetAbstractionMsg(64, [0.3, 0.6], [64, 128], 896, [[256, 512, 512], [256, 512, 512]])
        self.res3 = BottleneckMLP(1024, [256, 512])
        # --- Decoder ---
        self.fp3 = PointNetFeaturePropagation(1920, [512, 512])
        self.fp2 = PointNetFeaturePropagation(704, [256, 256])
        self.fp1 = PointNetFeaturePropagation(256 + in_channel, [128, 128, 128])

        # Prediction Head
        self.head_mlp = nn.Sequential(
            nn.Conv1d(128, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),        
            nn.Conv1d(128, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.4),       
            nn.Conv1d(64, 1, 1)  
        )

    def forward(self, xyz):
        B, C, N = xyz.shape
        l0_xyz = xyz[:, :3, :]
        
        if self.normal_channel:
            l0_points = xyz 
        else:
            l0_points = l0_xyz

        trans = None # No STN transformation

        # Encoder (MRG 雙路徑)
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l1_points = self.res1(l1_points)
        
        l2_xyz, l2_points_msg = self.sa2(l1_xyz, l1_points)
        _, l2_points_mrg = self.sa2_mrg(l0_xyz, l0_points)
        l2_points = torch.cat([l2_points_msg, l2_points_mrg], dim=1)
        l2_points = self.res2(l2_points)
        
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l3_points = self.res3(l3_points)
        
        # Decoder
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, l0_points, l1_points)

        x = self.head_mlp(l0_points)
        x = torch.sigmoid(x).squeeze(1) # [B, N]
        return x, trans

# ==========================================
# 3. Regression Loss (Region-Normalized MSE + Soft Dice)
# ==========================================
class get_loss(nn.Module):
    def __init__(self, bg_weight=0.2, threshold=0.1):
        super(get_loss, self).__init__()
        self.register_buffer('iden', torch.eye(3).view(1, 3, 3))
        self.threshold = threshold
        self.bg_weight = bg_weight 

    def forward(self, pred, target, trans_feat, epoch):
        if pred.dim() == 3: pred = pred.squeeze(-1)
        if target.dim() == 3: target = target.squeeze(-1)
        target = target.float()
        diff_sq = (pred - target) ** 2

        fg_mask = (target > self.threshold).float()
        bg_mask = 1.0 - fg_mask

        fg_sum = fg_mask.sum(dim=1).clamp(min=1e-8)
        bg_sum = bg_mask.sum(dim=1).clamp(min=1e-8)

        fg_loss = (diff_sq * fg_mask).sum(dim=1) / fg_sum
        bg_loss = (diff_sq * bg_mask).sum(dim=1) / bg_sum

        norm_mse_loss = (fg_loss + self.bg_weight * bg_loss).mean()

        intersection = (pred * target).sum(dim=1)
        union = (pred ** 2).sum(dim=1) + (target ** 2).sum(dim=1) 
        soft_dice = 1.0 - (2.0 * intersection + 1e-5) / (union + 1e-5)
        soft_dice_loss = soft_dice.mean()

        main_loss = norm_mse_loss + 0.5 * soft_dice_loss

        if trans_feat is not None:
            batch_size = trans_feat.size(0)
            iden = self.iden.repeat(batch_size, 1, 1)
            trans_ortho = torch.bmm(trans_feat, trans_feat.transpose(1, 2))
            stn_loss = torch.norm(trans_ortho - iden, p='fro')
        else:
            stn_loss = 0.0

        return 10.0 * main_loss + 0.1 * stn_loss
