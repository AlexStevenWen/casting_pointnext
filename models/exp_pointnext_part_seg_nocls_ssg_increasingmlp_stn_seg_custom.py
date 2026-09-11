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
# 1. MLP Block: INCREASINGMLP
# ==========================================
class IncreasingMLP(nn.Module):
    def __init__(self, in_channel, mlp):
        super(IncreasingMLP, self).__init__()
        self.conv1 = nn.Conv1d(in_channel, mlp[0], 1)
        self.bn1 = nn.BatchNorm1d(mlp[0])
        self.conv2 = nn.Conv1d(mlp[0], mlp[1], 1)
        self.bn2 = nn.BatchNorm1d(mlp[1])
        self.conv3 = nn.Conv1d(mlp[1], in_channel, 1) 
        self.bn3 = nn.BatchNorm1d(in_channel)
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return x

# ==========================================
# 2. Main Model: SSG + INCREASINGMLP + STN (SEG_CUSTOM)
# ==========================================
class get_model(nn.Module):
    def __init__(self, num_part, normal_channel=True, grid_num=4):
        super(get_model, self).__init__()
        
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel
        
        self.stn = STN3d(in_channel)

        # --- Encoder (SSG) ---
        self.sa1 = PointNetSetAbstraction(1024, 0.1, 32, in_channel + 3, [32, 32, 64], False)
        self.res1 = IncreasingMLP(64, [32, 48])
        self.sa2 = PointNetSetAbstraction(256, 0.2, 32, 64 + 3, [64, 64, 128], False)
        self.res2 = IncreasingMLP(128, [64, 96])
        self.sa3 = PointNetSetAbstraction(64, 0.4, 32, 128 + 3, [128, 128, 256], False)
        self.res3 = IncreasingMLP(256, [128, 192])
        # --- Decoder ---
        self.fp3 = PointNetFeaturePropagation(384, [256, 256])
        self.fp2 = PointNetFeaturePropagation(320, [128, 128])
        self.fp1 = PointNetFeaturePropagation(128 + in_channel, [128, 128, 128])

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

        trans = self.stn(l0_points)
        l0_xyz = torch.bmm(l0_xyz.transpose(1, 2), trans).transpose(1, 2)

        # Encoder
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l1_points = self.res1(l1_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
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
# 3. Segmentation Loss (Dice + Focal BCE)
# ==========================================
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        intersection = (pred * target).sum(dim=1)
        union = pred.sum(dim=1) + target.sum(dim=1)
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        pred = torch.clamp(pred, min=1e-7, max=1.0 - 1e-7)
        bce = - (target * torch.log(pred) + (1.0 - target) * torch.log(1.0 - pred))
        pt = target * pred + (1.0 - target) * (1.0 - pred)
        focal_loss = self.alpha * (1.0 - pt) ** self.gamma * bce
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()
        self.register_buffer('iden', torch.eye(3).view(1, 3, 3))
        self.dice_loss = DiceLoss()
        self.bce_loss = FocalLoss(alpha=0.5, gamma=2.0) 

    def forward(self, pred, target, trans_feat, epoch):
        target = target.float()
        loss_dice = self.dice_loss(pred, target)
        loss_bce = self.bce_loss(pred, target)
        total_seg_loss = loss_dice + loss_bce

        if trans_feat is not None:
            batch_size = trans_feat.size(0)
            iden = self.iden.repeat(batch_size, 1, 1)
            trans_ortho = torch.bmm(trans_feat, trans_feat.transpose(1, 2))
            stn_loss = torch.norm(trans_ortho - iden, p='fro')
        else:
            stn_loss = 0.0

        return total_seg_loss + 0.001 * stn_loss
