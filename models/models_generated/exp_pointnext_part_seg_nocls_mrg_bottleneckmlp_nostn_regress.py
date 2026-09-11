import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction, PointNetFeaturePropagation

# 如果有使用 STN，請確保 utils 中有 STN3d；若無則會回傳 None
try:
    from pointnet2_utils import STN3d
except ImportError:
    pass

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
# 2. Main Model: MRG + BOTTLENECKMLP + NOSTN
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
# 3. Loss Function (Dice + BCE)
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

class get_loss(nn.Module):
    def __init__(self, weight=None):
        super(get_loss, self).__init__()
        self.register_buffer('iden', torch.eye(3).view(1, 3, 3))
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCELoss() 

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
