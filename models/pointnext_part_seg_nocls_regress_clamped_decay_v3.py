import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstractionMsg, PointNetFeaturePropagation

# ==========================================
# 1. 空間變換模組 (STN)
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
        
        # 初始化為單位矩陣
        self.register_buffer('iden', torch.tensor([1,0,0,0,1,0,0,0,1], dtype=torch.float).view(1,9))

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

        x = x + self.iden
        x = x.view(-1, 3, 3)
        return x

# ==========================================
# 2. Inverted Residual MLP
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
# 3. 主模型: PointNeXt + STN + Dropout
# ==========================================
class get_model(nn.Module):
    def __init__(self, num_part, normal_channel=True, grid_num=4):
        super(get_model, self).__init__()
        
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel
        
        self.stn = STN3d(in_channel)

        # --- Encoder ---
        self.sa1 = PointNetSetAbstractionMsg(1024, [0.03, 0.06], [16, 32], in_channel, [[32, 32, 64], [64, 64, 128]])
        self.res1 = InvResMLP(192, [64, 128])
        
        self.sa2 = PointNetSetAbstractionMsg(256, [0.1, 0.2], [32, 64], 192, [[128, 128, 256], [128, 196, 256]])
        self.res2 = InvResMLP(512, [128, 256])
        
        self.sa3 = PointNetSetAbstractionMsg(64, [0.3, 0.6], [64, 128], 512, [[256, 512, 512], [256, 512, 512]])
        self.res3 = InvResMLP(1024, [256, 512])

        # --- Decoder ---
        self.fp3 = PointNetFeaturePropagation(1536, [512, 512])
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
        l0_points = xyz

        # [修正 1] STN 必須回傳矩陣 trans，不能是 None
        trans = self.stn(l0_points)
        l0_xyz = torch.bmm(l0_xyz.transpose(1, 2), trans).transpose(1, 2)
        
        l0_points = torch.cat([l0_xyz, xyz[:, 3:, :]], dim=1) if self.normal_channel else l0_xyz

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l1_points = self.res1(l1_points)

        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l2_points = self.res2(l2_points)

        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l3_points = self.res3(l3_points)

        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, l0_points, l1_points)

        x = self.head_mlp(l0_points)
        x = torch.sigmoid(x).transpose(2, 1)
        
        # [修正 2] 回傳 trans 供 Loss 使用
        return x.squeeze(-1), trans

# ==========================================
# 4. Loss Function (修正版：精度 + 穩定度)
# ==========================================
class get_loss(nn.Module):
    def __init__(self, weight=None):
        super(get_loss, self).__init__()
        # 準備單位矩陣 (GPU)
        self.register_buffer('iden', torch.eye(3).view(1, 3, 3))

    def forward(self, pred, target, trans_feat, epoch):
        """
        結合了：
        1. Alpha=50: 強迫預測值衝到 1.0 (解決不精問題)
        2. Warmup: 前 30 Epoch 不罰鬼影 (避免模型躺平)
        3. STN Regularizer: 強迫旋轉矩陣正規 (避免熱力圖變形糊掉)
        """
        diff = (pred - target) ** 2
        
        # [策略 1] 提高正樣本權重 (Alpha)
        alpha = 0
        beta = 0    
        weights = 1.0 + (target * alpha) + (pred * beta)
        main_loss = (diff * weights).mean()
        
        # [策略 3] STN 正則化 (解決熱力圖模糊)
        if trans_feat is not None:
            batch_size = trans_feat.size(0)
            iden = self.iden.repeat(batch_size, 1, 1)
            trans_ortho = torch.bmm(trans_feat, trans_feat.transpose(1, 2))
            stn_loss = torch.norm(trans_ortho - iden, p='fro')
        else:
            stn_loss = 0.0

        return 10.0 * main_loss + 0.1 * stn_loss