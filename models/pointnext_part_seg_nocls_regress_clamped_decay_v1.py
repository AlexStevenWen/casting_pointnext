import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstractionMsg, PointNetFeaturePropagation

# ==========================================
# 核心組件: Inverted Residual MLP
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
# 主模型: 純淨版 PointNeXt (無網格引導)
# ==========================================
class get_model(nn.Module):
    def __init__(self, num_part, normal_channel=True, grid_num=4):
        super(get_model, self).__init__()
        # 注意：grid_num 參數保留是為了相容 main.py 的呼叫，但在這裡不會被使用
        
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel

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
        
        # [關鍵差異] FP1 不再接收 Offset 和 Grid Confidence
        # 輸入通道恢復為標準的 256 + in_channel
        self.fp1 = PointNetFeaturePropagation(256 + in_channel, [128, 128, 128])

        # --- Regression Head ---
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        
        # [修改] 輸出單一通道 (Intensity)
        self.conv2 = nn.Conv1d(128, 1, 1)

    def forward(self, xyz):
        B, C, N = xyz.shape
        l0_xyz = xyz[:, :3, :]
        l0_points = xyz

        # 1. Encoder
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l1_points = self.res1(l1_points)

        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l2_points = self.res2(l2_points)

        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l3_points = self.res3(l3_points)

        # 2. Decoder
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        
        # [關鍵差異] 這裡直接傳入 l0_points，沒有 concat 任何額外特徵
        l0_points = self.fp1(l0_xyz, l1_xyz, l0_points, l1_points)

        # 3. Head
        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        x = self.conv2(x)
        
        # [修改] 使用 Sigmoid 進行 0~1 回歸
        x = torch.sigmoid(x).transpose(2, 1) # [B, N, 1]
        
        # 為了相容 main.py，回傳兩個值
        # 第二個值 (Grid Logits) 設為 None，因為這個模型不預測網格
        return x.squeeze(-1), None

# ==========================================
# 3. Loss Function
# ==========================================
class get_loss(nn.Module):
    def __init__(self, weight=None):
        super(get_loss, self).__init__()
        self.mse_criterion = nn.MSELoss()

    def forward(self, pred, target, grid_pred=None, grid_target=None):
        """
        純回歸模型的 Loss 只計算 MSE
        忽略 Grid 相關的輸入
        """
        # Regression Loss (MSE)
        # 加大權重以匹配實驗組的量級
        reg_loss = self.mse_criterion(pred, target)
        
        return 50.0 * reg_loss