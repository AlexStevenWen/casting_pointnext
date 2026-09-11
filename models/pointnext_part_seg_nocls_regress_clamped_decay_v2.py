import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstractionMsg, PointNetFeaturePropagation

# ==========================================
# 核心組件: Inverted Residual MLP (保持不變)
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
# 輔助組件: 空間變換模組 (STN)
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
        
        # [優化] 使用 register_buffer 自動管理 device
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
# 主模型: 純淨版 PointNeXt (無網格引導)
# ==========================================
class get_model(nn.Module):
    def __init__(self, num_part, normal_channel=True, grid_num=4):
        super(get_model, self).__init__()
        # 注意：grid_num 參數保留是為了相容 main.py 的呼叫介面
        
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel
        self.stn = STN3d(in_channel) # 保留 STN 以公平比較

        # --- Encoder ---
        # 參數設置與實驗組完全一致，確保公平性
        self.sa1 = PointNetSetAbstractionMsg(1024, [0.03, 0.06], [16, 32], in_channel, [[32, 32, 64], [64, 64, 128]])
        self.res1 = InvResMLP(192, [64, 128])
        
        self.sa2 = PointNetSetAbstractionMsg(256, [0.1, 0.2], [32, 64], 192, [[128, 128, 256], [128, 196, 256]])
        self.res2 = InvResMLP(512, [128, 256])
        
        self.sa3 = PointNetSetAbstractionMsg(64, [0.3, 0.6], [64, 128], 512, [[256, 512, 512], [256, 512, 512]])
        self.res3 = InvResMLP(1024, [256, 512])

        # --- Decoder ---
        self.fp3 = PointNetFeaturePropagation(1536, [512, 512])
        self.fp2 = PointNetFeaturePropagation(704, [256, 256])
        
        # [關鍵差異] FP1 輸入通道減少 (移除了 Grid 特徵與 Offset)
        # 實驗組: 256 + in_channel + 4 (offset + conf)
        # 對照組: 256 + in_channel
        self.fp1 = PointNetFeaturePropagation(256 + in_channel, [128, 128, 128])

        # --- Regression Head ---
        # self.conv1 = nn.Conv1d(128, 128, 1)
        # self.bn1 = nn.BatchNorm1d(128)
        # self.drop1 = nn.Dropout(0.5)
        # self.conv2 = nn.Conv1d(128, 1, 1)
        self.head_mlp = nn.Sequential(
            nn.Conv1d(128, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 64, 1),  # 新增一層過渡
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 1, 1)     # 輸出層
        )
        # 輸出單一通道 (Intensity)
        

    def forward(self, xyz):
        B, C, N = xyz.shape
        l0_xyz = xyz[:, :3, :]
        l0_points = xyz

        # 1. STN (保持與實驗組一致)
        trans = self.stn(l0_points)
        l0_xyz = torch.bmm(l0_xyz.transpose(1, 2), trans).transpose(1, 2)
        l0_points = torch.cat([l0_xyz, xyz[:, 3:, :]], dim=1) if self.normal_channel else l0_xyz

        # 2. Encoder
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l1_points = self.res1(l1_points)

        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l2_points = self.res2(l2_points)

        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l3_points = self.res3(l3_points)

        # 3. Decoder
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        
        # [關鍵差異] 直接傳入特徵，不進行 Feature Injection
        l0_points = self.fp1(l0_xyz, l1_xyz, l0_points, l1_points)

        # 4. Head
        # x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        # x = self.conv2(x)
        x = self.head_mlp(l0_points)

        

        # 使用 Sigmoid 進行 0~1 回歸
        x = torch.sigmoid(x).transpose(2, 1) # [B, N, 1]
        
        # [相容性處理] 回傳 None 作為 grid_logits
        # train.py 必須要能處理第二個返回值為 None 的情況
        return x.squeeze(-1), None

# ==========================================
# 3. Loss Function (無 Grid Loss)
# ==========================================
class get_loss(nn.Module):
    def __init__(self, weight=None):
        super(get_loss, self).__init__()
        self.mse_criterion = nn.MSELoss()

    def forward(self, pred, target, grid_pred=None, grid_target=None):
        """
        純回歸模型的 Loss 只計算 MSE
        grid_pred 與 grid_target 參數保留以維持 API 相容性，但不參與計算
        """
        # Regression Loss (MSE)
        # 保持 50.0 權重以公平比較 Loss 大小
        # reg_loss = self.mse_criterion(pred, target)
        
        # return 50.0 * reg_loss
        # [關鍵策略]
        # 找出真實值(target)有數值的地方 (例如大於 0.01)
        # 將這些地方的權重設為較大的數值 (例如 20.0)
        # 這代表：在澆口處猜錯，代價是背景處猜錯的 20 倍
        diff = (pred - target) ** 2
        weights = torch.ones_like(target)
        mask = target > 0.01 
        weights[mask] = 50.0  # 你可以調整這個數字 (10.0 ~ 50.0)
        
        # 3. 計算加權後的 MSE
        weighted_loss = (diff * weights).mean()
        
        return 10.0 * weighted_loss