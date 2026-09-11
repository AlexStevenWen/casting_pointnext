import torch
import torch.nn as nn
import torch.nn.functional as F
from models.pointnet2_utils import PointNetSetAbstraction

"""
(新) PointNet++ 姿態回歸模型 (Encoder-Only)
(修正 v2) 修正了 sa1 的 in_channel 維度
"""

class get_model(nn.Module):
    def __init__(self, output_dim=4, normal_channel=True): 
        super(get_model, self).__init__()
        
        # --- (*** 這是您缺少的關鍵修正 ***) ---
        in_channel_l0_points = 6 if normal_channel else 3
        self.normal_channel = normal_channel
        
        # sa1 接收 l0_points (in_channel_l0_points) + l0_xyz (3)
        sa1_in_channel = in_channel_l0_points + 3
        
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=sa1_in_channel, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        
        self.conv1 = nn.Conv1d(256, 1024, 1) # (sa2 輸出 256)
        self.bn_conv1 = nn.BatchNorm1d(1024)
        # --- (*** 修正結束 ***) ---

        # 3. (新) 串聯池化 (Max + Avg)
        self.pool_max = nn.AdaptiveMaxPool1d(1)
        self.pool_avg = nn.AdaptiveAvgPool1d(1)

        # 4. (新) 回歸頭 (FC layers)
        self.fc1 = nn.Linear(2048, 512) 
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn_fc2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, output_dim) 

    def forward(self, xyz):
        B, C, N = xyz.shape
        if self.normal_channel:
            l0_points = xyz           # (B, 6, N)
            l0_xyz = xyz[:, :3, :]    # (B, 3, N)
        else:
            l0_points = xyz           # (B, 3, N)
            l0_xyz = xyz              # (B, 3, N)
        
        # 1. 階層式編碼
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points) 
        # l2_points shape: (B, 256, 128)

        # 2. (新) 特徵壓縮
        x = F.relu(self.bn_conv1(self.conv1(l2_points)))
        
        # 3. (新) 串聯池化
        x_max = self.pool_max(x).view(B, 1024) 
        x_avg = self.pool_avg(x).view(B, 1024) 
        x = torch.cat((x_max, x_avg), dim=1)  # (B, 2048)
        
        # 4. (新) 回歸頭
        x = self.drop1(F.relu(self.bn_fc1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn_fc2(self.fc2(x))))
        x = self.fc3(x)
        
        # 5. 輸出歸一化
        if x.shape[1] == 4: 
            x = F.normalize(x, p=2, dim=1) 
        elif x.shape[1] == 3 or x.shape[1] == 6: 
            x = torch.tanh(x)
        
        return x, None 


class get_loss(nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        
    def forward(self, pred, target, trans_feat):
        if pred.shape[1] == 4:
            dot_product = torch.sum(pred * target, dim=1)
            abs_dot_product = torch.abs(dot_product)
            loss_per_sample = 1.0 - torch.clamp(abs_dot_product, -1.0, 1.0)
            loss = torch.mean(loss_per_sample)
        else:
            loss = F.l1_loss(pred, target)
        
        return loss