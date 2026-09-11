import torch
import torch.nn as nn
import torch.nn.functional as F
# (注意：請確保您使用的是 pointnet2_utils.py 或您修改過的 pointnet2_reg_utils.py)
from models.pointnet2_utils import PointNetSetAbstraction, PointNetFeaturePropagation

"""
(新) U-Net 架構的 PointNet++ 姿態回歸模型
(v3) 簡化了 Regression Head，移除了 conv1(128, 1024) 瓶頸
"""

class get_model(nn.Module):
    def __init__(self, output_dim=4, normal_channel=True): 
        super(get_model, self).__init__()
        
        in_channel_l0_points = 6 if normal_channel else 3
        self.normal_channel = normal_channel
        
        # --- Encoder ---
        sa1_in_channel = in_channel_l0_points + 3
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=sa1_in_channel, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)

        # --- Decoder ---
        self.fp3 = PointNetFeaturePropagation(in_channel=1024 + 256, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=256 + 128, mlp=[256, 128])
        fp1_in_channel = 128 + 3 + in_channel_l0_points
        self.fp1 = PointNetFeaturePropagation(in_channel=fp1_in_channel, mlp=[128, 128])

        # --- (*** 這是 v3 的關鍵修改 ***) ---
        
        # --- Final Pooling & Regression Head ---
        
        # (*** 已移除: self.conv1 = nn.Conv1d(128, 1024, 1) ***)
        # (*** 已移除: self.bn_conv1 = nn.BatchNorm1d(1024) ***)
        
        self.pool_max = nn.AdaptiveMaxPool1d(1)
        self.pool_avg = nn.AdaptiveAvgPool1d(1)

        # (*** 關鍵修改 ***)
        # 我們現在直接在 128 維特徵上 Pooling
        # Max (128) + Avg (128) = 256
        self.fc1 = nn.Linear(256, 512) # (輸入從 2048 改為 256)
        
        # (後面的層保持不變)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.bn_fc2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, output_dim) 
        
        # --- (*** 修正結束 ***) ---


    def forward(self, xyz):
        # Set Abstraction layers (Encoder)
        B, C, N = xyz.shape
        if self.normal_channel:
            l0_points = xyz           # (B, 6, N)
            l0_xyz = xyz[:, :3, :]    # (B, 3, N)
        else:
            l0_points = xyz           # (B, 3, N)
            l0_xyz = xyz              # (B, 3, N)
        
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        # Feature Propagation layers (Decoder)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        
        l0_features_skipped = torch.cat([l0_xyz, l0_points], 1)
        l0_points = self.fp1(l0_xyz, l1_xyz, l0_features_skipped, l1_points)
        # U-Net 輸出: l0_points shape (B, 128, N)

        # --- (*** 這是 v3 的關鍵修改 ***) ---

        # --- Final Pooling & Regression ---
        
        # (*** 已移除: x = F.relu(self.bn_conv1(self.conv1(l0_points))) ***)
        
        # (*** 關鍵修改 ***)
        # 直接在 U-Net 輸出 l0_points (B, 128, N) 上進行 Pooling
        x = l0_points 
        
        x_max = self.pool_max(x).view(B, 128) # (維度從 1024 改為 128)
        x_avg = self.pool_avg(x).view(B, 128) # (維度從 1024 改為 128)
        x = torch.cat((x_max, x_avg), dim=1)  # (B, 256)
        
        # (後面的 FC 層保持不變)
        x = self.drop1(F.relu(self.bn_fc1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn_fc2(self.fc2(x))))
        x = self.fc3(x)
        
        # --- (*** 修正結束 ***) ---

        
        if x.shape[1] == 4: 
            # (*** 建議: 增加 epsilon 避免 F.normalize 除以零 ***)
            x = F.normalize(x, p=2, dim=1, eps=1e-8) # 四元數
        elif x.shape[1] == 3 or x.shape[1] == 6: 
            x = torch.tanh(x) # 欧拉角
        
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