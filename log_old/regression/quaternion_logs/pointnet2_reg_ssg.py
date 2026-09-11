import torch
import torch.nn as nn
import torch.nn.functional as F
from models.pointnet2_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction

class get_model(nn.Module):
    def __init__(self, output_dim=4, normal_channel=True): # (預設 output_dim)
        super(get_model, self).__init__()
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel
        
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, output_dim) # <-- 輸出 N 維

    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
            
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        
        # --- (*** 關鍵修改 ***) ---
        if x.shape[1] == 4: # 如果是四元數模式
            # L2 歸一化，確保是單位四元數
            x = F.normalize(x, p=2, dim=1) 
        elif x.shape[1] == 3 or x.shape[1] == 6: # 如果是歐拉角模式
            # 使用 Tanh
            x = torch.tanh(x)
        # --- (*** 修改結束 ***) ---
        
        return x, None 


class get_loss(nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        
    def forward(self, pred, target, trans_feat):
        
        # --- (*** 關鍵修改：動態損失 ***) ---
        if pred.shape[1] == 4:
            # (模式 1: 四元數損失)
            # 1. 計算點積
            dot_product = torch.sum(pred * target, dim=1)
            # 2. 取絕對值 (因為 q 和 -q 相同)
            abs_dot_product = torch.abs(dot_product)
            # 3. 損失是 1.0 - |dot_product|
            loss_per_sample = 1.0 - torch.clamp(abs_dot_product, -1.0, 1.0)
            loss = torch.mean(loss_per_sample)
            
        else:
            # (模式 2: 歐拉角損失)
            loss = F.l1_loss(pred, target)
        # --- (*** 修改結束 ***) ---
        
        # (trans_feat 在 PointNet++ 中為 None，我們直接忽略它)
        return loss