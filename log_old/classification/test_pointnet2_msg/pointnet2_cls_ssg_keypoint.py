import torch.nn as nn
import torch.nn.functional as F
# 1. 確保引用的是你特化過的 keypoint 版本工具包
from pointnet2_utils_keypoint import PointNetSetAbstraction


class get_model(nn.Module):
    def __init__(self, num_class, normal_channel=False):
        super(get_model, self).__init__()
        
        # 2. 修改 in_channel 邏輯
        # PointNet++ 的 in_channel 指的是「進入 MLP 的總維度」
        # 在 SSG 中，第一層 SA 的輸入維度是：空間坐標(3) + 額外特徵(C)
        # 如果輸入是 XYZ + SegID (4維)，則 C = 1，in_channel = 3 + 1 = 4
        # 如果輸入是 XYZ + Normal + SegID (7維)，則 C = 4，in_channel = 3 + 4 = 7
        
        if normal_channel:
            # XYZ (3) + Normal (3) + SegID (1) = 7
            in_channel = 7
        else:
            # XYZ (3) + SegID (1) = 4
            in_channel = 4
            
        self.normal_channel = normal_channel
        
        # SA1: 處理原始點雲，將 4(或7) 維特徵映射到 128 維
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128], group_all=False)
        
        # SA2: 輸入維度 = SA1輸出(128) + 空間坐標(3) = 131
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        
        # SA3: 輸入維度 = SA2輸出(256) + 空間坐標(3) = 259
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, num_class)

    def forward(self, xyz):
        B, C, N = xyz.shape
        
        # 3. 在 forward 中正確拆分空間坐標與特徵
        # 為了配合 pointnet2_utils_keypoint.py 裡的邏輯：
        # 第一個參數是用於 FPS 和 Ball Query 的坐標
        # 第二個參數是附加在點上的特徵 (points)
        
        if self.normal_channel:
            # 假設輸入 [x, y, z, nx, ny, nz, seg]
            norm = xyz[:, 3:, :] # 包含 Normal (3) + Seg (1)
            xyz_coords = xyz[:, :3, :]
        else:
            # 假設輸入 [x, y, z, seg]
            norm = xyz[:, 3:, :] # 僅包含 Seg (1)
            xyz_coords = xyz[:, :3, :]

        # 逐層提取特徵
        l1_xyz, l1_points = self.sa1(xyz_coords, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        
        # 最後分類層
        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, -1)

        return x, l3_points


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)
        return total_loss