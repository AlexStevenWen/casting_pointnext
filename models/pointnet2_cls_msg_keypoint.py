import torch.nn as nn
import torch.nn.functional as F
# 1. 確保引用的是你剛才修改過的 keypoint 版本工具包
from pointnet2_utils_keypoint import PointNetSetAbstractionMsg, PointNetSetAbstraction


class get_model(nn.Module):
    def __init__(self, num_class, normal_channel=False):
        super(get_model, self).__init__()
        
        # 2. 修改 in_channel 邏輯
        # 你的輸入現在有 4 個通道 (XYZ + SegID)
        # PointNet++ 內部會將 XYZ (3維) 傳給採樣層，其餘維度作為特徵 (points)
        # 如果 normal_channel 為 True，輸入可能是 7 通道 (XYZ + Normal + SegID)
        if normal_channel:
            # 額外特徵包含 Normal(3) + SegID(1) = 4
            additional_feature_channels = 4 
        else:
            # 額外特徵只有 SegID(1) = 1
            additional_feature_channels = 1 
            
        self.normal_channel = normal_channel
        
        # SA1: in_channel 代表除了空間座標 XYZ 以外的特徵維度
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [16, 32, 128], 
                                              additional_feature_channels, 
                                              [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        
        # SA2: 320 是 SA1 三個分支輸出的總和 (64+128+128)
        self.sa2 = PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], 320, 
                                              [[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        
        # SA3: 640 是 SA2 三個分支輸出的總和 (128+256+256)，加上空間座標的 3 維
        self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 1024], True)
        
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, num_class)

    def forward(self, xyz):
        B, C, N = xyz.shape
        
        # 3. 在 forward 中正確拆分資料
        if self.normal_channel:
            # 假設輸入是 (B, 7, N) -> [x, y, z, nx, ny, nz, seg]
            additional_features = xyz[:, 3:, :] # 包含 Normal 和 SegID
            xyz_coords = xyz[:, :3, :]          # 僅包含空間座標
        else:
            # 假設輸入是 (B, 4, N) -> [x, y, z, seg]
            additional_features = xyz[:, 3:, :] # 僅包含 SegID
            xyz_coords = xyz[:, :3, :]          # 僅包含空間座標
            
        # 傳入 SA 層：第一參數是坐標 (用來採樣)，第二參數是特徵 (用來提取信息)
        l1_xyz, l1_points = self.sa1(xyz_coords, additional_features)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        
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
        # target 應該是 (B,) 的類別標籤
        total_loss = F.nll_loss(pred, target)
        return total_loss