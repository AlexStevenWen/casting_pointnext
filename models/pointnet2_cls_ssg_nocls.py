import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstraction, PointNetFeaturePropagation


class get_model(nn.Module):
    def __init__(self, num_class, normal_channel=True):
        super(get_model, self).__init__()
        # 輸入通道：XYZ (3) + 法向量 (3) = 6 或 只有 XYZ (3)
        if normal_channel:
            additional_channel = 3
        else:
            additional_channel = 0
        self.normal_channel = normal_channel
        in_channel = 6 if normal_channel else 3
        # --- Encoder (Set Abstraction 層) ---
        # SA1: 採樣 512 個點，半徑 0.2
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128], group_all=False)
        # SA2: 採樣 128 個點，半徑 0.4
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        # SA3: 全域特徵提取
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)

        # --- Decoder (Feature Propagation 層) ---
        # 將全域特徵還原回點級別特徵
        self.fp3 = PointNetFeaturePropagation(in_channel=1280, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=384, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=128 + additional_channel, mlp=[128, 128, 128])

        # 最終分類卷積層 (輸出維度為 num_class，即本體與澆口)
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_class, 1)

    def forward(self, xyz):
        l0_points = xyz
        l0_xyz = xyz[:, :3, :]

        # Encoder: 提取不同尺度的空間特徵
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        # Decoder: 插值還原並合併淺層特徵 (Skip Connections)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)

        # 預測輸出
        feat = F.relu(self.bn1(self.conv1(l0_points)))
        x = self.drop1(feat)
        x = self.conv2(x)
        
        # 轉換為 LogSoftmax 格式供 NLLLoss 使用
        x = F.log_softmax(x, dim=1)
        x = x.transpose(2, 1).contiguous()
        return x, l3_points


class get_loss(nn.Module):
    def __init__(self, weight=None):
        super(get_loss, self).__init__()
        self.weight = weight

    def forward(self, pred, target, trans_feat):
        # 支援權重加權，解決澆口點數過少的問題
        total_loss = F.nll_loss(pred, target, weight=self.weight)
        return total_loss