import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstractionMsg, PointNetFeaturePropagation

# 根據 PointNeXt 論文，核心在於 InvResMLP 模組
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

class get_model(nn.Module):
    def __init__(self, num_part, normal_channel=True):
        super(get_model, self).__init__()
        # PointNeXt 基礎輸入通道：XYZ(3) + Normal(3)
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel

        # --- Encoder: 遵循論文的 Stage-wise 下採樣 ---
        #self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2], [16, 32], in_channel, [[32, 32, 64], [64, 64, 128]])
        self.sa1 = PointNetSetAbstractionMsg(2048, [0.05, 0.1], [16, 32], in_channel, [[32, 32, 64], [64, 64, 128]])
        self.res1 = InvResMLP(192, [64, 64]) # 64+128 = 192

        self.sa2 = PointNetSetAbstractionMsg(128, [0.2, 0.4], [32, 64], 192, [[128, 128, 256], [128, 196, 256]])
        self.res2 = InvResMLP(512, [128, 128]) # 256+256 = 512

        self.sa3 = PointNetSetAbstractionMsg(32, [0.4, 0.8], [64, 128], 512, [[256, 512, 512], [256, 512, 512]])
        self.res3 = InvResMLP(1024, [256, 256]) # 512+512 = 1024

        # --- Decoder: Feature Propagation (FP) ---
        self.fp3 = PointNetFeaturePropagation(1536, [512, 512]) # 1024 + 512
        self.fp2 = PointNetFeaturePropagation(704, [256, 256])  # 512 + 192
        self.fp1 = PointNetFeaturePropagation(256 + in_channel, [128, 128, 128])

        # 最後分類層
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_part, 1)

    def forward(self, xyz, label=None): # NOCLS 版本忽略 label
        B, C, N = xyz.shape
        l0_xyz = xyz[:, :3, :]
        l0_points = xyz

        # Encoder Stage
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l1_points = self.res1(l1_points)

        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l2_points = self.res2(l2_points)

        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l3_points = self.res3(l3_points)

        # Decoder Stage
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, l0_points, l1_points)

        # Classification Head
        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        x = x.transpose(2, 1).contiguous()

        return x, l3_points

class get_loss(nn.Module):
    def __init__(self, weight=None):
        super(get_loss, self).__init__()
        self.weight = weight

    def forward(self, pred, target, trans_feat):
        # 針對鑄件任務，使用加權負對數似然損失
        total_loss = F.nll_loss(pred, target, weight=self.weight)
        return total_loss