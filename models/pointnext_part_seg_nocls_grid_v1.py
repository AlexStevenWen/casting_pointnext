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
    def __init__(self, num_part, normal_channel=True, grid_num=3):
        super(get_model, self).__init__()
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel
        self.grid_num = grid_num
        self.num_grid_out = grid_num ** 3 # 3x3x3 = 27

        # --- Encoder ---
        self.sa1 = PointNetSetAbstractionMsg(1024, [0.03, 0.06], [16, 32], in_channel, [[32, 32, 64], [64, 64, 128]])
        self.res1 = InvResMLP(192, [64, 128])

        self.sa2 = PointNetSetAbstractionMsg(256, [0.1, 0.2], [32, 64], 192, [[128, 128, 256], [128, 196, 256]])
        self.res2 = InvResMLP(512, [128, 256])

        self.sa3 = PointNetSetAbstractionMsg(64, [0.3, 0.6], [64, 128], 512, [[256, 512, 512], [256, 512, 512]])
        self.res3 = InvResMLP(1024, [256, 512])

        # --- 新增：Grid機率預測分支 (Grid Head) ---
        # 使用 SA3 提取的最深層全局特徵進行預測
        self.grid_head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.num_grid_out) # 輸出 27 格的 logits
        )

        # --- Decoder ---
        self.fp3 = PointNetFeaturePropagation(1536, [512, 512])
        self.fp2 = PointNetFeaturePropagation(704, [256, 256])
        self.fp1 = PointNetFeaturePropagation(256 + in_channel, [128, 128, 128])

        # 分類頭 (原本的點分割)
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_part, 1)

    def forward(self, xyz, label=None):
        B, C, N = xyz.shape
        l0_xyz = xyz[:, :3, :]
        l0_points = xyz

        # Encoder Stage
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l1_points = self.res1(l1_points)

        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l2_points = self.res2(l2_points)

        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l3_points = self.res3(l3_points) # B, 1024, 64

        # --- 提取全局特徵預測格子 ---
        # 對 64 個骨幹點做 Global Max Pooling
        global_feat = torch.max(l3_points, dim=2)[0] # B, 1024
        grid_logits = self.grid_head(global_feat) # B, 27

        # Decoder Stage (點分割)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, l0_points, l1_points)

        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        x = x.transpose(2, 1).contiguous()

        # 同時回傳點分割結果與格子預測結果
        return x, grid_logits

class get_loss(nn.Module):
    def __init__(self, weight=None):
        super(get_loss, self).__init__()
        self.weight = weight
        # 格子預測是多標籤分類，使用 BCE
        self.grid_criterion = nn.BCEWithLogitsLoss()

    def forward(self, pred, target, grid_pred, grid_target):
        # 1. 點分割損失
        seg_loss = F.nll_loss(pred, target, weight=self.weight, label_smoothing=0.1)
        
        # 2. 格子機率損失
        grid_loss = self.grid_criterion(grid_pred, grid_target)
        
        # 總損失 (可以調整 alpha 權重，這裡設定 1:1)
        return seg_loss + grid_loss