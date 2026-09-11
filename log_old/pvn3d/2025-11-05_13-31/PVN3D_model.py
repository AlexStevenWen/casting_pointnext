import torch
import torch.nn as nn
import torch.nn.functional as F

# (*** 關鍵 ***)
# 我們使用您修改過的、修復了 Bug 的 "旋轉敏感" Utils
from models.pointnet2_reg_utils import PointNetSetAbstraction

"""
PVN3D 簡化模型 (僅投票 + 姿態)

1. PointNet++ 骨幹 (sa1, sa2) 提取逐點特徵
2. 投票頭 (Voting Head) 讓每個點預測 8 個關鍵點的位置
3. 我們對每個關鍵點的「選票」進行平均，得到 8 個預測的關鍵點
"""

class get_model(nn.Module):
    def __init__(self, output_dim, normal_channel=True): 
        super(get_model, self).__init__()

        # (我們假設 output_dim 是 8, 代表 8 個關鍵點)
        self.num_keypoints = 8 

        in_channel_l0_points = 6 if normal_channel else 3
        self.normal_channel = normal_channel

        # --- 1. PointNet++ 骨幹 (Encoder) ---
        # (這是來自 pointnet2_reg_unet 的 "Encoder" 部分)
        sa1_in_channel = in_channel_l0_points + 3
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=sa1_in_channel, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)

        # --- 2. 投票頭 (Voting Head) ---
        # (我們在 sa2 輸出的 128 個特徵點上進行投票)
        # 每個點 (256 維特徵) 將投票給 8 個關鍵點
        # 每次投票包含 3 個值 (x, y, z 偏移量)
        # 總輸出 = 256 -> 8*3 = 24

        self.vote_head = nn.Sequential(
            nn.Conv1d(256, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, self.num_keypoints * 3, 1) # (B, 24, 128)
        )

    def forward(self, xyz):
        # --- 1. 骨幹 (Encoder) ---
        B, C, N = xyz.shape
        if self.normal_channel:
            l0_points = xyz           # (B, 6, N)
            l0_xyz = xyz[:, :3, :]    # (B, 3, N)
        else:
            l0_points = xyz           # (B, 3, N)
            l0_xyz = xyz              # (B, 3, N)

        # (sa1 接收 l0_xyz(3) 和 l0_points(3 or 6)) -> 輸出 l1_points(128)
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        # (sa2 接收 l1_xyz(3) 和 l1_points(128)) -> 輸出 l2_points(256)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points) 
        # l2_xyz: (B, 3, 128) - 128 個點的座標
        # l2_points: (B, 256, 128) - 128 個點的特徵

        # --- 2. 投票 ---
        # (B, 256, 128) -> (B, 24, 128)
        votes_xyz_offset = self.vote_head(l2_points) 

        # (B, 24, 128) -> (B, 128, 24) -> (B, 128, 8, 3)
        votes_xyz_offset = votes_xyz_offset.transpose(1, 2).view(B, 128, self.num_keypoints, 3)

        # (B, 3, 128) -> (B, 128, 3) -> (B, 128, 1, 3)
        l2_xyz_transposed = l2_xyz.transpose(1, 2).unsqueeze(2)

        # 每個點的絕對座標 + 每個點的偏移量 = 投票的絕對座標
        # (B, 128, 1, 3) + (B, 128, 8, 3) -> (B, 128, 8, 3)
        voted_keypoints = l2_xyz_transposed + votes_xyz_offset

        # --- 3. 姿態回歸 (Pose Regression) ---
        # 我們使用最簡單的池化：對所有 128 個點的投票取平均
        # (B, 128, 8, 3) -> (B, 8, 3)
        pred_keypoints = torch.mean(voted_keypoints, dim=1)

        # (返回預測的關鍵點)
        # (我們也返回 "None" 以匹配訓練腳本的 (pred, trans_feat) 格式)
        return pred_keypoints, None