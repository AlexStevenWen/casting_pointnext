import torch
import torch.nn as nn
import torch.nn.functional as F

# (*** 關鍵 ***)
# 我們使用您修改過的、修復了 Bug 的 "旋轉敏感" Utils
from models.pointnet2_reg_utils import PointNetSetAbstraction, farthest_point_sample, index_points

"""
(新) 真正的 PVN3D 模型 (v2)

1. PointNet++ 骨幹 (sa1, sa2) 提取逐點特徵 (128 個點)
2. 投票頭 (Voting Head) 讓 128 個點投票
3. 投票聚合 (Vote Aggregation) - 使用 FPS 找到 8 個聚類中心
4. 姿態網路 (Pose-Net) - 在 8 個聚類中心上運行 SA 模組
5. 姿態回歸 (Pose Regression) - 從 8 個姿態特徵回歸 8 個關鍵點
"""

class get_model(nn.Module):
    def __init__(self, output_dim, normal_channel=True): 
        super(get_model, self).__init__()
        
        if output_dim != 8:
            raise ValueError("PVN3D 'output_dim' 必須是 8 (代表 8 個關鍵點)")
        self.num_keypoints = 8 
        
        in_channel_l0_points = 6 if normal_channel else 3
        self.normal_channel = normal_channel
        
        # --- 1. PointNet++ 骨幹 (Encoder) ---
        sa1_in_channel = in_channel_l0_points + 3
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=sa1_in_channel, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        
        # --- 2. 投票頭 (Voting Head) ---
        # (每個點 (256 維特徵) 將投票給 1 個關鍵點 "中心")
        # 總輸出 = 256 -> 1*3 = 3 (xyz 偏移量)
        self.vote_head = nn.Sequential(
            nn.Conv1d(256, 128, 1), # (B, 256, 128)
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 3, 1) # (B, 3, 128)
        )
        
        # --- 3. 姿態網路 (Pose-Net) ---
        # (這是一個在 "8 個聚類中心" 上運行的 PointNet++)
        # 輸入頻道: 3 (xyz) + 256 (來自 sa2 的原始特徵)
        self.pose_net_sa1 = PointNetSetAbstraction(npoint=8, radius=0.4, nsample=16, in_channel=256 + 3, mlp=[256, 256, 512], group_all=False)
        self.pose_net_sa2 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=512 + 3, mlp=[512, 1024, 1024], group_all=True)

        # --- 4. 姿態回歸 (Pose Regression Head) ---
        # (從 1024 維的全域姿態特徵 回歸 8 個關鍵點)
        self.fc_head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.num_keypoints * 3) # 8*3 = 24
        )
            
    def forward(self, xyz):
        # --- 1. 骨幹 (Encoder) ---
        B, C, N = xyz.shape
        if self.normal_channel:
            l0_points = xyz
            l0_xyz = xyz[:, :3, :]
        else:
            l0_points = xyz
            l0_xyz = xyz
        
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points) 
        # l2_xyz: (B, 3, 128) - 128 個點的座標
        # l2_points: (B, 256, 128) - 128 個點的特徵
        
        # --- 2. 投票 ---
        # (B, 256, 128) -> (B, 3, 128)
        votes_xyz_offset = self.vote_head(l2_points) 
        # (B, 3, 128) -> (B, 128, 3)
        votes_xyz_offset = votes_xyz_offset.transpose(1, 2)
        
        # (B, 3, 128) -> (B, 128, 3)
        l2_xyz_transposed = l2_xyz.transpose(1, 2)
        
        # 投票的絕對座標
        # (B, 128, 3) + (B, 128, 3) -> (B, 128, 3)
        voted_xyz = l2_xyz_transposed + votes_xyz_offset
        
        # --- 3. 投票聚合 (Vote Aggregation) ---
        # 我們使用 FPS 從 128 個 "投票" 中選出 8 個 "聚類中心"
        # voted_xyz: (B, 128, 3)
        # l2_points: (B, 256, 128) -> (B, 128, 256)
        voted_features = l2_points.transpose(1, 2)
        
        # (*** 關鍵 ***)
        # 我們的 "Pose-Net" 需要 (xyz, features)
        # 我們使用 8 個聚類中心的 xyz (vote_centers_xyz)
        # 和 8 個聚類中心的 features (vote_centers_features)
        
        # (B, 128, 3) -> (B, 8)
        vote_centers_idx = farthest_point_sample(voted_xyz, self.num_keypoints)
        
        # (B, 8, 3)
        vote_centers_xyz = index_points(voted_xyz, vote_centers_idx)
        # (B, 8, 256)
        vote_centers_features = index_points(voted_features, vote_centers_idx)
        
        # --- 4. 姿態網路 (Pose-Net) ---
        # (將格式轉回 (B, C, N))
        # (B, 3, 8)
        vote_centers_xyz = vote_centers_xyz.transpose(1, 2)
        # (B, 256, 8)
        vote_centers_features = vote_centers_features.transpose(1, 2)
        
        # (sa1) (B, 256+3, 8) -> (B, 512, 8)
        pose_xyz, pose_features = self.pose_net_sa1(vote_centers_xyz, vote_centers_features)
        # (sa2) (B, 512+3, 8) -> (B, 1024, 1)
        pose_xyz_global, pose_features_global = self.pose_net_sa2(pose_xyz, pose_features)
        
        # (B, 1024, 1) -> (B, 1024)
        global_pose_feature = pose_features_global.view(B, 1024)

        # --- 5. 姿態回歸 ---
        # (B, 1024) -> (B, 24)
        pred_keypoints_flat = self.fc_head(global_pose_feature)
        
        # (B, 24) -> (B, 8, 3)
        pred_keypoints = pred_keypoints_flat.view(B, self.num_keypoints, 3)
        
        return pred_keypoints, None