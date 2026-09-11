import torch
import torch.nn as nn
import torch.nn.functional as F

# (*** 關鍵 ***)
# 我們使用您修改過的、修復了 Bug 的 "旋轉敏感" Utils
from models.pointnet2_reg_utils import PointNetSetAbstraction, farthest_point_sample, index_points

"""
(新) 真正的 PVN3D 模型 (v3)

1. PointNet++ 骨幹 (sa1, sa2)
2. 投票頭 (Voting Head)
3. 投票聚合 (Vote Aggregation) - FPS 找到 8 個聚類中心
4. 姿態網路 (Pose-Net sa1) - 提煉 8 個獨立的特徵
5. (*** 新 ***) 逐點回歸 (Per-Keypoint Head) - 直接在 8 個特徵上回歸
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
        self.vote_head = nn.Sequential(
            nn.Conv1d(256, 128, 1), 
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 3, 1) # (B, 3, 128)
        )
        
        # --- 3. 姿態網路 (Pose-Net) ---
        # (這是一個在 "8 個聚類中心" 上運行的 PointNet++)
        # 輸入頻道: 3 (xyz) + 256 (來自 sa2 的原始特徵)
        self.pose_net_sa1 = PointNetSetAbstraction(npoint=8, radius=0.4, nsample=16, in_channel=256 + 3, mlp=[256, 256, 512], group_all=False)

        # --- (*** 關鍵修正：移除 pose_net_sa2 和 fc_head ***) ---
        # self.pose_net_sa2 = PointNetSetAbstraction(..., group_all=True) <-- 移除
        # self.fc_head = nn.Sequential(...) <-- 移除

        # --- 4. (新) 逐點回歸頭 (Per-Keypoint Regression Head) ---
        # 我們直接在 pose_net_sa1 輸出的 8 個 512 維特徵上操作
        self.kp_head = nn.Sequential(
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 3, 1) # 輸出 8 個點的 (x, y, z) 偏移量
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
        # l2_xyz: (B, 3, 128) 
        # l2_points: (B, 256, 128)
        
        # --- 2. 投票 ---
        votes_xyz_offset = self.vote_head(l2_points) 
        votes_xyz_offset = votes_xyz_offset.transpose(1, 2) # (B, 128, 3)
        l2_xyz_transposed = l2_xyz.transpose(1, 2) # (B, 128, 3)
        voted_xyz = l2_xyz_transposed + votes_xyz_offset # (B, 128, 3)
        
        # --- 3. 投票聚合 (Vote Aggregation) ---
        voted_features = l2_points.transpose(1, 2) # (B, 128, 256)
        
        vote_centers_idx = farthest_point_sample(voted_xyz, self.num_keypoints) # (B, 8)
        vote_centers_xyz = index_points(voted_xyz, vote_centers_idx) # (B, 8, 3)
        vote_centers_features = index_points(voted_features, vote_centers_idx) # (B, 8, 256)
        
        # --- 4. 姿態網路 (Pose-Net) ---
        # (將格式轉回 (B, C, N))
        vote_centers_xyz_T = vote_centers_xyz.transpose(1, 2) # (B, 3, 8)
        vote_centers_features_T = vote_centers_features.transpose(1, 2) # (B, 256, 8)
        
        # (sa1) (B, 256+3, 8) -> (B, 512, 8)
        pose_xyz, pose_features = self.pose_net_sa1(vote_centers_xyz_T, vote_centers_features_T)
        # pose_features: (B, 512, 8) - 8 個點的 512 維獨立特徵
        
        # --- (*** 關鍵修正：移除 pose_net_sa2 和 fc_head ***) ---
        
        # --- 5. (新) 逐點回歸 ---
        # (B, 512, 8) -> (B, 3, 8)
        keypoint_offsets = self.kp_head(pose_features)
        
        # (B, 3, 8) -> (B, 8, 3)
        keypoint_offsets_T = keypoint_offsets.transpose(1, 2)
        
        # 最終預測 = 8 個聚類中心的座標 + 8 個預測的偏移量
        # (B, 8, 3) + (B, 8, 3) -> (B, 8, 3)
        pred_keypoints = vote_centers_xyz + keypoint_offsets_T
        
        return pred_keypoints, None