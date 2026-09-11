import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_
from pointnet2_utils import PointNetFeaturePropagation # 沿用您原有的 FP 層

# --- 基礎組件 ---
class Encoder(nn.Module):
    def __init__(self, encoder_channel=384):
        super().__init__()
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Conv1d(512, encoder_channel, 1)
        )

    def forward(self, point_groups):
        # point_groups: B, G, N, 3
        bs, g, n, _ = point_groups.shape
        x = point_groups.reshape(bs * g, n, 3).transpose(2, 1) # BG, 3, N
        feature = self.first_conv(x) # BG, 256, N
        feature_global = torch.max(feature, dim=2, keepdim=True)[0] # BG, 256, 1
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1) # BG, 512, N
        feature = self.second_conv(feature) # BG, C, N
        feature_global = torch.max(feature, dim=2, keepdim=False)[0] # BG, C
        return feature_global.reshape(bs, g, -1)

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )
    def forward(self, x):
        # x: B, G, C
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x

# --- 主模型 ---
class get_model(nn.Module):
    def __init__(self, num_part, normal_channel=False):
        super().__init__()
        self.trans_dim = 384
        self.num_group = 128
        self.group_size = 32

        # 1. Embedding & Positional Encoding
        self.encoder = Encoder(encoder_channel=self.trans_dim)
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128), nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        # 2. Transformer Blocks (Encoder)
        self.blocks = nn.ModuleList([Block(self.trans_dim, 6) for _ in range(4)])
        self.norm = nn.LayerNorm(self.trans_dim)

        # 3. Segmentation Head (需要將 Group 特徵插值回原始點)
        # 這裡我們使用您原有的 FP 層邏輯來做 Upsampling
        self.fp1 = PointNetFeaturePropagation(self.trans_dim + 3, [256, 256])
        self.conv1 = nn.Conv1d(256, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_part, 1)

    def forward(self, xyz, label=None):
        # xyz: B, 3, N (假設已 normalize)
        B, C, N = xyz.shape
        points = xyz.transpose(2, 1).contiguous() # B, N, 3

        # 簡單的 Grouping (FPS + KNN)
        # 注意：這裡建議直接調用 pointnet2_utils 裡的 FPS
        from pointnet2_utils import sample_and_group_all, index_points, farthest_point_sample
        
        fps_idx = farthest_point_sample(points, self.num_group) 
        center = index_points(points, fps_idx) # B, G, 3
        
        # 獲取 neighborhood (簡化版：直接用 KNN)
        dist = torch.cdist(center, points) # B, G, N
        idx = dist.topk(self.group_size, dim=-1, largest=False)[1] # B, G, K
        neighborhood = index_points(points, idx) # B, G, K, 3
        neighborhood = neighborhood - center.unsqueeze(2) # Normalize to center

        # Transformer Encoder
        group_input_tokens = self.encoder(neighborhood) # B, G, C
        pos = self.pos_embed(center)
        x = group_input_tokens + pos
        for block in self.blocks:
            x = block(x)
        x = self.norm(x) # B, G, C (這是 Group-level 特徵)

        # Decoder: 將 G 個點的特徵傳回 N 個點
        # 使用 Feature Propagation 插值
        l0_points = self.fp1(xyz, center.transpose(2, 1), xyz, x.transpose(2, 1)) # B, 256, N

        # Classification Head
        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        x = x.transpose(2, 1).contiguous()
        return x, None