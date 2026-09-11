import torch
import torch.nn as nn
import torch.nn.functional as F
from models.pointnext_utils import fps_op, ball_op, gather

# --- 原始論文核心：Inverted Residual MLP Block ---
class InvResMLP(nn.Module):
    def __init__(self, in_dim, expansion=4, radius=0.1, k=32):
        super().__init__()
        self.radius = radius
        self.k = k
        inter_dim = in_dim * expansion

        # 1. 局部特徵提取 (Set Abstraction)
        self.sa_conv = nn.Sequential(
            nn.Conv2d(in_dim + 3, in_dim, 1, bias=False),
            nn.BatchNorm2d(in_dim),
            nn.ReLU(inplace=True)
        )

        # 2. 逐點 MLP (Inverted Residual)
        self.mlp = nn.Sequential(
            nn.Conv1d(in_dim, inter_dim, 1, bias=False),
            nn.BatchNorm1d(inter_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(inter_dim, in_dim, 1, bias=False),
            nn.BatchNorm1d(in_dim)
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x, xyz):
        identity = x
        # Grouping (只在當前解析度下進行)
        n_idx = ball_op(self.radius, self.k, xyz.transpose(1, 2), xyz.transpose(1, 2))
        neighbor_xyz = (gather(xyz, n_idx) - xyz.unsqueeze(-1)) / self.radius
        neighbor_feat = torch.cat([gather(identity, n_idx), neighbor_xyz], dim=1)
        
        x = self.sa_conv(neighbor_feat).max(dim=-1)[0]
        x = self.mlp(x)
        return self.act(x + identity)

# --- 原始論文核心：Set Abstraction Block (下採樣版) ---
class SABlock(nn.Module):
    def __init__(self, in_dim, out_dim, stride=4, radius=0.1, k=32):
        super().__init__()
        self.stride = stride
        self.radius = radius
        self.k = k

        self.convs = nn.Sequential(
            nn.Conv2d(in_dim + 3, out_dim, 1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )
        self.skip = nn.Conv1d(in_dim, out_dim, 1, bias=False) if in_dim != out_dim or stride != 1 else nn.Identity()

    def forward(self, x, xyz):
        B, C, N = xyz.shape
        # 1. 下採樣 (FPS)
        idx = fps_op(xyz.transpose(1, 2), N // self.stride)
        new_xyz = xyz.gather(2, idx.unsqueeze(1).repeat(1, 3, 1))
        new_x_skip = x.gather(2, idx.unsqueeze(1).repeat(1, x.shape[1], 1))

        # 2. 局部區域特徵提取
        n_idx = ball_op(self.radius, self.k, xyz.transpose(1, 2), new_xyz.transpose(1, 2))
        neighbor_xyz = (gather(xyz, n_idx) - new_xyz.unsqueeze(-1)) / self.radius
        neighbor_feat = torch.cat([gather(x, n_idx), neighbor_xyz], dim=1)
        
        x = self.convs(neighbor_feat).max(dim=-1)[0]
        return x + self.skip(new_x_skip), new_xyz

# --- 標準版 PointNeXt 主架構 ---
class get_model(nn.Module):
    def __init__(self, num_class, normal_channel=False):
        super().__init__()
        # 標準版輸入：如果 normal_channel 為 True，除了 XYZ 還有 Normal (3維)
        self.additional_channel = 3 if normal_channel else 0
        
        dims = [32, 64, 128, 256, 512]
        blocks = [1, 2, 1, 1] 
        strides = [4, 4, 4, 4]
        radii = [0.1, 0.2, 0.4, 0.8]

        # Stem: 將初始特徵升維。如果是純 XYZ，我們直接投影
        # 論文中 PointNeXt 對純坐標點雲通常也會在輸入端做一次線性變換
        self.stem = nn.Sequential(
            nn.Conv1d(self.additional_channel + 3, dims[0], 1, bias=False),
            nn.BatchNorm1d(dims[0]),
            nn.ReLU(inplace=True)
        )

        self.stages = nn.ModuleList()
        for i in range(len(blocks)):
            stage = nn.ModuleList()
            stage.append(SABlock(dims[i], dims[i+1], stride=strides[i], radius=radii[i]))
            for _ in range(blocks[i] - 1):
                stage.append(InvResMLP(dims[i+1], radius=radii[i]))
            self.stages.append(stage)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(dims[-1], 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_class)
        )

    def forward(self, xyz):
        # xyz shape: [B, 3, N] 或 [B, 6, N]
        coords = xyz[:, :3, :]
        
        # 將全體作為初始特徵 (包含座標資訊)
        x = self.stem(xyz)
        
        for stage in self.stages:
            for block in stage:
                if isinstance(block, SABlock):
                    x, coords = block(x, coords)
                else:
                    x = block(x, coords)
        
        x = self.head(x)
        return F.log_softmax(x, dim=-1), None

class get_loss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, pred, target, trans_feat):
        return F.nll_loss(pred, target)