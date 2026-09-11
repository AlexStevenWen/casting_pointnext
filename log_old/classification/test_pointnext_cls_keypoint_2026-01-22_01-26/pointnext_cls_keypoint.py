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
        # Grouping
        n_idx = ball_op(self.radius, self.k, xyz.transpose(1, 2), xyz.transpose(1, 2))
        neighbor_xyz = (gather(xyz, n_idx) - xyz.unsqueeze(-1)) / self.radius
        neighbor_feat = torch.cat([gather(identity, n_idx), neighbor_xyz], dim=1)
        
        # SA 提取局部特徵並 Max Pooling
        x = self.sa_conv(neighbor_feat).max(dim=-1)[0]
        
        # MLP + 殘差連接
        x = self.mlp(x)
        return self.act(x + identity)

# --- 原始論文核心：Set Abstraction Block (用於下採樣) ---
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
        # 殘差路徑
        self.skip = nn.Conv1d(in_dim, out_dim, 1, bias=False) if in_dim != out_dim or stride != 1 else nn.Identity()

    def forward(self, x, xyz):
        B, C, N = xyz.shape
        # 1. 下採樣 (FPS)
        idx = fps_op(xyz.transpose(1, 2), N // self.stride)
        new_xyz = xyz.gather(2, idx.unsqueeze(1).repeat(1, 3, 1))
        new_x_skip = x.gather(2, idx.unsqueeze(1).repeat(1, x.shape[1], 1))

        # 2. 空間特徵提取 (Ball Query)
        n_idx = ball_op(self.radius, self.k, xyz.transpose(1, 2), new_xyz.transpose(1, 2))
        neighbor_xyz = (gather(xyz, n_idx) - new_xyz.unsqueeze(-1)) / self.radius
        neighbor_feat = torch.cat([gather(x, n_idx), neighbor_xyz], dim=1)
        
        # 3. 卷積與 Max Pooling
        x = self.convs(neighbor_feat).max(dim=-1)[0]
        return x + self.skip(new_x_skip), new_xyz

# --- PointNeXt 主模型架構 ---
class get_model(nn.Module):
    def __init__(self, num_class, normal_channel=False):
        super().__init__()
        # Keypoint 版輸入 1 (SegID) 或 4 (Normal+SegID)
        self.feat_dim = 4 if normal_channel else 1 
        
        # 論文配置: PointNeXt-S (Small 版)
        # dims: 每層的寬度, blocks: 每層 InvResMLP 的個數
        dims = [32, 64, 128, 256, 512]
        blocks = [1, 2, 1, 1] 
        strides = [4, 4, 4, 4]
        radii = [0.1, 0.2, 0.4, 0.8]

        # Stem: 初始升維
        self.stem = nn.Sequential(
            nn.Conv1d(self.feat_dim, dims[0], 1, bias=False),
            nn.BatchNorm1d(dims[0]),
            nn.ReLU(inplace=True)
        )

        self.stages = nn.ModuleList()
        for i in range(len(blocks)):
            stage = nn.ModuleList()
            # 每個 Stage 第一個 Block 負責下採樣
            stage.append(SABlock(dims[i], dims[i+1], stride=strides[i], radius=radii[i]))
            # 後續接 InvResMLP (保持解析度不變)
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

    def forward(self, data):
        # data: [B, 4, N] -> [XYZ, SegID]
        xyz = data[:, :3, :]
        features = data[:, 3:, :]
        
        x = self.stem(features)
        
        for stage in self.stages:
            for block in stage:
                if isinstance(block, SABlock):
                    x, xyz = block(x, xyz)
                else:
                    x = block(x, xyz)
        
        x = self.head(x)
        return F.log_softmax(x, dim=-1), None


class get_loss(nn.Module):
    def __init__(self, smoothing=0.2):
        super(get_loss, self).__init__()
        self.smoothing = smoothing

    # 關鍵修正：加入 *args 或明確指定 trans_feat 參數
    # 即使我們不用 trans_feat，也要定義它來接收腳本傳過來的值
    def forward(self, pred, target, trans_feat=None):
        """
        Label Smoothing Cross Entropy Loss
        """
        eps = self.smoothing
        n_class = pred.size(1)
        
        # 將 target 轉換為 one-hot 並進行平滑處理
        one_hot = torch.zeros_like(pred).scatter(1, target.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        
        # pred 已經是 log_softmax，所以計算交叉熵
        loss = -(one_hot * pred).sum(dim=1).mean()
        return loss