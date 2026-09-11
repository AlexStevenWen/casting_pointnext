import torch
import torch.nn as nn
import torch.nn.functional as F
from models.pointnext_utils import fps_op, ball_op, gather

# --- 新增：DropPath (隨機深度) 用於防止過擬合 ---
class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        # 處理不同維度的輸入 (相容 1D/2D Conv)
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor

# --- 修改：加入 drop_path 參數 ---
class InvResMLP(nn.Module):
    def __init__(self, in_dim, expansion=4, radius=0.1, k=32, drop_path=0.0):
        super().__init__()
        self.radius = radius
        self.k = k
        inter_dim = in_dim * expansion

        self.sa_conv = nn.Sequential(
            nn.Conv2d(in_dim + 3, in_dim, 1, bias=False),
            nn.BatchNorm2d(in_dim),
            nn.ReLU(inplace=True)
        )

        self.mlp = nn.Sequential(
            nn.Conv1d(in_dim, inter_dim, 1, bias=False),
            nn.BatchNorm1d(inter_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(inter_dim, in_dim, 1, bias=False),
            nn.BatchNorm1d(in_dim)
        )
        
        # 應用 DropPath
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x, xyz):
        identity = x
        n_idx = ball_op(self.radius, self.k, xyz.transpose(1, 2), xyz.transpose(1, 2))
        neighbor_xyz = (gather(xyz, n_idx) - xyz.unsqueeze(-1)) / self.radius
        neighbor_feat = torch.cat([gather(identity, n_idx), neighbor_xyz], dim=1)
        
        x = self.sa_conv(neighbor_feat).max(dim=-1)[0]
        x = self.mlp(x)
        
        # 修改：在殘差相加前經過 DropPath
        return self.act(self.drop_path(x) + identity)

# --- 未修改：SABlock 保持原樣 ---
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
        idx = fps_op(xyz.transpose(1, 2), N // self.stride)
        new_xyz = xyz.gather(2, idx.unsqueeze(1).repeat(1, 3, 1))
        new_x_skip = x.gather(2, idx.unsqueeze(1).repeat(1, x.shape[1], 1))

        n_idx = ball_op(self.radius, self.k, xyz.transpose(1, 2), new_xyz.transpose(1, 2))
        neighbor_xyz = (gather(xyz, n_idx) - new_xyz.unsqueeze(-1)) / self.radius
        neighbor_feat = torch.cat([gather(x, n_idx), neighbor_xyz], dim=1)
        
        x = self.convs(neighbor_feat).max(dim=-1)[0]
        return x + self.skip(new_x_skip), new_xyz

# --- 修改：主模型加入 DropPath Rate 計算 ---
class get_model(nn.Module):
    def __init__(self, num_class, normal_channel=False, drop_path_rate=0.3):
        super().__init__()
        self.feat_dim = 4 if normal_channel else 1 
        
        dims = [32, 64, 128, 256, 512]
        blocks = [1, 2, 1, 1] 
        strides = [4, 4, 4, 4]
        radii = [0.1, 0.2, 0.4, 0.8]

        # 計算隨深度遞增的 DropPath 機率
        total_blocks = sum(blocks) - len(blocks) # 扣除 SABlock
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]
        dp_cnt = 0

        self.stem = nn.Sequential(
            nn.Conv1d(self.feat_dim, dims[0], 1, bias=False),
            nn.BatchNorm1d(dims[0]),
            nn.ReLU(inplace=True)
        )

        self.stages = nn.ModuleList()
        for i in range(len(blocks)):
            stage = nn.ModuleList()
            stage.append(SABlock(dims[i], dims[i+1], stride=strides[i], radius=radii[i]))
            
            # 只有 InvResMLP 使用 DropPath
            for _ in range(blocks[i] - 1):
                stage.append(InvResMLP(dims[i+1], radius=radii[i], drop_path=dpr[dp_cnt]))
                dp_cnt += 1
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

    def forward(self, pred, target, trans_feat=None):
        eps = self.smoothing
        n_class = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, target.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        loss = -(one_hot * pred).sum(dim=1).mean()
        return loss