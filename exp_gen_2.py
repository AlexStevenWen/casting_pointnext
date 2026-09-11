import os
from itertools import product

# ==========================================
# 1. 通用基礎模板
# ==========================================
BASE_TEMPLATE = """import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction, PointNetFeaturePropagation

# ==========================================
# 0. STN3d (T-Net) 模組
# ==========================================
class STN3d(nn.Module):
    def __init__(self, channel):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        # 動態獲取當前 device 避免 CUDA 錯誤
        iden = torch.eye(3, dtype=torch.float32, device=x.device).view(1, 9).repeat(batchsize, 1)
        x = x + iden
        x = x.view(-1, 3, 3)
        return x

# ==========================================
# 1. MLP Block: __MLP_NAME__
# ==========================================
__MLP_CODE__

# ==========================================
# 2. Main Model: __SA_TYPE__ + __MLP_NAME__ + __STN_TYPE__ (__TASK_TYPE__)
# ==========================================
class get_model(nn.Module):
    def __init__(self, num_part, normal_channel=True, grid_num=4):
        super(get_model, self).__init__()
        
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel
        
__STN_INIT__

__ENCODER_DECODER_INIT__

        # Prediction Head
        self.head_mlp = nn.Sequential(
            nn.Conv1d(128, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),        
            nn.Conv1d(128, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.4),       
            nn.Conv1d(64, 1, 1)  
        )

    def forward(self, xyz):
        B, C, N = xyz.shape
        l0_xyz = xyz[:, :3, :]
        
        if self.normal_channel:
            l0_points = xyz 
        else:
            l0_points = l0_xyz

__STN_FORWARD__

__ENCODER_DECODER_FORWARD__

__FORWARD_HEAD__

__LOSS_CODE__
"""

# ==========================================
# 2. 任務專用程式碼 (包含消融實驗選項)
# ==========================================
TASK_CODES = {
    # --------------------------------------------------
    # 實驗組 A: 你優化的 回歸任務 (針對類別不平衡)
    # --------------------------------------------------
    "regress_custom": {
        "forward": """        x = self.head_mlp(l0_points)
        x = torch.sigmoid(x).squeeze(1) # [B, N]
        return x, trans""",
        "loss": """# ==========================================
# 3. Regression Loss (Region-Normalized MSE + Soft Dice)
# ==========================================
class get_loss(nn.Module):
    def __init__(self, bg_weight=0.2, threshold=0.1):
        super(get_loss, self).__init__()
        self.register_buffer('iden', torch.eye(3).view(1, 3, 3))
        self.threshold = threshold
        self.bg_weight = bg_weight 

    def forward(self, pred, target, trans_feat, epoch):
        if pred.dim() == 3: pred = pred.squeeze(-1)
        if target.dim() == 3: target = target.squeeze(-1)
        target = target.float()
        diff_sq = (pred - target) ** 2

        fg_mask = (target > self.threshold).float()
        bg_mask = 1.0 - fg_mask

        fg_sum = fg_mask.sum(dim=1).clamp(min=1e-8)
        bg_sum = bg_mask.sum(dim=1).clamp(min=1e-8)

        fg_loss = (diff_sq * fg_mask).sum(dim=1) / fg_sum
        bg_loss = (diff_sq * bg_mask).sum(dim=1) / bg_sum

        norm_mse_loss = (fg_loss + self.bg_weight * bg_loss).mean()

        intersection = (pred * target).sum(dim=1)
        union = (pred ** 2).sum(dim=1) + (target ** 2).sum(dim=1) 
        soft_dice = 1.0 - (2.0 * intersection + 1e-5) / (union + 1e-5)
        soft_dice_loss = soft_dice.mean()

        main_loss = norm_mse_loss + 0.5 * soft_dice_loss

        if trans_feat is not None:
            batch_size = trans_feat.size(0)
            iden = self.iden.repeat(batch_size, 1, 1)
            trans_ortho = torch.bmm(trans_feat, trans_feat.transpose(1, 2))
            stn_loss = torch.norm(trans_ortho - iden, p='fro')
        else:
            stn_loss = 0.0

        return 10.0 * main_loss + 0.1 * stn_loss"""
    },
    # --------------------------------------------------
    # 對照組 A: 原始 PointNeXt 回歸任務 (單純 MSE)
    # --------------------------------------------------
    "regress_orig": {
        "forward": """        x = self.head_mlp(l0_points)
        x = torch.sigmoid(x).squeeze(1) # [B, N]
        return x, trans""",
        "loss": """# ==========================================
# 3. Original Regression Loss (Standard MSE)
# ==========================================
class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()
        self.register_buffer('iden', torch.eye(3).view(1, 3, 3))
        self.mse_loss = nn.MSELoss()

    def forward(self, pred, target, trans_feat, epoch):
        if pred.dim() == 3: pred = pred.squeeze(-1)
        if target.dim() == 3: target = target.squeeze(-1)
        target = target.float()
        
        main_loss = self.mse_loss(pred, target)

        if trans_feat is not None:
            batch_size = trans_feat.size(0)
            iden = self.iden.repeat(batch_size, 1, 1)
            trans_ortho = torch.bmm(trans_feat, trans_feat.transpose(1, 2))
            stn_loss = torch.norm(trans_ortho - iden, p='fro')
        else:
            stn_loss = 0.0

        return 10.0 * main_loss + 0.1 * stn_loss"""
    },
    # --------------------------------------------------
    # 實驗組 B: 你優化的 分割任務 (Dice + Focal)
    # --------------------------------------------------
    "seg_custom": {
        "forward": """        x = self.head_mlp(l0_points)
        x = torch.sigmoid(x).squeeze(1) # [B, N]
        return x, trans""",
        "loss": """# ==========================================
# 3. Segmentation Loss (Dice + Focal BCE)
# ==========================================
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        intersection = (pred * target).sum(dim=1)
        union = pred.sum(dim=1) + target.sum(dim=1)
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        pred = torch.clamp(pred, min=1e-7, max=1.0 - 1e-7)
        bce = - (target * torch.log(pred) + (1.0 - target) * torch.log(1.0 - pred))
        pt = target * pred + (1.0 - target) * (1.0 - pred)
        focal_loss = self.alpha * (1.0 - pt) ** self.gamma * bce
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()
        self.register_buffer('iden', torch.eye(3).view(1, 3, 3))
        self.dice_loss = DiceLoss()
        self.bce_loss = FocalLoss(alpha=0.5, gamma=2.0) 

    def forward(self, pred, target, trans_feat, epoch):
        target = target.float()
        loss_dice = self.dice_loss(pred, target)
        loss_bce = self.bce_loss(pred, target)
        total_seg_loss = loss_dice + loss_bce

        if trans_feat is not None:
            batch_size = trans_feat.size(0)
            iden = self.iden.repeat(batch_size, 1, 1)
            trans_ortho = torch.bmm(trans_feat, trans_feat.transpose(1, 2))
            stn_loss = torch.norm(trans_ortho - iden, p='fro')
        else:
            stn_loss = 0.0

        return total_seg_loss + 0.001 * stn_loss"""
    },
    # --------------------------------------------------
    # 對照組 B: 原始 PointNeXt 分割任務 (單純 BCE)
    # --------------------------------------------------
    "seg_orig": {
        "forward": """        x = self.head_mlp(l0_points)
        x = torch.sigmoid(x).squeeze(1) # [B, N]
        return x, trans""",
        "loss": """# ==========================================
# 3. Original Segmentation Loss (Standard BCE)
# ==========================================
class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()
        self.register_buffer('iden', torch.eye(3).view(1, 3, 3))
        self.bce_loss = nn.BCELoss() 

    def forward(self, pred, target, trans_feat, epoch):
        target = target.float()
        main_loss = self.bce_loss(pred, target)

        if trans_feat is not None:
            batch_size = trans_feat.size(0)
            iden = self.iden.repeat(batch_size, 1, 1)
            trans_ortho = torch.bmm(trans_feat, trans_feat.transpose(1, 2))
            stn_loss = torch.norm(trans_ortho - iden, p='fro')
        else:
            stn_loss = 0.0

        return main_loss + 0.001 * stn_loss"""
    }
}

# ==========================================
# 3. MLP 類別實作 (保持不變)
# ==========================================
MLP_CODES = {
    "bottleneckmlp": """class BottleneckMLP(nn.Module):
    def __init__(self, in_channel, mlp):
        super(BottleneckMLP, self).__init__()
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
        return F.relu(x + residual)""",

    "invertedresidualmlp": """class InvertedResidualMLP(nn.Module):
    def __init__(self, in_channel, expansion_ratio=4):
        super(InvertedResidualMLP, self).__init__()
        hidden_dim = in_channel * expansion_ratio
        self.conv1 = nn.Conv1d(in_channel, hidden_dim, 1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.conv3 = nn.Conv1d(hidden_dim, in_channel, 1)
        self.bn3 = nn.BatchNorm1d(in_channel)
    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        return F.relu(x + residual)""",

    "increasingmlp": """class IncreasingMLP(nn.Module):
    def __init__(self, in_channel, mlp):
        super(IncreasingMLP, self).__init__()
        self.conv1 = nn.Conv1d(in_channel, mlp[0], 1)
        self.bn1 = nn.BatchNorm1d(mlp[0])
        self.conv2 = nn.Conv1d(mlp[0], mlp[1], 1)
        self.bn2 = nn.BatchNorm1d(mlp[1])
        self.conv3 = nn.Conv1d(mlp[1], in_channel, 1) 
        self.bn3 = nn.BatchNorm1d(in_channel)
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return x"""
}

# ==========================================
# 4. 動態建構 Encoder/Decoder (保持不變)
# ==========================================
def get_encoder_decoder_code(sa_type, mlp_type):
    if mlp_type == "invertedresidualmlp":
        mlp_inst = lambda inc, m1, m2: f"InvertedResidualMLP({inc}, expansion_ratio=4)"
    elif mlp_type == "bottleneckmlp":
        mlp_inst = lambda inc, m1, m2: f"BottleneckMLP({inc}, [{m1}, {m2}])"
    else:
        mlp_inst = lambda inc, m1, m2: f"IncreasingMLP({inc}, [{m1}, {m2}])"

    if sa_type == "msg":
        init_code = f"""        # --- Encoder (MSG) ---
        self.sa1 = PointNetSetAbstractionMsg(1024, [0.03, 0.06], [16, 32], in_channel, [[32, 32, 64], [64, 64, 128]])
        self.res1 = {mlp_inst(192, 64, 128)}
        self.sa2 = PointNetSetAbstractionMsg(256, [0.1, 0.2], [32, 64], 192, [[128, 128, 256], [128, 196, 256]])
        self.res2 = {mlp_inst(512, 128, 256)}
        self.sa3 = PointNetSetAbstractionMsg(64, [0.3, 0.6], [64, 128], 512, [[256, 512, 512], [256, 512, 512]])
        self.res3 = {mlp_inst(1024, 256, 512)}
        # --- Decoder ---
        self.fp3 = PointNetFeaturePropagation(1536, [512, 512])
        self.fp2 = PointNetFeaturePropagation(704, [256, 256])
        self.fp1 = PointNetFeaturePropagation(256 + in_channel, [128, 128, 128])"""
        
        forward_code = """        # Encoder
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l1_points = self.res1(l1_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l2_points = self.res2(l2_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l3_points = self.res3(l3_points)
        # Decoder
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, l0_points, l1_points)"""

    elif sa_type == "ssg":
        init_code = f"""        # --- Encoder (SSG) ---
        self.sa1 = PointNetSetAbstraction(1024, 0.1, 32, in_channel + 3, [32, 32, 64], False)
        self.res1 = {mlp_inst(64, 32, 48)}
        self.sa2 = PointNetSetAbstraction(256, 0.2, 32, 64 + 3, [64, 64, 128], False)
        self.res2 = {mlp_inst(128, 64, 96)}
        self.sa3 = PointNetSetAbstraction(64, 0.4, 32, 128 + 3, [128, 128, 256], False)
        self.res3 = {mlp_inst(256, 128, 192)}
        # --- Decoder ---
        self.fp3 = PointNetFeaturePropagation(384, [256, 256])
        self.fp2 = PointNetFeaturePropagation(320, [128, 128])
        self.fp1 = PointNetFeaturePropagation(128 + in_channel, [128, 128, 128])"""
        
        forward_code = """        # Encoder
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l1_points = self.res1(l1_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l2_points = self.res2(l2_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l3_points = self.res3(l3_points)
        # Decoder
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, l0_points, l1_points)"""

    else: # mrg
        init_code = f"""        # --- Encoder (MRG 變體) ---
        self.sa1 = PointNetSetAbstractionMsg(1024, [0.05, 0.1], [16, 32], in_channel, [[32, 32, 64], [64, 64, 128]])
        self.res1 = {mlp_inst(192, 64, 128)}
        self.sa2 = PointNetSetAbstractionMsg(256, [0.1, 0.2], [32, 64], 192, [[128, 128, 256], [128, 196, 256]])
        self.sa2_mrg = PointNetSetAbstractionMsg(256, [0.2, 0.4], [16, 32], in_channel, [[64, 64, 128], [64, 128, 256]])
        self.res2 = {mlp_inst(896, 256, 512)}
        self.sa3 = PointNetSetAbstractionMsg(64, [0.3, 0.6], [64, 128], 896, [[256, 512, 512], [256, 512, 512]])
        self.res3 = {mlp_inst(1024, 256, 512)}
        # --- Decoder ---
        self.fp3 = PointNetFeaturePropagation(1920, [512, 512])
        self.fp2 = PointNetFeaturePropagation(704, [256, 256])
        self.fp1 = PointNetFeaturePropagation(256 + in_channel, [128, 128, 128])"""
        
        forward_code = """        # Encoder (MRG 雙路徑)
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l1_points = self.res1(l1_points)
        
        l2_xyz, l2_points_msg = self.sa2(l1_xyz, l1_points)
        _, l2_points_mrg = self.sa2_mrg(l0_xyz, l0_points)
        l2_points = torch.cat([l2_points_msg, l2_points_mrg], dim=1)
        l2_points = self.res2(l2_points)
        
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l3_points = self.res3(l3_points)
        
        # Decoder
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, l0_points, l1_points)"""

    return init_code, forward_code

# ==========================================
# 5. 生成與存檔邏輯 (擴增至 72 種組合供消融實驗)
# ==========================================
stn_options = ["stn", "nostn"]
sa_options = ["msg", "ssg", "mrg"]
mlp_options = ["bottleneckmlp", "invertedresidualmlp", "increasingmlp"]
# 新增消融選項
task_options = ["regress_custom", "regress_orig", "seg_custom", "seg_orig"] 

output_dir = "models"
os.makedirs(output_dir, exist_ok=True)

count = 0
total_models = len(stn_options) * len(sa_options) * len(mlp_options) * len(task_options)

for task, stn, sa, mlp in product(task_options, stn_options, sa_options, mlp_options):
    filename = f"exp_pointnext_part_seg_nocls_{sa}_{mlp}_{stn}_{task}.py"
    
    mlp_code = MLP_CODES[mlp]
    init_code, forward_code = get_encoder_decoder_code(sa, mlp)
    
    if stn == "stn":
        stn_init = "        self.stn = STN3d(in_channel)"
        stn_forward = """        trans = self.stn(l0_points)
        l0_xyz = torch.bmm(l0_xyz.transpose(1, 2), trans).transpose(1, 2)"""
    else:
        stn_init = "        # No STN initialized"
        stn_forward = "        trans = None # No STN transformation"

    task_forward = TASK_CODES[task]["forward"]
    task_loss = TASK_CODES[task]["loss"]

    final_code = BASE_TEMPLATE.replace("__MLP_NAME__", mlp.upper())
    final_code = final_code.replace("__MLP_CODE__", mlp_code)
    final_code = final_code.replace("__SA_TYPE__", sa.upper())
    final_code = final_code.replace("__STN_TYPE__", stn.upper())
    final_code = final_code.replace("__TASK_TYPE__", task.upper())
    final_code = final_code.replace("__STN_INIT__", stn_init)
    final_code = final_code.replace("__STN_FORWARD__", stn_forward)
    final_code = final_code.replace("__ENCODER_DECODER_INIT__", init_code)
    final_code = final_code.replace("__ENCODER_DECODER_FORWARD__", forward_code)
    final_code = final_code.replace("__FORWARD_HEAD__", task_forward)
    final_code = final_code.replace("__LOSS_CODE__", task_loss)

    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(final_code)
        
    count += 1
    print(f"[{count}/{total_models}] 生成成功: {filename}")

print(f"\n🎉 完美！共 {total_models} 個模型腳本已全數生成於 '{output_dir}' 資料夾中！現在可以開始跑消融實驗了。")