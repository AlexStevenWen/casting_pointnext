import os
from itertools import product

# ==========================================
# 1. 通用基礎模板
# ==========================================
BASE_TEMPLATE = """import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction, PointNetFeaturePropagation

try:
    from pointnet2_utils import STN3d
except ImportError:
    pass

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
# 2. 任務專用程式碼 (Regress vs Segmentation)
# ==========================================
TASK_CODES = {
"regress": {
        "forward": """        x = self.head_mlp(l0_points)
        x = torch.sigmoid(x).squeeze(1) # [B, N] 修改：與 Seg 保持一致的 squeeze，避免維度錯亂
        return x, trans""",
        "loss": """# ==========================================
# 3. Regression Loss (Region-Normalized MSE + Soft Dice)
# ==========================================
class get_loss(nn.Module):
    def __init__(self, bg_weight=0.2, threshold=0.1):
        super(get_loss, self).__init__()
        self.register_buffer('iden', torch.eye(3).view(1, 3, 3))
        self.threshold = threshold
        # 控制背景(主體) Loss 對總 Loss 的貢獻比例，0.2 代表澆口更重要
        self.bg_weight = bg_weight 

    def forward(self, pred, target, trans_feat, epoch):
        # 將維度對齊: pred 和 target 都確保是 [B, N]
        if pred.dim() == 3: pred = pred.squeeze(-1)
        if target.dim() == 3: target = target.squeeze(-1)
        
        target = target.float()
        diff_sq = (pred - target) ** 2

        # -----------------------------------------------------------
        # 1. 區域正規化 MSE (Region-Normalized MSE)
        # -----------------------------------------------------------
        # 動態區分「澆口(前景)」與「主體(背景)」
        fg_mask = (target > self.threshold).float()
        bg_mask = 1.0 - fg_mask

        # 分別計算前景與背景的點數 (加上 1e-8 避免分母為 0)
        fg_sum = fg_mask.sum(dim=1).clamp(min=1e-8)
        bg_sum = bg_mask.sum(dim=1).clamp(min=1e-8)

        # 計算「澆口區域內」的平均 MSE 與「主體區域內」的平均 MSE
        fg_loss = (diff_sq * fg_mask).sum(dim=1) / fg_sum
        bg_loss = (diff_sq * bg_mask).sum(dim=1) / bg_sum

        # 將兩者結合，無論點數比例多懸殊，權重都能強制拉平
        norm_mse_loss = (fg_loss + self.bg_weight * bg_loss).mean()

        # -----------------------------------------------------------
        # 2. 連續型 Dice Loss (Soft Dice for Regression)
        # -----------------------------------------------------------
        # 專門用來對付模型不敢預測高數值(澆口)的問題
        intersection = (pred * target).sum(dim=1)
        # 為了適應連續數值，分母使用平方和
        union = (pred ** 2).sum(dim=1) + (target ** 2).sum(dim=1) 
        soft_dice = 1.0 - (2.0 * intersection + 1e-5) / (union + 1e-5)
        soft_dice_loss = soft_dice.mean()

        # -----------------------------------------------------------
        # 3. 總結 Loss
        # -----------------------------------------------------------
        # 結合 MSE 的數值精準度與 Dice 的結構對齊能力
        main_loss = norm_mse_loss + 0.5 * soft_dice_loss

        # STN 損失
        if trans_feat is not None:
            batch_size = trans_feat.size(0)
            iden = self.iden.repeat(batch_size, 1, 1)
            trans_ortho = torch.bmm(trans_feat, trans_feat.transpose(1, 2))
            stn_loss = torch.norm(trans_ortho - iden, p='fro')
        else:
            stn_loss = 0.0

        return 10.0 * main_loss + 0.1 * stn_loss"""
    },
    "seg": {
        "forward": """        x = self.head_mlp(l0_points)
        x = torch.sigmoid(x).squeeze(1) # [B, N]
        return x, trans""",
        "loss": """# ==========================================
# 3. Segmentation Loss (Dice + BCE)
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
        # 因為你的模型最後已經過 sigmoid，這裡的 pred 已經是 0~1 的機率
        # 加上微小值避免 log(0) 產生 NaN
        pred = torch.clamp(pred, min=1e-7, max=1.0 - 1e-7)
        
        # 計算二元交叉熵 (BCE)
        bce = - (target * torch.log(pred) + (1.0 - target) * torch.log(1.0 - pred))
        
        # 計算預測正確的機率 pt
        pt = target * pred + (1.0 - target) * (1.0 - pred)
        
        # 計算 Focal Loss
        focal_loss = self.alpha * (1.0 - pt) ** self.gamma * bce
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss
class get_loss(nn.Module):
    def __init__(self, weight=None, use_focal=True):
        super(get_loss, self).__init__()
        self.register_buffer('iden', torch.eye(3).view(1, 3, 3))
        self.dice_loss = DiceLoss()
        
        # 根據參數選擇使用 Focal Loss 還是原來的 BCELoss
        if use_focal:
            # gamma=2.0 是一個很好的起點，你可以調大 (例如 3.0) 來進一步增加錯誤懲罰
            self.bce_loss = FocalLoss(alpha=0.5, gamma=2.0) 
        else:
            self.bce_loss = nn.BCELoss() 

    def forward(self, pred, target, trans_feat, epoch):
        target = target.float()
        
        loss_dice = self.dice_loss(pred, target)
        loss_bce = self.bce_loss(pred, target)
        
        # 你可以透過調整兩者的權重來改變側重點
        # 例如讓 BCE(Focal) 的影響力更大: total_seg_loss = loss_dice + 2.0 * loss_bce
        total_seg_loss = loss_dice + loss_bce

        if trans_feat is not None:
            batch_size = trans_feat.size(0)
            iden = self.iden.repeat(batch_size, 1, 1)
            trans_ortho = torch.bmm(trans_feat, trans_feat.transpose(1, 2))
            stn_loss = torch.norm(trans_ortho - iden, p='fro')
        else:
            stn_loss = 0.0

        return total_seg_loss + 0.001 * stn_loss"""
    }
}

# ==========================================
# 3. MLP 類別實作
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
# 4. 動態建構 Encoder/Decoder
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
# 5. 生成與存檔邏輯 (36 種組合)
# ==========================================
stn_options = ["stn", "nostn"]
sa_options = ["msg", "ssg", "mrg"]
mlp_options = ["bottleneckmlp", "invertedresidualmlp", "increasingmlp"]
task_options = ["regress", "seg"]  # 新增任務維度

output_dir = "models"
os.makedirs(output_dir, exist_ok=True)

count = 0
for task, stn, sa, mlp in product(task_options, stn_options, sa_options, mlp_options):
    filename = f"exp_pointnext_part_seg_nocls_{sa}_{mlp}_{stn}_{task}.py"
    
    mlp_code = MLP_CODES[mlp]
    init_code, forward_code = get_encoder_decoder_code(sa, mlp)
    
    # 判斷 STN
    if stn == "stn":
        stn_init = "        self.stn = STN3d(in_channel)"
        stn_forward = """        trans = self.stn(l0_points)
        l0_xyz = torch.bmm(l0_xyz.transpose(1, 2), trans).transpose(1, 2)"""
    else:
        stn_init = "        # No STN initialized"
        stn_forward = "        trans = None # No STN transformation"

    # 取得任務對應的 Forward 結尾與 Loss
    task_forward = TASK_CODES[task]["forward"]
    task_loss = TASK_CODES[task]["loss"]

    # 替換模板
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
    print(f"[{count}/36] 生成成功: {filename}")

print(f"\n🎉 完美！36 個模型腳本 (包含 Regress 與 Seg 兩種任務) 已全數生成於 '{output_dir}' 資料夾中！")