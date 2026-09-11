import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction # (確保 import 這些)
class get_model(nn.Module):
    def __init__(self, output_dim=3, normal_channel=True): # <--- (修改 1)
        super(get_model, self).__init__()
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel
        
        # (PointNet++ 的階層式編碼器，這整段保持不變)
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        
        # --- (修改 2：將「分類頭」改成「回歸頭」) ---
        # (刪除舊的 fc1, bn1, drop1, fc2, bn2, drop2, fc3)
        # (替換為我們熟悉的回歸層)
        
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, output_dim) # <-- 輸出 3D
        # --- (修改結束) ---

    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
            
        # (PointNet++ 的階層式編碼，保持不變)
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        
        # --- (修改 3：使用新的「回歸頭」) ---
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        
        # (*** 關鍵 ***) 加上 Tanh 激活函數
        x = torch.tanh(x)
        
        # (PointNet v1 會回傳 trans_feat, PointNet++ 不會)
        # (所以我們回傳 x 和一個 None 值)
        return x, None # <--- (修改 4)
        # --- (修改結束) ---

# --- (修改 5：加入 get_loss) ---
# (從您的 pointnet_reg_1d.py 複製 get_loss 過來)
class get_loss(nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target, trans_feat):
        # (使用 L1 Loss)
        loss = F.l1_loss(pred, target)

        # (因為 PointNet++ 不回傳 trans_feat, mat_diff_loss 永遠為 0)
        if trans_feat is None:
            mat_diff_loss = 0.0
        else:
            mat_diff_loss = feature_transform_reguliarzer(trans_feat)

        total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        return total_loss