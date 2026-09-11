import torch # <--- (修正 1: 補上 Import)
import torch.nn as nn
import torch.nn.functional as F
from models.pointnet2_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction # (路徑修正為 models.pointnet2_utils)


class get_model(nn.Module):
    def __init__(self, output_dim=3, normal_channel=True): 
        super(get_model, self).__init__()
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel
        
        # (PointNet++ 的階層式編碼器)
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        
        # (「回歸頭」)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, output_dim) # <-- 輸出 N 維

    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
            
        # (PointNet++ 的階層式編碼)
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        
        # (使用「回歸頭」)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        
        # (加上 Tanh 激活函數)
        x = torch.tanh(x)
        
        # (回傳 x 和 None，以匹配訓練腳本中 criterion 的簽名)
        return x, None 


# --- (修正 2: 大幅簡化 get_loss) ---
class get_loss(nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001): # (mat_diff_loss_scale 參數現在不會被用到)
        super(get_loss, self).__init__()
        
    def forward(self, pred, target, trans_feat):
        # (使用 L1 Loss)
        loss = F.l1_loss(pred, target)
        
        # (trans_feat 在 PointNet++ 中為 None，我們直接忽略它)
        # (不再需要 mat_diff_loss)
        
        return loss