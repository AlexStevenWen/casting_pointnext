import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
# --- 1. 修改 Import 路徑 ---
from pointnet_utils_keypoint import PointNetEncoder, feature_transform_reguliarzer

class get_model(nn.Module):
    def __init__(self, k=40, normal_channel=False):
        super(get_model, self).__init__()
        
        # --- 2. 根據你的需求修改 Channel 邏輯 ---
        # 假設你的輸入是 (B, 1024, 4)，其中 4 是 XYZ + 分割維度
        if normal_channel:
            # 如果你有 XYZ + Normal(3) + Segmentation(1) = 7
            channel = 7 
        else:
            # 只有 XYZ(3) + Segmentation(1) = 4
            channel = 4
            
        self.feat = PointNetEncoder(global_feat=True, feature_transform=True, channel=channel)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x 的輸入預期為 (B, 1024, 4) 或 (B, 4, 1024)
        # 依照我們在 pointnet_utils_keypoint.py 的設計，它會自動處理轉置
        x, trans, trans_feat = self.feat(x)
        
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x, trans_feat

class get_loss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target, trans_feat):
        loss = F.nll_loss(pred, target)
        mat_diff_loss = feature_transform_reguliarzer(trans_feat)

        total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        return total_loss