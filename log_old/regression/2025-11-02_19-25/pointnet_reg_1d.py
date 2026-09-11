import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from pointnet_utils import PointNetEncoder, feature_transform_reguliarzer

class get_model(nn.Module):
    # 1. 將 k=40 改成 output_dim=6 (或 3)
    def __init__(self, output_dim=6, normal_channel=True):
        super(get_model, self).__init__()
        if normal_channel:
            channel = 6
        else:
            channel = 3
        self.feat = PointNetEncoder(global_feat=True, feature_transform=True, channel=channel,use_stn=False)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        # 2. 讓 fc3 輸出的維度為 output_dim
        self.fc3 = nn.Linear(256, output_dim)
        self.dropout = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        # 3. 移除 F.log_softmax(x, dim=1)
        #    回歸任務直接返回 fc3 的輸出
        x = torch.tanh(x)
        return x, trans_feat

class get_loss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target, trans_feat):
            # (*** 檢查您是否使用了 L1 Loss ***)
            loss = F.l1_loss(pred, target)
            # (*** ------------------ ***)
            
            mat_diff_loss = feature_transform_reguliarzer(trans_feat)
            
            # (*** 檢查 mat_diff_loss 是否為 None ***)
            if trans_feat is None:
                mat_diff_loss = 0 # 如果 STN 關閉了，trans_feat 為 None
                
            total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
            return total_loss