import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
# --- 1. 修改 Import 路徑為你的新檔名 ---
from pointnet_utils_keypoint import PointNetEncoder, feature_transform_reguliarzer


class get_model(nn.Module):
    def __init__(self, num_class):
        super(get_model, self).__init__()
        self.k = num_class
        
        # --- 2. 修改輸入通道數 ---
        # 假設你的輸入是 (B, 1024, 4) 或 (B, 4, 1024)
        # 如果只有 XYZ + 分割維度，channel=4
        # 如果原本是 9 (XYZ+RGB+Normal)，現在多一維就變成 10
        input_channel = 4 
        self.feat = PointNetEncoder(global_feat=False, feature_transform=True, channel=input_channel)
        
        # --- 3. 計算 conv1 的輸入維度 ---
        # PointNetEncoder 在 global_feat=False 時會拼接：
        # 全局特徵 (1024) + 點特徵 (64) = 1088
        # 這個 1088 是固定的，不需要因為輸入變 4 而改動
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        # 確保 x 進入時是 (B, C, N)
        # 根據我們在 pointnet_utils_keypoint.py 的修改，
        # forward 內部已經有處理轉置邏輯，所以這裡直接傳入即可
        
        batchsize = x.size()[0]
        # 注意：如果 x 是 (B, N, C)，x.size()[2] 就會是通道數而不是點數
        # 建議用 x.shape 判斷點數
        if x.size(1) == 4: # 代表形狀是 (B, 4, N)
            n_pts = x.size()[2]
        else: # 代表形狀是 (B, N, 4)
            n_pts = x.size()[1]
            
        x, trans, trans_feat = self.feat(x)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        
        x = x.transpose(2, 1).contiguous()
        x = F.log_softmax(x.view(-1, self.k), dim=-1)
        x = x.view(batchsize, n_pts, self.k)
        return x, trans_feat

class get_loss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target, trans_feat, weight=None):
        # target shape: (B, N)
        # pred shape: (B, N, K) -> view(-1, K)
        loss = F.nll_loss(pred.view(-1, pred.size(-1)), target.view(-1), weight=weight)
        mat_diff_loss = feature_transform_reguliarzer(trans_feat)
        total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        return total_loss


if __name__ == '__main__':
    # 測試程式碼
    model = get_model(13)
    # 模擬 4 通道輸入 (B, C, N) -> (12, 4, 2048)
    xyz_plus_seg = torch.rand(12, 4, 2048)
    pred, trans_feat = model(xyz_plus_seg)
    print(f"Output shape: {pred.shape}") # 應為 (12, 2048, 13)