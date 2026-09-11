import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
from pointnet_utils import PointNetEncoder, feature_transform_reguliarzer

class get_model(nn.Module):
    def __init__(self, num_class, normal_channel=True):
        super(get_model, self).__init__()
        # 鑄件任務中 num_class 通常為 2 (本體, 澆口)
        self.k = num_class
        
        # 根據是否包含法向量(normal)決定輸入通道數
        # 如果 args.normal 為 True，channel 為 6 (XYZ + Normal)
        # 原始 PointNet SemSeg 預設常為 9 (XYZ, RGB, Normalized XYZ)，這裡改為動態
        channel = 6 if normal_channel else 3
        
        # PointNetEncoder 會提取局部與全局特徵的拼接 (通常是 1088 維)
        self.feat = PointNetEncoder(global_feat=False, feature_transform=True, channel=channel)
        
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        
        # x: [B, D, N] -> x_feat: [B, 1088, N]
        x, trans, trans_feat = self.feat(x)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        
        x = x.transpose(2, 1).contiguous()
        # 使用 log_softmax 搭配 NLLLoss
        x = F.log_softmax(x.view(-1, self.k), dim=-1)
        x = x.view(batchsize, n_pts, self.k)
        
        return x, trans_feat

class get_loss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001, weight=None):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale
        # 接收外部傳入的類別權重 [1.0, 10.0]
        self.weight = weight

    def forward(self, pred, target, trans_feat):
        # pred: [B*N, K], target: [B*N]
        loss = F.nll_loss(pred, target, weight=self.weight)
        
        # 特徵矩陣正則化，保證旋轉不變性
        mat_diff_loss = feature_transform_reguliarzer(trans_feat)
        
        total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        return total_loss

if __name__ == '__main__':
    # 測試代碼：假設 2 個類別，輸入包含法向量 (6通道)
    model = get_model(2, normal_channel=True)
    xyz_normal = torch.rand(12, 6, 2048)
    pred, trans_feat = model(xyz_normal)
    print('Output shape:', pred.shape) # 預期 [12, 2048, 2]