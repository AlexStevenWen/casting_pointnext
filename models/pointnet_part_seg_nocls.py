import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
from pointnet_utils import STN3d, STNkd, feature_transform_reguliarzer

class get_model(nn.Module):
    def __init__(self, part_num=2, num_classes=1, normal_channel=True):
        super(get_model, self).__init__()
        self.part_num = part_num
        self.num_classes = num_classes # 這裡現在是 1
        channel = 6 if normal_channel else 3
        
        self.stn = STN3d(channel)
        self.conv1 = nn.Conv1d(channel, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 128, 1)
        self.conv4 = nn.Conv1d(128, 512, 1)
        self.conv5 = nn.Conv1d(512, 2048, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(2048)
        
        self.fstn = STNkd(k=128)
        
        # 關鍵修正：動態計算輸入維度
        # 4928 是 64+128+128+512+2048 (各層) + 2048 (Global)
        self.convs1 = nn.Conv1d(4928 + self.num_classes, 256, 1)
        self.convs2 = nn.Conv1d(256, 256, 1)
        self.convs3 = nn.Conv1d(256, 128, 1)
        self.convs4 = nn.Conv1d(128, part_num, 1)
        self.bns1 = nn.BatchNorm1d(256)
        self.bns2 = nn.BatchNorm1d(256)
        self.bns3 = nn.BatchNorm1d(128)

    def forward(self, point_cloud, label):
        B, D, N = point_cloud.size()
        trans = self.stn(point_cloud)
        point_cloud = point_cloud.transpose(2, 1)
        if D > 3:
            point_cloud, feature = point_cloud.split(3, dim=2)
        point_cloud = torch.bmm(point_cloud, trans)
        if D > 3:
            point_cloud = torch.cat([point_cloud, feature], dim=2)

        point_cloud = point_cloud.transpose(2, 1)

        out1 = F.relu(self.bn1(self.conv1(point_cloud)))
        out2 = F.relu(self.bn2(self.conv2(out1)))
        out3 = F.relu(self.bn3(self.conv3(out2)))

        trans_feat = self.fstn(out3)
        x = out3.transpose(2, 1)
        net_transformed = torch.bmm(x, trans_feat)
        net_transformed = net_transformed.transpose(2, 1)

        out4 = F.relu(self.bn4(self.conv4(net_transformed)))
        out5 = self.bn5(self.conv5(out4))
        out_max = torch.max(out5, 2, keepdim=True)[0]
        out_max = out_max.view(-1, 2048)

        # 這裡會拼接 1 維的 One-hot (或你設定的 num_classes)
        out_max = torch.cat([out_max, label.squeeze(1)], 1)
        
        # 動態獲取拼接後的維度
        combined_dim = out_max.shape[1] 
        expand = out_max.view(-1, combined_dim, 1).repeat(1, 1, N)
        
        # 拼接所有特徵
        concat = torch.cat([expand, out1, out2, out3, out4, out5], 1)
        
        # 如果 convs1 定義正確，這裡就不會報錯
        net = F.relu(self.bns1(self.convs1(concat)))
        net = F.relu(self.bns2(self.convs2(net)))
        net = F.relu(self.bns3(self.convs3(net)))
        net = self.convs4(net)
        
        net = net.transpose(2, 1).contiguous()
        net = F.log_softmax(net.view(-1, self.part_num), dim=-1)
        net = net.view(B, N, self.part_num)

        return net, trans_feat

class get_loss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001, weight=None):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale
        # 這裡支持傳入權重
        self.weight = weight 

    def forward(self, pred, target, trans_feat):
        # 使用傳入的權重進行負對數似然損失計算
        loss = F.nll_loss(pred, target, weight=self.weight)
        
        # PointNet 特有的特徵矩陣正則化損失
        mat_diff_loss = feature_transform_reguliarzer(trans_feat)
        
        total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        return total_loss