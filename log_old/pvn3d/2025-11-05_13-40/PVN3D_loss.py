import torch
import torch.nn as nn
import torch.nn.functional as F

"""
PVN3D 損失函數

我們只計算「姿態損失」(Loss_pose)，即預測的 8 個關鍵點
與真實的 8 個關鍵點之間的 L1 距離。
"""

class get_loss(nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat=None):
        # pred shape: (B, 8, 3) - 預測的 8 個關鍵點
        # target shape: (B, 8, 3) - 真實的 8 個關鍵點
        # trans_feat: (None) - 我們不需要它

        # L1 Loss (Mean Absolute Error)
        loss = F.l1_loss(pred, target)

        return loss