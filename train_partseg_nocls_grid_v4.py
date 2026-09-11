"""
Author: Benny (Full Diagnostics + Original Logger Restored)
Date: 2025 Revised for High Recall & IoU
"""
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import datetime
import logging
import sys
import importlib
import shutil
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from scipy.spatial.distance import directed_hausdorff
from data_utils.ShapeNetDataLoader_grid import PartNormalDataset

plt.switch_backend('agg') 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

# --- 新增: Dice Loss 用於提升小目標 IoU ---
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        # pred shape: [B, N, C]  (例如: [16, 4096, 2])
        # target shape: [B, N]   (例如: [16, 4096])
        
        # 1. 因為模型輸出已經是 LogSoftmax，所以這裡用 exp 轉回機率 (0~1)
        #    如果是原始 Logits，才需要用 F.softmax
        pred_prob = torch.exp(pred)
        
        # 2. 關鍵修正：取最後一個維度的 index 1 (代表 Class 1 澆口)
        #    原本錯誤寫法 pred[:, 1, :] 會抓到"第二個點"，導致維度變成 32
        #    正確寫法 pred[:, :, 1] 會抓到"所有點的 Class 1 機率"
        pred_gate = pred_prob[:, :, 1].contiguous().view(-1)
        
        target_gate = target.contiguous().view(-1)

        # 現在兩者都是 (65536,)，可以運算了
        intersection = (pred_gate * target_gate).sum()
        union = pred_gate.sum() + target_gate.sum()

        dice_score = (2. * intersection + self.smooth) / (union + self.smooth)
        
        return 1 - dice_score

# --- 指標計算工具 (保持不變) ---
def compute_hausdorff(pc1, pc2):
    if len(pc1) == 0 or len(pc2) == 0: return 1.0
    d1 = directed_hausdorff(pc1, pc2)[0]
    d2 = directed_hausdorff(pc2, pc1)[0]
    return max(d1, d2)

def get_stats(preds, targets):
    """ 計算並返回全指標 """
    preds = preds.astype(bool); targets = targets.astype(bool)
    tp = np.sum(preds & targets)
    fp = np.sum(preds & ~targets)
    fn = np.sum(~preds & targets)
    tn = np.sum(~preds & ~targets)
    
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 1.0 if np.sum(targets) == 0 else 0.0
    acc = (tp + tn) / len(preds)
    class_acc = tp / np.sum(targets) if np.sum(targets) > 0 else 1.0
    return p, r, dice, iou, class_acc, acc

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0); pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1))); pc = pc / m
    return pc

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1: m.inplace = True

def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='pointnext_part_seg_nocls_grid')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epoch', default=501, type=int)
    parser.add_argument('--learning_rate', default=0.0002, type=float)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--log_dir', type=str, default=None)
    parser.add_argument('--decay_rate', type=float, default=1e-3)
    parser.add_argument('--npoint', type=int, default=4096)
    parser.add_argument('--normal', action='store_true', default=False)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--grid_num', type=int, default=4)
    return parser.parse_args()

def plot_performance(exp_dir, history):
    epochs = range(1, len(history['train_loss']) + 1)
    
    # 1. Loss 曲線
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history['train_loss'], 'r-', label='Train Loss')
    plt.plot(epochs, history['test_loss'], 'b--', label='Test Loss')
    plt.title('Loss Convergence'); plt.legend(); plt.grid(True); plt.savefig(os.path.join(exp_dir, 'loss.png')); plt.close()

    # 2. Point 分支效能 (mIoU / Mean Acc)
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history['pt_miou'], 'b-', label='Point mIoU')
    plt.plot(epochs, history['pt_mean_acc'], 'g--', label='Point Mean Acc')
    plt.plot(epochs, history['pt_gate_iou'], 'm-', label='Gate IoU')
    plt.title('Point Branch Metrics'); plt.legend(); plt.grid(True); plt.savefig(os.path.join(exp_dir, 'pt_metrics.png')); plt.close()

    # 3. Grid 分支效能 (mIoU / GateGrid IoU)
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history['grid_miou'], 'c-', label='Grid mIoU')
    plt.plot(epochs, history['grid_mean_acc'], 'k--', label='Grid Mean Acc')
    plt.plot(epochs, history['grid_gate_iou'], 'y-', label='Gate Grid IoU')
    plt.title('Grid Branch Metrics'); plt.legend(); plt.grid(True); plt.savefig(os.path.join(exp_dir, 'grid_metrics.png')); plt.close()

    # 4. 保守度分析 (Precision vs Recall)
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history['pt_gate_p'], 'g-', label='Point Precision')
    plt.plot(epochs, history['pt_gate_r'], 'r-', label='Point Recall')
    plt.axhline(y=0.5, color='gray', linestyle='--')
    plt.title('Conservative Analysis (Bias)'); plt.legend(); plt.grid(True); plt.savefig(os.path.join(exp_dir, 'bias.png')); plt.close()

    # 5. 空間誤差 (Hausdorff)
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history['hausdorff'], 'k-', label='Hausdorff Distance')
    plt.title('Spatial Error Analysis'); plt.legend(); plt.grid(True); plt.savefig(os.path.join(exp_dir, 'hausdorff.png')); plt.close()

def main(args):
    def log_string(str):
        logger.info(str); print(str)

    # --- 恢復原始 LOGGER 設定 ---
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    exp_dir = Path('./log/part_seg/').joinpath(f"{args.model}_{timestr}")
    exp_dir.mkdir(exist_ok=True, parents=True)
    checkpoints_dir, log_dir = exp_dir.joinpath('checkpoints/'), exp_dir.joinpath('logs/')
    checkpoints_dir.mkdir(exist_ok=True); log_dir.mkdir(exist_ok=True)

    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler = logging.FileHandler(f'{log_dir}/{args.model}.txt')
    file_handler.setFormatter(formatter); logger.addHandler(file_handler)

    log_string('PARAMETER ...')
    log_string(args)

    # 數據加載
    TRAIN_DATASET = PartNormalDataset(root=args.data_dir, npoints=args.npoint, split='trainval', grid_num=args.grid_num, augment=True)
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    TEST_DATASET = PartNormalDataset(root=args.data_dir, npoints=args.npoint, split='test', grid_num=args.grid_num, augment=False)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=4)

    MODEL = importlib.import_module(args.model)
    classifier = MODEL.get_model(2, normal_channel=args.normal, grid_num=args.grid_num).cuda()
    classifier.apply(inplace_relu)
    
    # --- 優化 Loss 設置 ---
    # 1. NLL Loss (基礎分類) - 保留高權重 30.0
    seg_weights = torch.Tensor([1.0, 20.0]).cuda() 
    nll_criterion = torch.nn.NLLLoss(weight=seg_weights).cuda()
    
    # 2. Dice Loss (核心修改：直接優化 IoU)
    dice_criterion = DiceLoss().cuda()
    
    # 3. Grid Loss (粗粒度引導)
    grid_criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0]).cuda()).cuda()
    
    optimizer = torch.optim.Adam(classifier.parameters(), lr=args.learning_rate, weight_decay=args.decay_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch, eta_min=1e-6)

    history = {
        'train_loss': [], 'test_loss': [],
        'pt_miou': [], 'pt_mean_acc': [], 'pt_overall_acc': [], 'pt_gate_iou': [], 'pt_body_iou': [], 
        'pt_gate_p': [], 'pt_gate_r': [], 'pt_dice': [],
        'grid_miou': [], 'grid_mean_acc': [], 'grid_overall_acc': [], 'grid_gate_iou': [], 'grid_empty_iou': [],
        'grid_p': [], 'grid_r': [], 'grid_dice': [], 'hausdorff': []
    }
    best_gate_iou = 0

    for epoch in range(args.epoch):
        log_string(f'**** Epoch {epoch + 1} ****')
        classifier.train()
        train_loss_epoch = []
        for i, (points, label, target, grid_gt) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
            optimizer.zero_grad()
            points, target, grid_gt = points.float().cuda(), target.long().cuda(), grid_gt.float().cuda()
            
            seg_pred, grid_pred = classifier(points.transpose(2, 1))
            
            # --- Hybrid Loss 計算 ---
            # 結合 NLL(分類準確) + Dice(幾何重疊) + Grid(區域引導)
            loss_nll = nll_criterion(seg_pred.contiguous().view(-1, 2), target.view(-1))
            loss_dice = dice_criterion(seg_pred, target)
            loss_grid = grid_criterion(grid_pred, grid_gt)
            
            # Dice Loss 權重設為 1.0，與 NLL 共同作用
            loss = loss_nll + loss_dice + 2.5 * loss_grid
            
            loss.backward()
            optimizer.step()
            train_loss_epoch.append(loss.item())
            
        history['train_loss'].append(np.mean(train_loss_epoch))

        with torch.no_grad():
            classifier.eval()
            m = {k: [] for k in history.keys() if k != 'train_loss'}

            for batch_id, (points, label, target, grid_gt) in enumerate(testDataLoader):
                cur_batch_size = points.size(0)
                points_cuda, target_cuda, grid_gt_cuda = points.float().cuda(), target.long().cuda(), grid_gt.float().cuda()
                
                seg_pred, grid_pred = classifier(points_cuda.transpose(2, 1))
                
                # Test Loss 計算 (保持一致)
                l_nll = nll_criterion(seg_pred.view(-1, 2), target_cuda.view(-1))
                l_dice = dice_criterion(seg_pred, target_cuda)
                l_grid = grid_criterion(grid_pred, grid_gt_cuda)
                m['test_loss'].append((l_nll + l_dice + 2.5 * l_grid).item())
                
                pt_pred = np.argmax(seg_pred.cpu().numpy(), 2) 
                pt_gt = target.numpy()
                gr_probs = torch.sigmoid(grid_pred)
                gr_pred = (gr_probs > 0.5).cpu().numpy()
                gr_gt = grid_gt.numpy()

                # --- 修正後的過濾邏輯 (Boost Recall) ---
                if epoch > 50: # 提早開始觀測效果
                    # 關鍵修改：將閾值從 0.3 降至 0.05
                    # 邏輯：只要 Grid 覺得"有點可能"是澆口，就保留 Point 的預測，不輕易殺死 Recall
                    grid_mask = (gr_probs > 0.05).cpu().numpy()
                    
                    points_np = points.numpy()
                    for b in range(cur_batch_size):
                        for n in range(args.npoint):
                            # 只檢查被預測為澆口的點
                            if pt_pred[b, n] == 1:
                                xyz = points_np[b, n]
                                grid_size = 2.0 / args.grid_num
                                # 計算該點對應的 Grid Index
                                gi, gj, gk = (int((xyz[i] + 1.0) / grid_size) for i in range(3))
                                gi, gj, gk = (min(v, args.grid_num-1) for v in (gi, gj, gk))
                                
                                # 只有當 Grid 完全確認這裡是背景(機率<0.05)時，才過濾掉
                                idx = gi * (args.grid_num**2) + gj * args.grid_num + gk
                                if grid_mask[b, idx] == 0: 
                                    pt_pred[b, n] = 0

                for b in range(cur_batch_size):
                    # 1. Point Branch
                    p1, r1, d1, iou1, acc1, oa1 = get_stats(pt_pred[b] == 1, pt_gt[b] == 1)
                    _, _, _, iou0, acc0, _ = get_stats(pt_pred[b] == 0, pt_gt[b] == 0)
                    m['pt_gate_p'].append(p1); m['pt_gate_r'].append(r1); m['pt_dice'].append(d1)
                    m['pt_gate_iou'].append(iou1); m['pt_body_iou'].append(iou0)
                    m['pt_miou'].append((iou1 + iou0) / 2); m['pt_mean_acc'].append((acc1 + acc0) / 2); m['pt_overall_acc'].append(oa1)
                    m['hausdorff'].append(compute_hausdorff(points[b][pt_pred[b]==1].numpy(), points[b][pt_gt[b]==1].numpy()))
                    # 2. Grid Branch
                    gp1, gr1, gd1, giou1, gacc1, oa_g = get_stats(gr_pred[b] == 1, gr_gt[b] == 1)
                    _, _, _, giou0, gacc0, _ = get_stats(gr_pred[b] == 0, gr_gt[b] == 0)
                    m['grid_p'].append(gp1); m['grid_r'].append(gr1); m['grid_dice'].append(gd1)
                    m['grid_gate_iou'].append(giou1); m['grid_empty_iou'].append(giou0)
                    m['grid_miou'].append((giou1 + giou0) / 2); m['grid_mean_acc'].append((gacc1 + gacc0) / 2); m['grid_overall_acc'].append(oa_g)

            for key in m.keys(): history[key].append(np.mean(m[key]))

            # --- Log 輸出 ---
            log_string(f'Test Total Loss: {history["test_loss"][-1]:.4f}')
            log_string(f'Point Branch -> mIoU: {history["pt_miou"][-1]:.4f} | OverallAcc: {history["pt_overall_acc"][-1]:.4f} | MeanAcc: {history["pt_mean_acc"][-1]:.4f}')
            log_string(f'               Gate IoU: {history["pt_gate_iou"][-1]:.4f}, Body IoU: {history["pt_body_iou"][-1]:.4f}, Dice: {history["pt_dice"][-1]:.4f}, HD: {history["hausdorff"][-1]:.4f}')
            log_string(f'               Gate Bias -> Precision: {history["pt_gate_p"][-1]:.4f} | Recall: {history["pt_gate_r"][-1]:.4f}')
            log_string(f'Grid Branch  -> mIoU: {history["grid_miou"][-1]:.4f} | OverallAcc: {history["grid_overall_acc"][-1]:.4f} | MeanAcc: {history["grid_mean_acc"][-1]:.4f}')
            log_string(f'               GateGridIoU: {history["grid_gate_iou"][-1]:.4f}, EmptyGridIoU: {history["grid_empty_iou"][-1]:.4f}, Dice: {history["grid_dice"][-1]:.4f}')
            log_string(f'               Grid Bias -> Precision: {history["grid_p"][-1]:.4f} | Recall: {history["grid_r"][-1]:.4f}')
            
            plot_performance(exp_dir, history)
            # 使用 Gate IoU 作為儲存模型的依據
            if history['pt_gate_iou'][-1] >= best_gate_iou:
                best_gate_iou = history['pt_gate_iou'][-1]
                torch.save({'model_state_dict': classifier.state_dict(), 'gate_iou': best_gate_iou}, str(checkpoints_dir) + '/best_gate_model.pth')
                log_string(f'Saving Model with Gate IoU: {best_gate_iou:.4f}')
        
        scheduler.step()

if __name__ == '__main__':
    args = parse_args()
    main(args)