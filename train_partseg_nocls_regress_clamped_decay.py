"""
Author: Benny (Universal Regression Edition) - Enhanced Visualization
Date: 2025 Revised for Regression Task (With Spatial Metrics & Multi-View Plotting)
Description: 
    - 自動適配有無 Grid Branch 的模型
    - 新增指標: Spatial Shift (位置誤差) & Weighted Shift (重心誤差)
    - 新增視覺化: 自動存儲多視角對比圖 (Pred vs GT)
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
from data_utils.ShapeNetDataLoader_grid_clamped_decay_v2 import PartNormalDataset

# 設定 Matplotlib 後端，避免在無螢幕伺服器報錯
plt.switch_backend('agg') 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

# ==========================================
#      視覺化工具函數 (新增部分)
# ==========================================
def visualize_regression_batch(points, preds, targets, save_path, epoch):
    """
    繪製 3D 點雲回歸對比圖 (多視角)
    points: (N, 3) numpy array
    preds: (N,) numpy array (預測熱力值)
    targets: (N,) numpy array (真實熱力值)
    save_path: 圖片儲存路徑
    """
    # 建立畫布：2 列 (Pred/GT) x 3 行 (不同視角)
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(f'Epoch {epoch} - Prediction vs Ground Truth', fontsize=16)

    # 定義三個視角 (Elevation, Azimuth)
    views = [
        (0, 0, 'Front View'),      # 正視
        (90, -90, 'Top View'),     # 俯視
        (30, 45, 'Iso View')       # 等角視圖
    ]
    
    # 找出統一的 Color Range (讓對比更公平)
    vmin = min(preds.min(), targets.min())
    vmax = max(preds.max(), targets.max())

    # --- 第一排：Prediction ---
    for i, (elev, azim, title) in enumerate(views):
        ax = fig.add_subplot(2, 3, i + 1, projection='3d')
        p = ax.scatter(points[:, 0], points[:, 2], points[:, 1], c=preds, cmap='jet', s=2, vmin=vmin, vmax=vmax)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f'Pred - {title}')
        ax.axis('off') # 隱藏座標軸讓圖更乾淨

    # --- 第二排：Ground Truth ---
    for i, (elev, azim, title) in enumerate(views):
        ax = fig.add_subplot(2, 3, i + 4, projection='3d')
        p = ax.scatter(points[:, 0], points[:, 2], points[:, 1], c=targets, cmap='jet', s=2, vmin=vmin, vmax=vmax)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f'GT - {title}')
        ax.axis('off')

    # 添加 Colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(p, cax=cbar_ax, label='Heatmap Intensity')

    plt.tight_layout(rect=[0, 0, 0.9, 1]) # 調整佈局避免重疊
    plt.savefig(save_path, dpi=150)
    plt.close(fig)

# ==========================================
#      數值計算工具函數
# ==========================================
def get_regression_stats(preds, targets):
    mae = np.mean(np.abs(preds - targets))
    mse = np.mean((preds - targets) ** 2)
    return mae, mse

def get_spatial_metrics(points, preds, targets, threshold=0.5):
    """
    計算空間位置誤差 (Spatial Error)
    points: (B, N, 3) 點雲座標
    preds: (B, N) 預測熱力值
    targets: (B, N) 真實熱力值
    """
    batch_size = points.shape[0]
    
    # 1. Argmax Shift (單點峰值誤差)
    pred_max_idx = torch.argmax(preds, dim=1) # (B,)
    gt_max_idx = torch.argmax(targets, dim=1) # (B,)
    
    # 取出座標
    pred_xyz = points[torch.arange(batch_size), pred_max_idx, :] # (B, 3)
    gt_xyz = points[torch.arange(batch_size), gt_max_idx, :]     # (B, 3)
    
    # 計算歐式距離
    argmax_dists = torch.norm(pred_xyz - gt_xyz, dim=1)
    
    # 2. Weighted Center Shift (重心誤差 - 解決"中心很多"的問題)
    weighted_dists = []
    for b in range(batch_size):
        p = points[b]   # (N, 3)
        i = preds[b]    # (N,)
        t_center = gt_xyz[b] # (3,) 真實中心
        
        # 篩選高分點 (使用 ReLU 確保非負)
        weights = torch.clamp(i, min=0) 
        total_weight = weights.sum()
        
        if total_weight > 1e-6:
            # 計算重心: sum(P * w) / sum(w)
            weighted_center = (p * weights.unsqueeze(1)).sum(dim=0) / total_weight
            d = torch.norm(weighted_center - t_center).item()
        else:
            # 如果預測全為 0，退化回 Argmax 距離
            d = argmax_dists[b].item()
            
        weighted_dists.append(d)
        
    return argmax_dists.mean().item(), np.mean(weighted_dists)

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
    # 新增視覺化頻率參數
    parser.add_argument('--vis_freq', type=int, default=5, help='Save visualization every N epochs')
    return parser.parse_args()

def plot_performance(exp_dir, history):
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history['train_loss'], 'r-', label='Train Loss')
    plt.plot(epochs, history['test_loss'], 'b--', label='Test Loss')
    plt.title('Loss Convergence'); plt.legend(); plt.grid(True); plt.savefig(os.path.join(exp_dir, 'loss.png')); plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history['test_mae'], 'b-', label='Test MAE')
    plt.plot(epochs, history['test_mse'], 'g--', label='Test MSE')
    plt.title('Regression Error'); plt.legend(); plt.grid(True); plt.savefig(os.path.join(exp_dir, 'regression_error.png')); plt.close()
    
    if 'shift_err' in history:
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, history['shift_err'], 'k-', label='Argmax Shift Error')
        plt.plot(epochs, history['w_shift_err'], 'm--', label='Weighted Shift Error')
        plt.title('Spatial Center Shift Error'); plt.legend(); plt.grid(True); plt.savefig(os.path.join(exp_dir, 'spatial_error.png')); plt.close()

def main(args):
    def log_string(str):
        logger.info(str); print(str)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    exp_dir = Path('./log/part_seg/').joinpath(f"{args.model}_{timestr}")
    exp_dir.mkdir(exist_ok=True, parents=True)
    checkpoints_dir, log_dir = exp_dir.joinpath('checkpoints/'), exp_dir.joinpath('logs/')
    checkpoints_dir.mkdir(exist_ok=True); log_dir.mkdir(exist_ok=True)
    
    # 新增：建立視覺化結果資料夾
    visual_dir = exp_dir.joinpath('visual_results/')
    visual_dir.mkdir(exist_ok=True)

    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler = logging.FileHandler(f'{log_dir}/{args.model}.txt')
    file_handler.setFormatter(formatter); logger.addHandler(file_handler)

    log_string('PARAMETER ...')
    log_string(args)

    TRAIN_DATASET = PartNormalDataset(root=args.data_dir, npoints=args.npoint, split='trainval', grid_num=args.grid_num, augment=True)
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    TEST_DATASET = PartNormalDataset(root=args.data_dir, npoints=args.npoint, split='test', grid_num=args.grid_num, augment=False)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=4)

    MODEL = importlib.import_module(args.model)
    classifier = MODEL.get_model(2, normal_channel=args.normal, grid_num=args.grid_num).cuda()
    classifier.apply(inplace_relu)
    
    grid_criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0]).cuda()).cuda()
    mse_criterion = torch.nn.MSELoss().cuda() 
    
    optimizer = torch.optim.Adam(classifier.parameters(), lr=args.learning_rate, weight_decay=args.decay_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch, eta_min=1e-6)

    history = {'train_loss': [], 'test_loss': [], 'test_mae': [], 'test_mse': [], 'grid_acc': [], 'shift_err': [], 'w_shift_err': []}
    best_mae = float('inf')

    for epoch in range(args.epoch):
        log_string(f'**** Epoch {epoch + 1} ****')
        classifier.train()
        train_loss_epoch = []
        
        for i, (points, label, target, grid_gt) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
            optimizer.zero_grad()
            points, target, grid_gt = points.float().cuda(), target.float().cuda(), grid_gt.float().cuda()
            
            # Forward
            seg_pred, grid_logits = classifier(points.transpose(2, 1))
            pred_intensity = seg_pred
            
            loss_mse = mse_criterion(pred_intensity, target)
            
            if grid_logits is not None:
                loss_grid = grid_criterion(grid_logits, grid_gt)
                loss = 50.0 * loss_mse + 2.0 * loss_grid
            else:
                loss = 50.0 * loss_mse
            
            loss.backward()
            # 這會限制梯度的最大範數 (Norm) 不超過 1.0
            # 防止因為加權 Loss 導致的梯度爆炸，讓 Loss 曲線平滑一點
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss_epoch.append(loss.item())
            
        history['train_loss'].append(np.mean(train_loss_epoch))

        with torch.no_grad():
            classifier.eval()
            m = {'test_loss': [], 'mae': [], 'mse': [], 'grid_acc': [], 'shift_err': [], 'w_shift_err': []}

            for batch_id, (points, label, target, grid_gt) in enumerate(testDataLoader):
                points_cuda, target_cuda, grid_gt_cuda = points.float().cuda(), target.float().cuda(), grid_gt.float().cuda()
                
                seg_pred, grid_logits = classifier(points_cuda.transpose(2, 1))
                pred_intensity = seg_pred
                
                l_mse = mse_criterion(pred_intensity, target_cuda)
                
                if grid_logits is not None:
                    l_grid = grid_criterion(grid_logits, grid_gt_cuda)
                    m['test_loss'].append((50.0 * l_mse + 2.0 * l_grid).item())
                    
                    gr_probs = torch.sigmoid(grid_logits)
                    gr_pred_binary = (gr_probs > 0.5).float()
                    grid_acc = (gr_pred_binary == grid_gt_cuda).float().mean().item()
                    m['grid_acc'].append(grid_acc)
                else:
                    m['test_loss'].append((50.0 * l_mse).item())
                    m['grid_acc'].append(0.0)
                
                # 計算空間誤差
                s_err, w_s_err = get_spatial_metrics(points_cuda, pred_intensity, target_cuda)
                m['shift_err'].append(s_err)
                m['w_shift_err'].append(w_s_err)

                mae, mse = get_regression_stats(pred_intensity.cpu().numpy(), target.numpy())
                m['mae'].append(mae)
                m['mse'].append(mse)
                
                # [探針與視覺化核心代碼]
                # 只對每個 Epoch 的第 0 個 Batch 進行詳細檢查與繪圖
                if batch_id == 0:
                    batch_pred_max = pred_intensity.max().item()
                    batch_gt_max = target_cuda.max().item()
                    mask = (target_cuda > 0.1)
                    avg_pred_on_gate = pred_intensity[mask].mean().item() if mask.sum() > 0 else 0.0
                    
                    log_string(f'---- PROBE [Epoch {epoch+1}] ----')
                    log_string(f'GT Max Val     : {batch_gt_max:.4f}')
                    log_string(f'Pred Max Val   : {batch_pred_max:.4f}')
                    log_string(f'Gate Avg Pred  : {avg_pred_on_gate:.4f}')
                    log_string(f'Batch Shift Err: {s_err:.4f}')
                    
                    # ----------------------------------------------------
                    # [視覺化邏輯] 每 N 個 Epoch 或 最後一個 Epoch 畫一次圖
                    # ----------------------------------------------------
                    if (epoch + 1) % args.vis_freq == 0 or (epoch + 1) == args.epoch:
                        save_name = visual_dir.joinpath(f'epoch_{epoch+1}_batch0.png')
                        
                        # 準備數據 (取 batch 中的第 0 個樣本)
                        v_points = points[0].numpy()      # (N, 3)
                        v_preds = pred_intensity[0].cpu().numpy()  # (N,)
                        v_targets = target[0].numpy()     # (N,)
                        
                        log_string(f'Saving visualization to {save_name} ...')
                        visualize_regression_batch(v_points, v_preds, v_targets, save_name, epoch+1)
                    
                    log_string(f'-------------------------------')

            epoch_mae = np.mean(m['mae'])
            epoch_mse = np.mean(m['mse'])
            epoch_grid_acc = np.mean(m['grid_acc'])
            epoch_shift_err = np.mean(m['shift_err'])
            epoch_w_shift_err = np.mean(m['w_shift_err'])
            
            history['test_loss'].append(np.mean(m['test_loss']))
            history['test_mae'].append(epoch_mae)
            history['test_mse'].append(epoch_mse)
            history['grid_acc'].append(epoch_grid_acc)
            history['shift_err'].append(epoch_shift_err)
            history['w_shift_err'].append(epoch_w_shift_err)

            log_string(f'Test Total Loss: {history["test_loss"][-1]:.4f}')
            log_string(f'MAE: {epoch_mae:.5f} | MSE: {epoch_mse:.5f}')
            log_string(f'Shift Error -> Argmax: {epoch_shift_err:.4f} | Weighted: {epoch_w_shift_err:.4f}')
            
            if grid_logits is not None:
                log_string(f'Grid Acc: {epoch_grid_acc:.4f}')
            
            plot_performance(exp_dir, history)
            
            if epoch_mae <= best_mae:
                best_mae = epoch_mae
                torch.save({'model_state_dict': classifier.state_dict(), 'mae': best_mae}, str(checkpoints_dir) + '/best_regression_model.pth')
                log_string(f'Saving Model with Lowest MAE: {best_mae:.5f}')
        
        scheduler.step()

if __name__ == '__main__':
    args = parse_args()
    main(args)