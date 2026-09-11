"""
Author: Benny (Universal Regression Edition) - Enhanced Visualization & Anti-Overfitting
Date: 2025 Revised for Regression Task (With Spatial Metrics & Smart Rewind)
Description: 
    - [修正] 適配 STN 模型架構 (接收 trans_feat)
    - [修正] 使用 MODEL.get_loss 進行 STN 正則化與 Warmup
    - [修正] 預設 Learning Rate 降為 0.0001
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
#      視覺化工具函數
# ==========================================
def visualize_regression_batch(points, preds, targets, save_path, epoch):
    """
    繪製 3D 點雲回歸對比圖 (多視角)
    """
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(f'Epoch {epoch} - Prediction vs Ground Truth', fontsize=16)

    views = [
        (0, 0, 'Front View'),      # 正視
        (90, -90, 'Top View'),     # 俯視
        (30, 45, 'Iso View')       # 等角視圖
    ]
    
    vmin = min(preds.min(), targets.min())
    vmax = max(preds.max(), targets.max())

    # --- Prediction ---
    for i, (elev, azim, title) in enumerate(views):
        ax = fig.add_subplot(2, 3, i + 1, projection='3d')
        p = ax.scatter(points[:, 0], points[:, 2], points[:, 1], c=preds, cmap='jet', s=2, vmin=vmin, vmax=vmax)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f'Pred - {title}')
        ax.axis('off')

    # --- Ground Truth ---
    for i, (elev, azim, title) in enumerate(views):
        ax = fig.add_subplot(2, 3, i + 4, projection='3d')
        p = ax.scatter(points[:, 0], points[:, 2], points[:, 1], c=targets, cmap='jet', s=2, vmin=vmin, vmax=vmax)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f'GT - {title}')
        ax.axis('off')

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(p, cax=cbar_ax, label='Heatmap Intensity')

    plt.tight_layout(rect=[0, 0, 0.9, 1])
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
    計算空間位置誤差 (支援多澆口 Multi-Gate)
    只要打中其中一個澆口，就算準確
    """
    batch_size = points.shape[0]
    
    # 1. Argmax Shift
    pred_max_idx = torch.argmax(preds, dim=1)
    pred_xyz = points[torch.arange(batch_size), pred_max_idx, :]
    
    batch_min_dists = []
    
    for b in range(batch_size):
        t = targets[b]
        p_xyz = pred_xyz[b]
        
        # 找出所有真實澆口候選點
        gt_mask = t > threshold
        
        if gt_mask.sum() > 0:
            gt_candidates = points[b][gt_mask]
            # 計算到最近的一個真實澆口的距離
            dists = torch.norm(gt_candidates - p_xyz, dim=1)
            min_dist = torch.min(dists).item()
        else:
            min_dist = 1.0 # 懲罰值
            
        batch_min_dists.append(min_dist)
        
    return np.mean(batch_min_dists), 0.0 # Weighted 暫時不使用

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1: m.inplace = True

def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='pointnext_part_seg_nocls_grid')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epoch', default=501, type=int)
    # [修正] 預設 LR 降為 0.0001 以穩定 STN
    parser.add_argument('--learning_rate', default=0.0001, type=float)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--log_dir', type=str, default=None)
    parser.add_argument('--decay_rate', type=float, default=1e-3)
    parser.add_argument('--npoint', type=int, default=4096)
    parser.add_argument('--normal', action='store_true', default=False)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--grid_num', type=int, default=4)
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
    
    visual_dir = exp_dir.joinpath('visual_results/')
    visual_dir.mkdir(exist_ok=True)

    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler = logging.FileHandler(f'{log_dir}/{args.model}.txt')
    file_handler.setFormatter(formatter); logger.addHandler(file_handler)

    log_string('PARAMETER ...')
    log_string(args)

    TRAIN_DATASET = PartNormalDataset(root=args.data_dir, npoints=args.npoint, split='trainval', class_choice=None, normal_channel=args.normal, augment=True, grid_num=args.grid_num)
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)

    TEST_DATASET = PartNormalDataset(root=args.data_dir, npoints=args.npoint, split='test', class_choice=None, normal_channel=args.normal, augment=False, grid_num=args.grid_num)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=4)

    MODEL = importlib.import_module(args.model)
    classifier = MODEL.get_model(2, normal_channel=args.normal, grid_num=args.grid_num).cuda()
    classifier.apply(inplace_relu)
    
    # [修正] 使用 MODEL 中定義的 Loss，包含 STN 正則化與 Warmup
    criterion = MODEL.get_loss().cuda()
    
    optimizer = torch.optim.Adam(classifier.parameters(), lr=args.learning_rate, weight_decay=args.decay_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch, eta_min=1e-6)

    history = {'train_loss': [], 'test_loss': [], 'test_mae': [], 'test_mse': [], 'grid_acc': [], 'shift_err': [], 'w_shift_err': []}
    best_mae = float('inf')

    # ===================================================
    # Smart Rewind & Punish 機制變數初始化
    # ===================================================
    patience_counter = 0
    patience_limit = 15          
    overfit_protection_count = 0 
    max_protections = 3           
    current_weight_decay = args.decay_rate 
    # ===================================================

    for epoch in range(args.epoch):
        log_string(f'**** Epoch {epoch + 1} ****')
        classifier.train()
        train_loss_epoch = []
        
        for i, (points, label, target, grid_gt) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
            optimizer.zero_grad()
            points, target = points.float().cuda(), target.float().cuda()
            
            # [修正] 接收 trans_feat
            seg_pred, trans_feat = classifier(points.transpose(2, 1))
            pred_intensity = seg_pred
            
            # [修正] 呼叫 MODEL.get_loss，傳入 epoch 和 trans_feat
            loss = criterion(pred_intensity, target, trans_feat, epoch)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss_epoch.append(loss.item())
            
        history['train_loss'].append(np.mean(train_loss_epoch))

        with torch.no_grad():
            classifier.eval()
            m = {'test_loss': [], 'mae': [], 'mse': [], 'grid_acc': [], 'shift_err': [], 'w_shift_err': []}

            for batch_id, (points, label, target, grid_gt) in enumerate(testDataLoader):
                points_cuda, target_cuda = points.float().cuda(), target.float().cuda()
                
                # [修正] 接收 trans_feat
                seg_pred, trans_feat = classifier(points_cuda.transpose(2, 1))
                pred_intensity = seg_pred
                
                # [修正] 計算驗證 Loss (同樣傳入 epoch)
                val_loss = criterion(pred_intensity, target_cuda, trans_feat, epoch)
                m['test_loss'].append(val_loss.item())
                
                # 計算空間誤差
                s_err, w_s_err = get_spatial_metrics(points_cuda, pred_intensity, target_cuda)
                m['shift_err'].append(s_err)
                m['w_shift_err'].append(w_s_err)

                mae, mse = get_regression_stats(pred_intensity.cpu().numpy(), target.numpy())
                m['mae'].append(mae)
                m['mse'].append(mse)
                
                # [探針與視覺化]
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
                    
                    if (epoch + 1) % args.vis_freq == 0 or (epoch + 1) == args.epoch:
                        save_name = visual_dir.joinpath(f'epoch_{epoch+1}_batch0.png')
                        v_points = points[0].numpy()
                        v_preds = pred_intensity[0].cpu().numpy()
                        v_targets = target[0].numpy()
                        log_string(f'Saving visualization to {save_name} ...')
                        visualize_regression_batch(v_points, v_preds, v_targets, save_name, epoch+1)
                    
                    log_string(f'-------------------------------')

            epoch_mae = np.mean(m['mae'])
            epoch_mse = np.mean(m['mse'])
            epoch_shift_err = np.mean(m['shift_err'])
            
            history['test_loss'].append(np.mean(m['test_loss']))
            history['test_mae'].append(epoch_mae)
            history['test_mse'].append(epoch_mse)
            history['shift_err'].append(epoch_shift_err)
            
            log_string(f'Test Total Loss: {history["test_loss"][-1]:.4f}')
            log_string(f'MAE: {epoch_mae:.5f} | MSE: {epoch_mse:.5f}')
            log_string(f'Shift Error (Nearest Gate): {epoch_shift_err:.4f}')
            
            plot_performance(exp_dir, history)
            
            # ===========================================================
            # Smart Rewind & Punish 邏輯
            # ===========================================================
            if epoch_mae <= best_mae:
                best_mae = epoch_mae
                patience_counter = 0 
                
                # 存檔 (確保轉為 float)
                torch.save({'model_state_dict': classifier.state_dict(), 'mae': float(best_mae)}, str(checkpoints_dir) + '/best_regression_model.pth')
                log_string(f'Saving Model with Lowest MAE: {best_mae:.5f}')
            
            else:
                patience_counter += 1
                log_string(f'Warning: Validation stagnated for {patience_counter}/{patience_limit} epochs.')

                if patience_counter >= patience_limit:
                    if overfit_protection_count < max_protections:
                        log_string('!!!! DETECTED STAGNATION / OVERFITTING !!!!')
                        log_string('>>>> Initiating Smart Rewind & Punish Protocol...')
                        
                        # 1. Rewind
                        checkpoint = torch.load(str(checkpoints_dir) + '/best_regression_model.pth', weights_only=False)
                        classifier.load_state_dict(checkpoint['model_state_dict'])
                        log_string('>>>> Model rewound to best state.')

                        # 2. Punish
                        current_weight_decay = current_weight_decay * 2.0 
                        current_weight_decay = min(current_weight_decay, 0.1) 
                        
                        for param_group in optimizer.param_groups:
                            param_group['weight_decay'] = current_weight_decay
                        
                        log_string(f'>>>> Regularization increased! New Weight Decay: {current_weight_decay}')

                        # 3. Warm Restart
                        restart_lr = args.learning_rate * 0.5
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = restart_lr
                        
                        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch - epoch, eta_min=1e-6)
                        
                        log_string('>>>> Learning Rate Restarted. Resume Training...')
                        
                        patience_counter = 0
                        overfit_protection_count += 1
                        
                    else:
                        log_string('!!!! Maximum protections reached. Stopping training early. !!!!')
                        break
            # ===========================================================
        
        scheduler.step()

if __name__ == '__main__':
    args = parse_args()
    main(args)