"""
Author: Benny (Universal Regression Edition) - Enhanced Visualization & Anti-Overfitting
Date: 2025 Revised for Regression Task (With Spatial Metrics & Smart Rewind)
Description: 
    - [修正] 整合 STN 架構與 Loss (接收 trans_feat)
    - [修正] 評估指標改為 Success Rate (成功率) 以避免被背景雜訊誤導
    - [新增] Error Distribution Plot: 可視化所有 Batch 的誤差分佈
    - [新增] 智慧回溯機制: 當 Success Rate 停滯時自動讀取舊檔並加重懲罰
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

# 設定 Matplotlib 後端
plt.switch_backend('agg') 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

# ==========================================
#      視覺化工具函數
# ==========================================
def visualize_regression_batch(points, preds, targets, save_path, epoch):
    """ 繪製 3D 點雲回歸對比圖 (多視角) """
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(f'Epoch {epoch} - Prediction vs Ground Truth', fontsize=16)

    views = [(0, 0, 'Front View'), (90, -90, 'Top View'), (30, 45, 'Iso View')]
    vmin = min(preds.min(), targets.min())
    vmax = max(preds.max(), targets.max())

    for i, (elev, azim, title) in enumerate(views):
        ax = fig.add_subplot(2, 3, i + 1, projection='3d')
        p = ax.scatter(points[:, 0], points[:, 2], points[:, 1], c=preds, cmap='jet', s=2, vmin=vmin, vmax=vmax)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f'Pred - {title}')
        ax.axis('off')

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

def plot_error_distribution(exp_dir, all_errors, epoch, tolerance=0.1):
    """
    [新增] 繪製所有 Batch 的誤差分佈直方圖
    這能讓你看到整體數據集的表現，而不僅僅是平均值
    """
    plt.figure(figsize=(10, 6))
    plt.hist(all_errors, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(x=tolerance, color='r', linestyle='--', label=f'Tolerance ({tolerance})')
    plt.title(f'Epoch {epoch} - Shift Error Distribution (All Batches)')
    plt.xlabel('Spatial Shift Error')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(exp_dir, 'error_dist.png')) # 每次覆蓋最新的
    plt.close()

# ==========================================
#      數值計算工具函數
# ==========================================
def get_regression_stats(preds, targets):
    mae = np.mean(np.abs(preds - targets))
    mse = np.mean((preds - targets) ** 2)
    return mae, mse

def get_spatial_metrics(points, preds, targets, threshold=0.5):
    """
    計算空間位置誤差 & 成功率
    回傳: 
    1. batch_avg_dist: 該 Batch 的平均誤差
    2. hits: 該 Batch 中誤差小於 tolerance 的樣本數
    3. raw_dists: 該 Batch 中每個樣本的具體誤差 (用於畫分佈圖)
    """
    batch_size = points.shape[0]
    tolerance = 0.1 # 成功門檻 (約 5%~10% 尺寸)
    
    pred_max_idx = torch.argmax(preds, dim=1)
    pred_xyz = points[torch.arange(batch_size), pred_max_idx, :]
    
    batch_min_dists = []
    hits = 0
    
    for b in range(batch_size):
        t = targets[b]
        p_xyz = pred_xyz[b]
        
        gt_mask = t > threshold
        if gt_mask.sum() > 0:
            gt_candidates = points[b][gt_mask]
            # 計算到最近的一個真實澆口的距離 (解決多澆口問題)
            dists = torch.norm(gt_candidates - p_xyz, dim=1)
            min_dist = torch.min(dists).item()
        else:
            min_dist = 1.0 # 懲罰值
            
        batch_min_dists.append(min_dist)
        
        if min_dist < tolerance:
            hits += 1
            
    return np.mean(batch_min_dists), hits, batch_min_dists

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1: m.inplace = True

def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='pointnext_part_seg_nocls_grid')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epoch', default=501, type=int)
    # [建議] 降低 LR 以穩定 STN
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
        plt.plot(epochs, history['shift_err'], 'k-', label='Avg Shift Error')
        plt.plot(epochs, history['success_rate'], 'm--', label='Success Rate') # 新增成功率曲線
        plt.title('Spatial Metrics'); plt.legend(); plt.grid(True); plt.savefig(os.path.join(exp_dir, 'spatial_error.png')); plt.close()

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
    
    # [修正] 使用 MODEL 內建的 Loss (含 STN 正則化 & Warmup)
    criterion = MODEL.get_loss().cuda()
    
    optimizer = torch.optim.Adam(classifier.parameters(), lr=args.learning_rate, weight_decay=args.decay_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch, eta_min=1e-6)

    history = {'train_loss': [], 'test_loss': [], 'test_mae': [], 'test_mse': [], 'shift_err': [], 'success_rate': []}
    
    # 改用 Success Rate 作為最佳模型指標 (初始值 -1)
    best_success_rate = -1.0
    best_shift_err = float('inf')

    # ===================================================
    # Smart Rewind & Punish 機制變數
    # ===================================================
    patience_counter = 0
    patience_limit = 15          
    overfit_protection_count = 0 
    max_protections = 5           
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
            
            # [修正] 傳入 epoch 與 trans_feat 給 Loss
            loss = criterion(pred_intensity, target, trans_feat, epoch)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss_epoch.append(loss.item())
            
        history['train_loss'].append(np.mean(train_loss_epoch))

        with torch.no_grad():
            classifier.eval()
            m = {'test_loss': [], 'mae': [], 'mse': [], 'shift_err': []}
            
            total_hits = 0
            total_samples = 0
            all_sample_errors = [] 

            # [設定] 你想看前幾個 Batch 的圖？ (這裡設為 4)
            VIS_BATCH_COUNT = 8 

            for batch_id, (points, label, target, grid_gt) in enumerate(testDataLoader):
                points_cuda, target_cuda = points.float().cuda(), target.float().cuda()
                
                seg_pred, trans_feat = classifier(points_cuda.transpose(2, 1))
                pred_intensity = seg_pred
                
                val_loss = criterion(pred_intensity, target_cuda, trans_feat, epoch)
                m['test_loss'].append(val_loss.item())
                
                s_err, hits, raw_dists = get_spatial_metrics(points_cuda, pred_intensity, target_cuda)
                
                m['shift_err'].append(s_err)
                total_hits += hits
                total_samples += points.shape[0]
                all_sample_errors.extend(raw_dists)

                mae, mse = get_regression_stats(pred_intensity.cpu().numpy(), target.numpy())
                m['mae'].append(mae)
                m['mse'].append(mse)
                
                # ========================================================
                # [修正] 多 Batch 視覺化與探針 (Visualizing Multiple Batches)
                # ========================================================
                if batch_id < VIS_BATCH_COUNT:
                    # 1. 計算該 Batch 的統計數據 (Probe) - 這部分保持看整體
                    batch_pred_max = pred_intensity.max().item()
                    batch_gt_max = target_cuda.max().item()
                    mask = (target_cuda > 0.1)
                    avg_pred_on_gate = pred_intensity[mask].mean().item() if mask.sum() > 0 else 0.0
                    
                    log_string(f'---- PROBE [Epoch {epoch+1} | Batch {batch_id}] ----')
                    log_string(f'GT Max: {batch_gt_max:.4f} | Pred Max: {batch_pred_max:.4f}')
                    log_string(f'Gate Avg Pred: {avg_pred_on_gate:.4f}')
                    log_string(f'Shift Err (Avg): {s_err:.4f}')
                    
                    # 2. 存圖邏輯：只有在符合頻率時執行
                    if (epoch + 1) % args.vis_freq == 0 or (epoch + 1) == args.epoch:
                        
                        # [設定] 每個 Batch 你想存幾個樣本？(避免存太多圖)
                        # 設為 4 代表會存 sample 0, 1, 2, 3
                        SAMPLES_TO_SAVE = 30 
                        
                        # 確保不會超過實際 batch size (例如最後一個 batch 可能只有 2 個)
                        actual_samples = min(points.shape[0], SAMPLES_TO_SAVE)
                        
                        for sample_idx in range(actual_samples):
                            # [關鍵修正] 這裡不再寫死 [0]，而是用 [sample_idx]
                            v_points = points[sample_idx].numpy()
                            v_preds = pred_intensity[sample_idx].cpu().numpy()
                            v_targets = target[sample_idx].numpy()
                            
                            # 檔名加入 sample_id
                            save_name = visual_dir.joinpath(f'epoch_{epoch+1}_batch{batch_id}_sample{sample_idx}.png')
                            
                            log_string(f'Saving visualization to {save_name} ...')
                            visualize_regression_batch(v_points, v_preds, v_targets, save_name, epoch+1)
                        
                    log_string(f'-------------------------------------------')
                # ========================================================

            # 計算 epoch 級別的指標
            epoch_mae = np.mean(m['mae'])
            epoch_mse = np.mean(m['mse'])
            epoch_shift_err = np.mean(m['shift_err'])
            epoch_success_rate = total_hits / total_samples # 成功率
            
            history['test_loss'].append(np.mean(m['test_loss']))
            history['test_mae'].append(epoch_mae)
            history['test_mse'].append(epoch_mse)
            history['shift_err'].append(epoch_shift_err)
            history['success_rate'].append(epoch_success_rate)
            
            # [新增] 繪製所有 Batch 的誤差分佈圖
            if (epoch + 1) % args.vis_freq == 0:
                plot_error_distribution(exp_dir, all_sample_errors, epoch + 1)

            log_string(f'Test Total Loss: {history["test_loss"][-1]:.4f}')
            log_string(f'MAE: {epoch_mae:.5f} | Shift Err: {epoch_shift_err:.4f}')
            log_string(f'Success Rate (Err < 0.1): {epoch_success_rate*100:.2f}%')
            
            plot_performance(exp_dir, history)
            
            # 判斷是否更新最佳模型
            is_best = False
            
            if epoch_success_rate > best_success_rate:
                # 情況 A: 成功率創新高 -> 直接存
                is_best = True
            elif epoch_success_rate == best_success_rate:
                # 情況 B: 成功率持平，檢查精確度 (MSE)
                # 如果 MSE 變小了，代表熱力圖變尖了 -> 也要存
                if epoch_mse < (best_mse - 1e-5): # 加一點容忍度
                    is_best = True
                    log_string(f'>> Success Rate tied ({epoch_success_rate*100:.1f}%), but MSE improved ({best_mse:.5f} -> {epoch_mse:.5f}). Saving...')

            if is_best:
                best_success_rate = epoch_success_rate
                best_mse = epoch_mse # 記得更新 best_mse
                best_shift_err = epoch_shift_err
                patience_counter = 0 
                
                torch.save({
                    'model_state_dict': classifier.state_dict(), 
                    'success_rate': float(best_success_rate),
                    'mse': float(best_mse),
                    'shift_err': float(epoch_shift_err)
                }, str(checkpoints_dir) + '/best_regression_model.pth')
                
                log_string(f'Saving Model with Best Performance (SR: {best_success_rate*100:.2f}%, MSE: {best_mse:.5f})')
            
            else:
                patience_counter += 1
                log_string(f'Warning: Performance stagnated for {patience_counter}/{patience_limit} epochs.')
                if patience_counter >= patience_limit:
                    if overfit_protection_count < max_protections:
                        log_string('!!!! DETECTED STAGNATION / OVERFITTING !!!!')
                        log_string('>>>> Initiating Smart Rewind & Punish Protocol...')
                        
                        # 1. Rewind
                        checkpoint = torch.load(str(checkpoints_dir) + '/best_regression_model.pth', weights_only=False)
                        classifier.load_state_dict(checkpoint['model_state_dict'])
                        best_success_rate = checkpoint['success_rate']
                        log_string(f'>>>> Model rewound to best state (SR: {best_success_rate*100:.2f}%).')

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