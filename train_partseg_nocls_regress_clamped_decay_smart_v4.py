"""
Author: Benny (Universal Regression Edition) - Final Output with Original Filenames
Date: 2025 Revised 
Description: 
    - [修復] 補回 --optimizer 參數定義，並支援 AdamW
    - [功能] out_data 結構: 分為 train/ 和 test/，其下再分 prediction/ 和 ground_truth/
    - [功能] 輸出檔名直接使用資料集原始檔名 (例如 Casting_1001.txt)
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
import pandas as pd
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
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(f'Epoch {epoch} - Prediction vs Ground Truth', fontsize=16)
    views = [(0, 0, 'Front View'), (90, -90, 'Top View'), (30, 45, 'Iso View')]
    vmin = min(preds.min(), targets.min())
    vmax = max(preds.max(), targets.max())

    for i, (elev, azim, title) in enumerate(views):
        ax = fig.add_subplot(2, 3, i + 1, projection='3d')
        ax.scatter(points[:, 0], points[:, 2], points[:, 1], c=preds, cmap='jet', s=2, vmin=vmin, vmax=vmax)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f'Pred - {title}'); ax.axis('off')

    for i, (elev, azim, title) in enumerate(views):
        ax = fig.add_subplot(2, 3, i + 4, projection='3d')
        ax.scatter(points[:, 0], points[:, 2], points[:, 1], c=targets, cmap='jet', s=2, vmin=vmin, vmax=vmax)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f'GT - {title}'); ax.axis('off')

    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.savefig(save_path, dpi=150); plt.close(fig)

def plot_error_distribution(exp_dir, all_errors, epoch, tolerance=0.1):
    plt.figure(figsize=(10, 6))
    plt.hist(all_errors, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(x=tolerance, color='r', linestyle='--', label=f'Tolerance ({tolerance})')
    plt.title(f'Epoch {epoch} - Shift Error Distribution')
    plt.xlabel('Spatial Shift Error'); plt.ylabel('Count')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(exp_dir, 'error_dist.png')); plt.close()

# ==========================================
#      數值計算工具函數
# ==========================================
def get_regression_stats(preds, targets):
    mae = np.mean(np.abs(preds - targets))
    mse = np.mean((preds - targets) ** 2)
    return mae, mse

def get_spatial_metrics(points, preds, targets, threshold=0.5):
    batch_size = points.shape[0]
    tolerance = 0.1 
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
            dists = torch.norm(gt_candidates - p_xyz, dim=1)
            min_dist = torch.min(dists).item()
        else:
            min_dist = 1.0 
        batch_min_dists.append(min_dist)
        if min_dist < tolerance: hits += 1
    return np.mean(batch_min_dists), hits, batch_min_dists

def inplace_relu(m):
    if m.__class__.__name__.find('ReLU') != -1: m.inplace = True

def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='pointnext_part_seg_nocls_grid')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epoch', default=501, type=int)
    parser.add_argument('--learning_rate', default=0.0001, type=float)
    parser.add_argument('--gpu', type=str, default='0')
    
    # [修復] 這裡補回了 optimizer 參數
    parser.add_argument('--optimizer', type=str, default='Adam')
    
    parser.add_argument('--log_dir', type=str, default=None)
    parser.add_argument('--decay_rate', type=float, default=1e-3)
    parser.add_argument('--npoint', type=int, default=4096)
    parser.add_argument('--normal', action='store_true', default=False)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--grid_num', type=int, default=4)
    parser.add_argument('--vis_freq', type=int, default=5)
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
        plt.plot(epochs, history['success_rate'], 'm--', label='Success Rate')
        plt.title('Spatial Metrics'); plt.legend(); plt.grid(True); plt.savefig(os.path.join(exp_dir, 'spatial_error.png')); plt.close()

# ====================================================================
# [核心功能] 依照原始檔名儲存推論結果
# ====================================================================
def save_dataset_inference(model, dataset, batch_size, split_name, output_root_dir, log_func):
    """
    對指定的 dataset 進行推論並存檔
    dataset: PartNormalDataset 物件
    split_name: 'train' 或 'test'
    """
    log_func(f'--- Processing split: [{split_name}] (Count: {len(dataset)}) ---')
    
    # [關鍵] 建立一個不打亂 (shuffle=False) 的 Loader，確保順序與 dataset.datapath 一致
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # 建立目錄結構
    split_dir = output_root_dir.joinpath(split_name)
    pred_dir = split_dir.joinpath('prediction')
    gt_dir = split_dir.joinpath('ground_truth')
    
    split_dir.mkdir(exist_ok=True)
    pred_dir.mkdir(exist_ok=True)
    gt_dir.mkdir(exist_ok=True)
    
    model.eval()
    results_list = []
    tolerance = 0.1
    
    # 這裡的 global_idx 對應到 dataset.datapath 的 index
    current_idx = 0
    
    with torch.no_grad():
        for batch_id, (points, label, target, grid_gt) in enumerate(tqdm(dataloader, desc=f"Saving {split_name}")):
            points_cuda, target_cuda = points.float().cuda(), target.float().cuda()
            
            seg_pred, trans_feat = model(points_cuda.transpose(2, 1))
            
            _, _, batch_dists = get_spatial_metrics(points_cuda, seg_pred, target_cuda, threshold=0.5)
            
            points_np = points.numpy()
            pred_np = seg_pred.cpu().numpy()
            target_np = target_cuda.cpu().numpy()
            
            batch_size_current = points.shape[0]
            
            for i in range(batch_size_current):
                # 從 dataset 中獲取原始檔名
                try:
                    _, original_path = dataset.datapath[current_idx]
                    filename_stem = os.path.splitext(os.path.basename(original_path))[0]
                    filename = f"{filename_stem}.txt"
                except IndexError:
                    filename = f"Unknown_{current_idx}.txt"
                    log_func(f"Error retrieving filename for index {current_idx}")

                spatial_err = batch_dists[i]
                is_success = spatial_err < tolerance
                
                p_flat = pred_np[i].flatten()
                t_flat = target_np[i].flatten()
                
                mae = np.mean(np.abs(p_flat - t_flat))
                mse = np.mean((p_flat - t_flat) ** 2)
                
                data_pred = np.hstack((points_np[i], p_flat.reshape(-1, 1)))
                np.savetxt(pred_dir.joinpath(filename), data_pred, fmt='%.6f', delimiter=' ')
                
                data_gt = np.hstack((points_np[i], t_flat.reshape(-1, 1)))
                np.savetxt(gt_dir.joinpath(filename), data_gt, fmt='%.6f', delimiter=' ')
                
                results_list.append({
                    'FileName': filename_stem,
                    'Split': split_name,
                    'Success': is_success,
                    'Spatial_Error': spatial_err,
                    'MAE': mae,
                    'MSE': mse
                })

                current_idx += 1
                
    return results_list

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
    
    criterion = MODEL.get_loss().cuda()
    
    # [修復] 根據參數選擇優化器
    if args.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(classifier.parameters(), lr=args.learning_rate, weight_decay=args.decay_rate)
        log_string("Using Optimizer: AdamW")
    else:
        optimizer = torch.optim.Adam(classifier.parameters(), lr=args.learning_rate, weight_decay=args.decay_rate)
        log_string("Using Optimizer: Adam")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch, eta_min=1e-6)

    history = {'train_loss': [], 'test_loss': [], 'test_mae': [], 'test_mse': [], 'shift_err': [], 'success_rate': []}
    
    best_success_rate = -1.0
    best_mse = float('inf')
    best_shift_err = float('inf')

    patience_counter = 0
    patience_limit = 15           
    overfit_protection_count = 0 
    max_protections = 5            
    current_weight_decay = args.decay_rate 

    try:
        for epoch in range(args.epoch):
            log_string(f'**** Epoch {epoch + 1} ****')
            classifier.train()
            train_loss_epoch = []
            
            for i, (points, label, target, grid_gt) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
                optimizer.zero_grad()
                points, target = points.float().cuda(), target.float().cuda()
                
                seg_pred, trans_feat = classifier(points.transpose(2, 1))
                loss = criterion(seg_pred, target, trans_feat, epoch)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss_epoch.append(loss.item())
                
            history['train_loss'].append(np.mean(train_loss_epoch))

            with torch.no_grad():
                classifier.eval()
                m = {'test_loss': [], 'mae': [], 'mse': [], 'shift_err': []}
                total_hits = 0; total_samples = 0
                all_sample_errors = []

                for batch_id, (points, label, target, grid_gt) in enumerate(testDataLoader):
                    points_cuda, target_cuda = points.float().cuda(), target.float().cuda()
                    seg_pred, trans_feat = classifier(points_cuda.transpose(2, 1))
                    
                    val_loss = criterion(seg_pred, target_cuda, trans_feat, epoch)
                    m['test_loss'].append(val_loss.item())
                    
                    s_err, hits, raw_dists = get_spatial_metrics(points_cuda, seg_pred, target_cuda)
                    m['shift_err'].append(s_err)
                    total_hits += hits
                    total_samples += points.shape[0]
                    all_sample_errors.extend(raw_dists)

                    mae, mse = get_regression_stats(seg_pred.cpu().numpy(), target.numpy())
                    m['mae'].append(mae); m['mse'].append(mse)
                    
                    if batch_id < 8 and ((epoch + 1) % args.vis_freq == 0 or (epoch + 1) == args.epoch):
                         SAMPLES_TO_SAVE = min(points.shape[0], 5)
                         for s_idx in range(SAMPLES_TO_SAVE):
                             visualize_regression_batch(points[s_idx].numpy(), seg_pred[s_idx].cpu().numpy(), target[s_idx].numpy(), 
                                                        visual_dir.joinpath(f'epoch_{epoch+1}_batch{batch_id}_sample{s_idx}.png'), epoch+1)

                epoch_mae = np.mean(m['mae'])
                epoch_mse = np.mean(m['mse'])
                epoch_shift_err = np.mean(m['shift_err'])
                epoch_success_rate = total_hits / total_samples
                
                history['test_loss'].append(np.mean(m['test_loss']))
                history['test_mae'].append(epoch_mae)
                history['test_mse'].append(epoch_mse)
                history['shift_err'].append(epoch_shift_err)
                history['success_rate'].append(epoch_success_rate)
                
                if (epoch + 1) % args.vis_freq == 0:
                    plot_error_distribution(exp_dir, all_sample_errors, epoch + 1)

                log_string(f'Test Total Loss: {history["test_loss"][-1]:.4f}')
                log_string(f'MAE: {epoch_mae:.5f} | Shift Err: {epoch_shift_err:.4f}')
                log_string(f'Success Rate: {epoch_success_rate*100:.2f}%')
                
                plot_performance(exp_dir, history)
                
                is_best = False
                if epoch_success_rate > best_success_rate: is_best = True
                elif epoch_success_rate == best_success_rate and epoch_mse < (best_mse - 1e-5): is_best = True

                if is_best:
                    best_success_rate = epoch_success_rate; best_mse = epoch_mse; best_shift_err = epoch_shift_err
                    patience_counter = 0 
                    torch.save({
                        'model_state_dict': classifier.state_dict(), 
                        'success_rate': float(best_success_rate), 'mse': float(best_mse), 'shift_err': float(epoch_shift_err)
                    }, str(checkpoints_dir) + '/best_regression_model.pth')
                    log_string(f'Saving Model with Best Performance (SR: {best_success_rate*100:.2f}%)')
                else:
                    patience_counter += 1
                    if patience_counter >= patience_limit:
                        if overfit_protection_count < max_protections:
                            log_string('!!!! DETECTED STAGNATION. Smart Rewind !!!!')
                            checkpoint = torch.load(str(checkpoints_dir) + '/best_regression_model.pth', weights_only=False)
                            classifier.load_state_dict(checkpoint['model_state_dict'])
                            best_success_rate = checkpoint['success_rate']
                            current_weight_decay = min(current_weight_decay * 2.0, 0.1)
                            for pg in optimizer.param_groups: pg['weight_decay'] = current_weight_decay
                            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch - epoch, eta_min=1e-6)
                            patience_counter = 0; overfit_protection_count += 1
                        else:
                            log_string('!!!! Stopping training early. !!!!'); break
            scheduler.step()

    except KeyboardInterrupt:
        log_string('Training interrupted by user.')

    # ====================================================================================
    # [最後階段] 載入最佳模型，並針對 Train 與 Test 分別進行推論與存檔
    # ====================================================================================
    log_string('')
    log_string('=======================================================')
    log_string(' Saving Final Results with ORIGINAL Filenames ... ')
    log_string('=======================================================')

    out_data_dir = exp_dir.joinpath('out_data/')
    out_data_dir.mkdir(exist_ok=True)
    
    best_model_path = str(checkpoints_dir) + '/best_regression_model.pth'
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, weights_only=False)
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string(f'Loaded Best Model (SR: {checkpoint["success_rate"]*100:.2f}%)')

    # 1. 處理 Test Set 
    # (雖然原本 TEST_DATASET 就是 augment=False，但為了保險建議重新宣告，或直接沿用 TEST_DATASET 亦可)
    log_string('--- Processing Test Set (Clean) ---')
    TEST_DATASET_CLEAN = PartNormalDataset(
        root=args.data_dir, 
        npoints=args.npoint, 
        split='test', 
        class_choice=None, 
        normal_channel=args.normal, 
        augment=False,   # <--- 確保這裡是 False
        grid_num=args.grid_num
    )
    test_results = save_dataset_inference(classifier, TEST_DATASET_CLEAN, args.batch_size, 'test', out_data_dir, log_string)
    
    # 2. 處理 Train Set [關鍵修改在這裡！]
    # 原本的 TRAIN_DATASET 有開 augment=True，這裡我們必須建立一個新的、關閉 augment 的 Dataset
    log_string('--- Processing Train Set (Clean) ---')
    TRAIN_DATASET_CLEAN = PartNormalDataset(
        root=args.data_dir, 
        npoints=args.npoint, 
        split='trainval', # 注意確認你的 split 名稱是否正確 (通常是 train 或 trainval)
        class_choice=None, 
        normal_channel=args.normal, 
        augment=False,    # <--- [重點] 強制關閉資料增強，角度才不會亂轉
        grid_num=args.grid_num
    )
    train_results = save_dataset_inference(classifier, TRAIN_DATASET_CLEAN, args.batch_size, 'train', out_data_dir, log_string)

    # 3. 彙整 CSV 報表
    all_results = test_results + train_results
    if len(all_results) > 0:
        df_result = pd.DataFrame(all_results)
        output_csv_path = os.path.join(log_dir, 'final_inference_report.csv')
        df_result.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
        
        log_string(f'Saved Detailed Report to: {output_csv_path}')
        log_string(f'Saved All Files to: {out_data_dir} (Structure: train/test -> prediction/ground_truth)')

if __name__ == '__main__':
    args = parse_args()
    main(args)