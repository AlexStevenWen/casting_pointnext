"""
Author: Benny (Universal Segmentation Edition) - Final Output with Original Filenames
Date: 2025 Revised for Binary Segmentation
Description: 
    - [模式] 二元分割 (Binary Segmentation)
    - [指標] mIoU (Intersection over Union) & Accuracy
    - [功能] out_data 結構: 分為 train/ 和 test/，其下再分 prediction/ 和 ground_truth/
    - [功能] 輸出檔名直接使用資料集原始檔名
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
#      視覺化工具函數 (分割版)
# ==========================================
def visualize_segmentation_batch(points, preds, targets, save_path, epoch):
    """
    preds: (N,) 0~1 float probability
    targets: (N,) 0 or 1 integer/float
    """
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(f'Epoch {epoch} - Segmentation (Pred vs GT)', fontsize=16)
    views = [(0, 0, 'Front View'), (90, -90, 'Top View'), (30, 45, 'Iso View')]
    
    # 使用 coolwarm colormap: 0(藍)=背景, 1(紅)=前景
    cmap = 'coolwarm' 

    for i, (elev, azim, title) in enumerate(views):
        ax = fig.add_subplot(2, 3, i + 1, projection='3d')
        # Pred 顯示機率值或二值化後的結果
        ax.scatter(points[:, 0], points[:, 2], points[:, 1], c=preds, cmap=cmap, s=2, vmin=0, vmax=1)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f'Pred (Prob) - {title}'); ax.axis('off')

    for i, (elev, azim, title) in enumerate(views):
        ax = fig.add_subplot(2, 3, i + 4, projection='3d')
        ax.scatter(points[:, 0], points[:, 2], points[:, 1], c=targets, cmap=cmap, s=2, vmin=0, vmax=1)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f'GT (Mask) - {title}'); ax.axis('off')

    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.savefig(save_path, dpi=150); plt.close(fig)

# ==========================================
#      數值計算工具函數 (分割版)
# ==========================================
def get_segmentation_metrics(preds, targets, threshold=0.5):
    """
    計算 Accuracy 和 IoU
    preds: [B, N] (probabilities)
    targets: [B, N] (0 or 1)
    """
    pred_bin = (preds > threshold).astype(int)
    target_bin = targets.astype(int)

    # 1. Accuracy
    correct = np.sum(pred_bin == target_bin)
    total = pred_bin.size
    accuracy = correct / total

    # 2. IoU (Intersection over Union) for Foreground (Class 1)
    # 我們主要關心前景的重疊率
    intersection = np.sum(np.logical_and(pred_bin == 1, target_bin == 1), axis=1)
    union = np.sum(np.logical_or(pred_bin == 1, target_bin == 1), axis=1)
    
    # 避免除以 0
    iou = (intersection + 1e-6) / (union + 1e-6)
    
    return accuracy, np.mean(iou) # 回傳 batch 的平均 IoU

def inplace_relu(m):
    if m.__class__.__name__.find('ReLU') != -1: m.inplace = True

def parse_args():
    parser = argparse.ArgumentParser('Model')
    # 預設改為你常用的分割模型名稱
    parser.add_argument('--model', type=str, default='pointnext_part_seg_nocls_grid')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epoch', default=201, type=int) # 分割通常不需要那麼多 epoch，可自行調整
    parser.add_argument('--learning_rate', default=0.001, type=float) # 分割可以用稍大的 LR
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--log_dir', type=str, default=None)
    parser.add_argument('--decay_rate', type=float, default=1e-4)
    parser.add_argument('--npoint', type=int, default=4096)
    parser.add_argument('--normal', action='store_true', default=False)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--grid_num', type=int, default=4)
    parser.add_argument('--vis_freq', type=int, default=5)
    parser.add_argument('--use_smart_rewind', action='store_true', default=False, 
                        help='是否開啟 Smart Rewind (動態調整 Loss 與權重)')
    parser.add_argument('--patience', type=int, default=999, 
                        help='提早結束的容忍 Epoch 數。設為 999 代表永不提早結束，硬跑完。')
    parser.add_argument('--tolerance', type=float, default=0.1, help='空間誤差的容忍距離')
    parser.add_argument('--reg_threshold', type=float, default=0.5, help='回歸數值轉為 Mask 的閾值')
    return parser.parse_args()

def plot_performance(exp_dir, history):
    epochs = range(1, len(history['train_loss']) + 1)
    
    # 1. 統一圖表：Loss 曲線
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history['train_loss'], 'r-', label='Train Loss')
    plt.plot(epochs, history['test_loss'], 'b--', label='Test Loss')
    plt.title('Loss Convergence')
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(exp_dir, 'loss.png')); plt.close()

    # 2. 統一圖表：mIoU 曲線
    if 'test_iou' in history:
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, history['test_iou'], 'g-', label='Test mIoU')
        plt.title('Segmentation Overlap (mIoU)')
        plt.xlabel('Epoch'); plt.ylabel('mIoU')
        plt.legend(); plt.grid(True)
        plt.savefig(os.path.join(exp_dir, 'metrics_iou.png')); plt.close()

    # 3. 統一圖表：空間誤差與成功率曲線
    if 'shift_err' in history and 'success_rate' in history:
        fig, ax1 = plt.subplots(figsize=(10, 5))
        color = 'tab:red'
        ax1.set_xlabel('Epoch'); ax1.set_ylabel('Shift Error (Distance)', color=color)
        ax1.plot(epochs, history['shift_err'], color=color, label='Avg Shift Error')
        ax1.tick_params(axis='y', labelcolor=color)
        
        ax2 = ax1.twinx()  
        color = 'tab:blue'
        ax2.set_ylabel('Success Rate', color=color)  
        ax2.plot(epochs, history['success_rate'], color=color, linestyle='--', label='Success Rate')
        ax2.tick_params(axis='y', labelcolor=color)
        
        # [修正這裡] 先設定標題 (加上 pad 增加間距)，畫網格，最後才呼叫 tight_layout()
        plt.title('Spatial Metrics', pad=15) 
        plt.grid(True, alpha=0.3)
        fig.tight_layout() 
        plt.savefig(os.path.join(exp_dir, 'metrics_spatial.png')); plt.close()
    # 4. 專屬圖表：Accuracy 曲線 (分割模型專用)
    if 'test_acc' in history:
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, history['test_acc'], 'k-', label='Test Accuracy')
        plt.title('Pixel/Point Accuracy')
        plt.xlabel('Epoch'); plt.ylabel('Accuracy')
        plt.legend(); plt.grid(True)
        plt.savefig(os.path.join(exp_dir, 'metrics_accuracy.png')); plt.close()

    # 5. 專屬圖表：MAE / MSE 曲線 (回歸模型專用)
    if 'test_mae' in history and 'test_mse' in history:
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, history['test_mae'], 'b-', label='Test MAE')
        plt.plot(epochs, history['test_mse'], 'g--', label='Test MSE')
        plt.title('Regression Error')
        plt.xlabel('Epoch'); plt.ylabel('Error Value')
        plt.legend(); plt.grid(True)
        plt.savefig(os.path.join(exp_dir, 'regression_error.png')); plt.close()
def plot_error_distribution(exp_dir, all_errors, epoch, tolerance=0.1):
    # 防呆機制：如果陣列是空的，直接印出警告並跳出
    if not all_errors or len(all_errors) == 0:
        print(f"Warning: all_errors is empty at epoch {epoch}. Skipping histogram plot.")
        return
        
    # 確保資料型態轉換為純 Python float，避免 numpy tensor 卡住
    clean_errors = [float(e) for e in all_errors]
    
    plt.figure(figsize=(10, 6))
    plt.hist(clean_errors, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(x=tolerance, color='r', linestyle='--', label=f'Tolerance ({tolerance})')
    plt.title(f'Epoch {epoch} - Shift Error Distribution')
    plt.xlabel('Spatial Shift Error'); plt.ylabel('Count')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(exp_dir, 'error_dist.png')); plt.close()
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
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    split_dir = output_root_dir.joinpath(split_name)
    pred_dir = split_dir.joinpath('prediction')
    gt_dir = split_dir.joinpath('ground_truth')
    
    split_dir.mkdir(exist_ok=True)
    pred_dir.mkdir(exist_ok=True)
    gt_dir.mkdir(exist_ok=True)
    
    model.eval()
    results_list = []
    
    current_idx = 0
    
    with torch.no_grad():
        for batch_id, (points, label, target, grid_gt) in enumerate(tqdm(dataloader, desc=f"Saving {split_name}")):
            points_cuda, target_cuda = points.float().cuda(), target.float().cuda()
            
            # Forward
            seg_pred, trans_feat = model(points_cuda.transpose(2, 1))
            
            points_np = points.numpy()
            pred_np = seg_pred.cpu().numpy()
            target_np = target_cuda.cpu().numpy()
            
            batch_size_current = points.shape[0]
            
            for i in range(batch_size_current):
                try:
                    _, original_path = dataset.datapath[current_idx]
                    filename_stem = os.path.splitext(os.path.basename(original_path))[0]
                    filename = f"{filename_stem}.txt"
                except IndexError:
                    filename = f"Unknown_{current_idx}.txt"

                # ===== 呼叫統一指標 (單筆資料) =====
                p_pts = points_np[i:i+1]
                p_pred = pred_np[i:i+1]
                p_tgt = target_np[i:i+1]
                
                single_iou, single_shift_err, hits, _ = get_unified_metrics(
                    p_pts, p_pred, p_tgt, mode='segmentation', tolerance=0.1
                )
                is_success = bool(hits > 0)

                # 準備存檔資料: x, y, z, pred_prob, gt_label
                p_flat = pred_np[i].flatten()
                t_flat = target_np[i].flatten()
                
                data_pred = np.hstack((points_np[i], p_flat.reshape(-1, 1)))
                np.savetxt(pred_dir.joinpath(filename), data_pred, fmt='%.6f', delimiter=' ')
                
                data_gt = np.hstack((points_np[i], t_flat.reshape(-1, 1)))
                np.savetxt(gt_dir.joinpath(filename), data_gt, fmt='%.6f', delimiter=' ')
                
                # ===== 統一 CSV 報表欄位 =====
                results_list.append({
                    'FileName': filename_stem,
                    'Split': split_name,
                    'Success': is_success,
                    'mIoU': single_iou,
                    'Shift_Error': single_shift_err
                })

                current_idx += 1
                
    return results_list
def get_unified_metrics(points, preds, targets, mode, tolerance=0.1, reg_threshold=0.5):
    """
    統一計算 IoU 與 空間偏移誤差
    points: [B, N, 3] numpy array
    preds: [B, N] numpy array (分割為 0~1 機率, 回歸為連續數值)
    targets: [B, N] numpy array (真實 0/1 Mask)
    mode: 'segmentation' 或 'regression'
    """
    batch_size = points.shape[0]
    batch_iou = []
    batch_shift_err = []
    hits = 0
    
    for b in range(batch_size):
        pts = points[b]
        pred = preds[b]
        gt = targets[b]
        
        # 1. 統一產生 Binarized Mask -------------------------
        if mode == 'segmentation':
            # 分割：機率大於 0.5 視為前景
            pred_mask = (pred > 0.5).astype(int)
        elif mode == 'regression':
            # 回歸：數值大於特定閾值視為前景 (可依據你的數值分佈調整 reg_threshold)
            pred_mask = (pred > reg_threshold).astype(int) 
            
        gt_mask = (gt > 0.5).astype(int)
        
        # 2. 計算 IoU ----------------------------------------
        intersection = np.sum(np.logical_and(pred_mask == 1, gt_mask == 1))
        union = np.sum(np.logical_or(pred_mask == 1, gt_mask == 1))
        iou = (intersection + 1e-6) / (union + 1e-6)
        batch_iou.append(iou)
        
        # 3. 計算 Spatial Shift Error (找中心點) -------------
        # 取 GT 真實標籤的幾何重心
        if np.sum(gt_mask) > 0:
            gt_center = np.mean(pts[gt_mask == 1], axis=0)
        else:
            gt_center = None
            
        # 取預測的中心點
        if mode == 'segmentation':
            if np.sum(pred_mask) > 0:
                pred_center = np.mean(pts[pred_mask == 1], axis=0) # 分割取預測 Mask 的重心
            else:
                pred_center = None
        elif mode == 'regression':
            pred_center = pts[np.argmax(pred)] # 回歸取數值最高 (熱力圖最亮) 的點作為靶心

        # 計算歐氏距離誤差
        if gt_center is not None and pred_center is not None:
            shift_err = np.linalg.norm(pred_center - gt_center)
            batch_shift_err.append(shift_err)
            if shift_err < tolerance:
                hits += 1
        else:
            # 如果預測完全沒抓到，給予最大懲罰值 (1.0)
            batch_shift_err.append(1.0)

    return np.mean(batch_iou), np.mean(batch_shift_err), hits, batch_shift_err
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
    
    # 載入資料集
    TRAIN_DATASET = PartNormalDataset(root=args.data_dir, npoints=args.npoint, split='trainval', class_choice=None, normal_channel=args.normal, augment=True, grid_num=args.grid_num)
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)

    TEST_DATASET = PartNormalDataset(root=args.data_dir, npoints=args.npoint, split='test', class_choice=None, normal_channel=args.normal, augment=False, grid_num=args.grid_num)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # 載入模型
    MODEL = importlib.import_module(args.model)
    # num_part=2 代表二元分割
    classifier = MODEL.get_model(2, normal_channel=args.normal, grid_num=args.grid_num).cuda()
    classifier.apply(inplace_relu)
    
    criterion = MODEL.get_loss().cuda()
    
    if args.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(classifier.parameters(), lr=args.learning_rate, weight_decay=args.decay_rate)
        log_string("Using Optimizer: AdamW")
    else:
        optimizer = torch.optim.Adam(classifier.parameters(), lr=args.learning_rate, weight_decay=args.decay_rate)
        log_string("Using Optimizer: Adam")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch, eta_min=1e-6)

    # 修改歷史記錄字典，移除 regression 指標，加入分割指標
    history = {'train_loss': [], 'test_loss': [], 'test_acc': [], 'test_iou': [], 'shift_err': [], 'success_rate': []}
    
    best_iou = -1.0
    best_epoch = 0
    patience_counter = 0
    # patience_limit 統一改由 args.patience 控制，所以這裡就不需要寫死了
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

            # --- Validation ---
            with torch.no_grad():
                classifier.eval()
                # 1. 字典裡補上 'acc' 的空陣列
                m = {'test_loss': [], 'iou': [], 'shift_err': [], 'acc': []} 
                total_hits = 0; total_samples = 0
                all_sample_errors = []
                
                for batch_id, (points, label, target, grid_gt) in enumerate(testDataLoader):
                    points_cuda, target_cuda = points.float().cuda(), target.float().cuda()
                    seg_pred, trans_feat = classifier(points_cuda.transpose(2, 1))
                    
                    val_loss = criterion(seg_pred, target_cuda, trans_feat, epoch)
                    m['test_loss'].append(val_loss.item())
                    
                    pts_np = points.numpy()
                    pred_np = seg_pred.cpu().numpy()
                    gt_np = target_cuda.cpu().numpy()
                    
                    iou, s_err, hits, raw_dists = get_unified_metrics(
                        pts_np, pred_np, gt_np, 
                        mode='segmentation', 
                        tolerance=args.tolerance # 記得使用剛剛新增的參數
                    )
                    
                    # 2. [補回計算 Accuracy]
                    pred_mask = (pred_np > 0.5).astype(int)
                    gt_mask = (gt_np > 0.5).astype(int)
                    acc = np.mean(pred_mask == gt_mask)
                    
                    m['iou'].append(iou)
                    m['shift_err'].append(s_err)
                    m['acc'].append(acc) # 把 acc 存進字典
                    total_hits += hits
                    total_samples += pts_np.shape[0]
                    all_sample_errors.extend(raw_dists)
                    
                    # (原本的視覺化程式碼保留...)
                    if batch_id < 5 and ((epoch + 1) % args.vis_freq == 0 or (epoch + 1) == args.epoch):
                         SAMPLES_TO_SAVE = min(points.shape[0], 3)
                         for s_idx in range(SAMPLES_TO_SAVE):
                             visualize_segmentation_batch(
                                 points[s_idx].numpy(), seg_pred[s_idx].cpu().numpy(), target[s_idx].numpy(), 
                                 visual_dir.joinpath(f'epoch_{epoch+1}_batch{batch_id}_sample{s_idx}.png'), epoch+1)

                # 3. [計算整個 epoch 的平均]
                epoch_acc = np.mean(m['acc']) # 這裡就不會再報 NameError 了！
                epoch_iou = np.mean(m['iou'])
                epoch_shift_err = np.mean(m['shift_err'])
                epoch_success_rate = total_hits / total_samples
                
                history['test_loss'].append(np.mean(m['test_loss']))
                history['test_acc'].append(epoch_acc)
                history['test_iou'].append(epoch_iou)
                history['shift_err'].append(epoch_shift_err)
                history['success_rate'].append(epoch_success_rate)
                
                log_string(f'Test Loss: {history["test_loss"][-1]:.4f}')
                log_string(f'mIoU: {epoch_iou*100:.2f}% | Shift Err: {epoch_shift_err:.4f} | Success Rate: {epoch_success_rate*100:.2f}% | Acc: {epoch_acc*100:.2f}%')
                
                if (epoch + 1) % args.vis_freq == 0:
                    plot_error_distribution(exp_dir, all_sample_errors, epoch + 1, tolerance=args.tolerance)

                plot_performance(exp_dir, history)
                df_history = pd.DataFrame(history)
                df_history.index = df_history.index + 1  # 讓 index 從 1 開始，代表 Epoch
                df_history.index.name = 'Epoch'
                df_history.to_csv(os.path.join(exp_dir, 'training_metrics_history.csv'))
                # Save Best Model based on mIoU
                if epoch_iou > best_iou:
                    best_iou = epoch_iou
                    best_epoch = epoch + 1
                    patience_counter = 0  
                    torch.save({
                        'model_state_dict': classifier.state_dict(), 
                        'iou': float(best_iou), 'accuracy': float(epoch_acc)
                    }, str(checkpoints_dir) + '/best_segmentation_model.pth')
                    log_string(f'Saving Model with Best mIoU: {best_iou*100:.2f}%')
                else:
                    patience_counter += 1
                    if patience_counter >= args.patience:
                        # ===== 這裡加入了開關判斷 =====
                        if args.use_smart_rewind:
                            if overfit_protection_count < max_protections:
                                log_string('!!!! DETECTED STAGNATION. Smart Rewind !!!!')
                                checkpoint = torch.load(str(checkpoints_dir) + '/best_segmentation_model.pth', weights_only=False)
                                classifier.load_state_dict(checkpoint['model_state_dict'])
                                best_iou = checkpoint['iou']
                                
                                # 保留動態調整的程式碼 (因為有開啟開關)
                                current_weight_decay = min(current_weight_decay * 2.0, 0.1)
                                for pg in optimizer.param_groups: pg['weight_decay'] = current_weight_decay
                                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch - epoch, eta_min=1e-6)
                                patience_counter = 0; overfit_protection_count += 1
                            else:
                                log_string('!!!! Stopping training early (Max Protections Reached). !!!!'); break
                        else:
                            # 沒開啟 Smart Rewind，直接觸發普通的提早結束
                            log_string(f'!!!! 模型已連續 {args.patience} 個 Epoch 無進步，觸發 Early Stopping 提早結束訓練 !!!!')
                            break
                        # ==============================
            scheduler.step()

    except KeyboardInterrupt:
        log_string('Training interrupted by user.')

    # ====================================================================================
    # [最後階段] 載入最佳模型，並針對 Train 與 Test 分別進行推論與存檔
    # ====================================================================================
    log_string('')
    log_string('=======================================================')
    log_string(f' Saving Final Results (Best IoU: {best_iou*100:.2f}% at Epoch {best_epoch}) ')
    log_string('=======================================================')

    out_data_dir = exp_dir.joinpath('out_data/')
    out_data_dir.mkdir(exist_ok=True)
    
    best_model_path = str(checkpoints_dir) + '/best_segmentation_model.pth'
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, weights_only=False)
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string(f'Loaded Best Model from checkpoint.')

    # 1. 處理 Test Set 
    log_string('--- Processing Test Set (Clean) ---')
    TEST_DATASET_CLEAN = PartNormalDataset(
        root=args.data_dir, 
        npoints=args.npoint, 
        split='test', 
        class_choice=None, 
        normal_channel=args.normal, 
        augment=False,   # 確保關閉 Augment
        grid_num=args.grid_num
    )
    test_results = save_dataset_inference(classifier, TEST_DATASET_CLEAN, args.batch_size, 'test', out_data_dir, log_string)
    
    # 2. 處理 Train Set
    log_string('--- Processing Train Set (Clean) ---')
    TRAIN_DATASET_CLEAN = PartNormalDataset(
        root=args.data_dir, 
        npoints=args.npoint, 
        split='trainval', 
        class_choice=None, 
        normal_channel=args.normal, 
        augment=False,    # [重點] 強制關閉資料增強
        grid_num=args.grid_num
    )
    train_results = save_dataset_inference(classifier, TRAIN_DATASET_CLEAN, args.batch_size, 'train', out_data_dir, log_string)

    # 3. 彙整 CSV 報表
    all_results = test_results + train_results
    if len(all_results) > 0:
        df_result = pd.DataFrame(all_results)
        output_csv_path = os.path.join(log_dir, 'final_segmentation_report.csv')
        df_result.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
        
        log_string(f'Saved Detailed Report to: {output_csv_path}')
        log_string(f'Saved All Files to: {out_data_dir}')

if __name__ == '__main__':
    args = parse_args()
    main(args)