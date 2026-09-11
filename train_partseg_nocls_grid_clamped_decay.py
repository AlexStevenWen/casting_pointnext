"""
Author: Benny (Distance Regression / Heatmap Edition + PROBE)
Date: 2025 Revised for Regression Task
Description: 
    - 配合 Single Channel Sigmoid Output 模型
    - [新增] 探針監控：檢查模型預測的最大值，防止全零陷阱
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
from data_utils.ShapeNetDataLoader_grid_clamped_decay import PartNormalDataset

plt.switch_backend('agg') 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

def get_regression_stats(preds, targets):
    """ 計算回歸指標: MAE, MSE """
    mae = np.mean(np.abs(preds - targets))
    mse = np.mean((preds - targets) ** 2)
    return mae, mse

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
    
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history['train_loss'], 'r-', label='Train Loss')
    plt.plot(epochs, history['test_loss'], 'b--', label='Test Loss')
    plt.title('Loss Convergence'); plt.legend(); plt.grid(True); plt.savefig(os.path.join(exp_dir, 'loss.png')); plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history['test_mae'], 'b-', label='Test MAE (L1 Error)')
    plt.plot(epochs, history['test_mse'], 'g--', label='Test MSE (L2 Error)')
    plt.title('Regression Error (Lower is Better)'); plt.legend(); plt.grid(True); plt.savefig(os.path.join(exp_dir, 'regression_error.png')); plt.close()

def main(args):
    def log_string(str):
        logger.info(str); print(str)

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

    history = {'train_loss': [], 'test_loss': [], 'test_mae': [], 'test_mse': [], 'grid_acc': []}
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
            
            # 使用 Sigmoid output
            pred_intensity = seg_pred
            
            # Loss Calculation
            loss_mse = mse_criterion(pred_intensity, target)
            loss_grid = grid_criterion(grid_logits, grid_gt)
            
            loss = 50.0 * loss_mse + 2.0 * loss_grid
            
            loss.backward()
            optimizer.step()
            train_loss_epoch.append(loss.item())
            
        history['train_loss'].append(np.mean(train_loss_epoch))

        with torch.no_grad():
            classifier.eval()
            m = {'test_loss': [], 'mae': [], 'mse': [], 'grid_acc': []}

            for batch_id, (points, label, target, grid_gt) in enumerate(testDataLoader):
                cur_batch_size = points.size(0)
                points_cuda, target_cuda, grid_gt_cuda = points.float().cuda(), target.float().cuda(), grid_gt.float().cuda()
                
                seg_pred, grid_logits = classifier(points_cuda.transpose(2, 1))
                pred_intensity = seg_pred
                
                l_mse = mse_criterion(pred_intensity, target_cuda)
                l_grid = grid_criterion(grid_logits, grid_gt_cuda)
                m['test_loss'].append((50.0 * l_mse + 2.0 * l_grid).item())
                
                # Metrics
                mae, mse = get_regression_stats(pred_intensity.cpu().numpy(), target.numpy())
                m['mae'].append(mae)
                m['mse'].append(mse)
                
                # Grid Acc
                gr_probs = torch.sigmoid(grid_logits)
                gr_pred_binary = (gr_probs > 0.5).float()
                grid_acc = (gr_pred_binary == grid_gt_cuda).float().mean().item()
                m['grid_acc'].append(grid_acc)

                # =======================================================
                # [探針代碼] 監控第 0 個 Batch 的輸出分佈
                # =======================================================
                if batch_id == 0:
                    # 1. 取得模型預測的最大值 (看它有沒有勇氣預測 > 0 的數)
                    batch_pred_max = pred_intensity.max().item()
                    # 2. 取得真實標籤的最大值 (確認這個 Batch 裡真的有澆口)
                    batch_gt_max = target.max().item()
                    # 3. 取得澆口區域的平均預測值 (模型在目標位置預測了多少?)
                    #    mask = target > 0.1 (只看真實澆口區域)
                    mask = (target > 0.1)
                    if mask.sum() > 0:
                        avg_pred_on_gate = pred_intensity[mask].mean().item()
                    else:
                        avg_pred_on_gate = 0.0
                    
                    log_string(f'---- PROBE [Epoch {epoch}] ----')
                    log_string(f'GT Max Val   : {batch_gt_max:.4f}')
                    log_string(f'Pred Max Val : {batch_pred_max:.4f} (若 < 0.01 代表模型在偷懶)')
                    log_string(f'Gate Avg Pred: {avg_pred_on_gate:.4f} (澆口區域的平均預測強度)')
                    log_string(f'-------------------------------')

            epoch_mae = np.mean(m['mae'])
            epoch_mse = np.mean(m['mse'])
            epoch_grid_acc = np.mean(m['grid_acc'])
            
            history['test_loss'].append(np.mean(m['test_loss']))
            history['test_mae'].append(epoch_mae)
            history['test_mse'].append(epoch_mse)
            history['grid_acc'].append(epoch_grid_acc)

            log_string(f'Test Total Loss: {history["test_loss"][-1]:.4f}')
            log_string(f'Regression Error -> MAE: {epoch_mae:.5f} | MSE: {epoch_mse:.5f}')
            log_string(f'Grid Branch      -> Accuracy: {epoch_grid_acc:.4f}')
            
            plot_performance(exp_dir, history)
            
            if epoch_mae <= best_mae:
                best_mae = epoch_mae
                torch.save({'model_state_dict': classifier.state_dict(), 'mae': best_mae}, str(checkpoints_dir) + '/best_regression_model.pth')
                log_string(f'Saving Model with Lowest MAE: {best_mae:.5f}')
        
        scheduler.step()

if __name__ == '__main__':
    args = parse_args()
    main(args)