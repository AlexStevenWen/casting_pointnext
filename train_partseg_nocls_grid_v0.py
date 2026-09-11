"""
Author: Benny (Modified for Casting PartSeg NOCLS with Full Diagnostics & Grid-Probability)
Date: 2025 Revised
"""
import argparse
import os
import torch
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
# 確保使用的是帶有 Grid 的數據加載器
from data_utils.ShapeNetDataLoader_grid import PartNormalDataset

plt.switch_backend('agg') 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True

def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='pointnext_part_seg_nocls_grid', help='model name')
    parser.add_argument('--batch_size', type=int, default=16, help='batch Size during training')
    parser.add_argument('--epoch', default=251, type=int, help='epoch to run')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--gpu', type=str, default='0', help='specify GPU devices')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD')
    parser.add_argument('--log_dir', type=str, default=None, help='log path')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--npoint', type=int, default=4096, help='point Number')
    parser.add_argument('--normal', action='store_true', default=False, help='use normals')
    parser.add_argument('--data_dir', type=str, required=True, help='data directory')
    parser.add_argument('--grid_num', type=int, default=3, help='n for nxnxn grid')
    return parser.parse_args()

def compute_grid_metrics(pred_logits, target_labels, threshold=0.5):
    """ 計算格子準確率與格子 IoU """
    preds = (torch.sigmoid(pred_logits) > threshold).float()
    correct = (preds == target_labels).float()
    acc = correct.mean().item()
    
    intersection = (preds * target_labels).sum().item()
    union = ((preds + target_labels) > 0).float().sum().item()
    grid_iou = intersection / union if union > 0 else 1.0
    return acc, grid_iou

def plot_performance(exp_dir, history):
    epochs = range(1, len(history['train_loss']) + 1)
    
    # 圖表 1: Loss 曲線
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['train_loss'], 'r-', label='Train Total Loss')
    if 'test_loss' in history and len(history['test_loss']) > 0:
        plt.plot(epochs, history['test_loss'], 'b--', label='Test Total Loss')
    plt.title('Loss Convergence')
    plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(exp_dir, 'loss_curves.png'))
    plt.close()

    # 圖表 2: Metrics 曲線 (視覺化診斷)
    plt.figure(figsize=(12, 7))
    plt.axhline(y=0.9, color='gray', linestyle=':', alpha=0.5, label='90% Threshold')
    plt.plot(epochs, history['test_acc'], 'g-', alpha=0.4, label='Accuracy (整體)')
    plt.plot(epochs, history['test_iou'], 'b-', alpha=0.4, label='mIoU (平均)')
    plt.plot(epochs, history['test_body_iou'], 'k:', alpha=0.6, label='Body IoU')
    
    if 'test_grid_iou' in history:
        plt.plot(epochs, history['test_grid_iou'], 'c-', label='Grid IoU (定位)')
    if 'test_gate_iou' in history:
        # 加粗粉紅線代表最重要的澆口指標
        plt.plot(epochs, history['test_gate_iou'], 'm--', linewidth=2.5, label='CRITICAL: Gate IoU')
        
    plt.title('Performance Metrics with Grid Supervision')
    plt.ylim(-0.05, 1.05); plt.legend(loc='upper left', bbox_to_anchor=(1, 1)); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, 'metrics_curves.png'))
    plt.close('all')

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

    TRAIN_DATASET = PartNormalDataset(root=args.data_dir, npoints=args.npoint, split='trainval', grid_num=args.grid_num, augment=True)
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    TEST_DATASET = PartNormalDataset(root=args.data_dir, npoints=args.npoint, split='test', grid_num=args.grid_num, augment=False)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    num_part = 2 
    MODEL = importlib.import_module(args.model)
    classifier = MODEL.get_model(num_part, normal_channel=args.normal, grid_num=args.grid_num).cuda()
    classifier.apply(inplace_relu)
    
    # 策略：增加格子權重到 2.0，降低澆口權重到 20.0 以維持穩定性
    seg_weights = torch.Tensor([1.0, 10.0]).cuda() 
    seg_criterion = torch.nn.NLLLoss(weight=seg_weights).cuda()
    pos_weight = torch.tensor([20.0]).cuda() # 給予有標籤格子 5 倍權重
    grid_criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight).cuda()

    optimizer = torch.optim.Adam(classifier.parameters(), lr=args.learning_rate, weight_decay=args.decay_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch, eta_min=1e-5)

    history = {'train_loss': [], 'grid_loss': [], 'test_loss': [], 'test_iou': [], 
               'test_acc': [], 'test_body_iou': [], 'test_gate_iou': [], 'test_grid_acc': [], 'test_grid_iou': []}
    best_gate_iou = 0

    for epoch in range(args.epoch):
        log_string(f'**** Epoch {epoch + 1} ****')
        classifier.train()
        train_loss_epoch, grid_loss_epoch = [], []
        
        for i, (points, label, target, grid_gt) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
            optimizer.zero_grad()
            points, target, grid_gt = points.float().cuda(), target.long().cuda(), grid_gt.float().cuda()
            points = points.transpose(2, 1) 
            seg_pred, grid_pred = classifier(points)
            
            loss_seg = seg_criterion(seg_pred.contiguous().view(-1, num_part), target.view(-1))
            loss_grid = grid_criterion(grid_pred, grid_gt)
            
            total_loss = loss_seg + 2.0 * loss_grid
            total_loss.backward()
            optimizer.step()
            
            train_loss_epoch.append(total_loss.item())
            grid_loss_epoch.append(loss_grid.item())

        history['train_loss'].append(np.mean(train_loss_epoch))
        history['grid_loss'].append(np.mean(grid_loss_epoch))

        with torch.no_grad():
            classifier.eval()
            test_loss_list, test_grid_loss_list = [], []
            total_correct, total_seen = 0, 0
            part_ious_total = {0: [], 1: []}
            shape_ious_list = []
            grid_acc_list, grid_iou_list = [] , []

            for batch_id, (points, label, target, grid_gt) in enumerate(testDataLoader):
                cur_batch_size, NUM_POINT, _ = points.size()
                points, target, grid_gt = points.float().cuda(), target.long().cuda(), grid_gt.float().cuda()
                points = points.transpose(2, 1)
                
                seg_pred, grid_pred = classifier(points)
                
                l_seg = seg_criterion(seg_pred.view(-1, num_part), target.view(-1))
                l_grid = grid_criterion(grid_pred, grid_gt)
                test_loss_list.append((l_seg + l_grid).item())
                test_grid_loss_list.append(l_grid.item())

                # 格子指標
                g_acc, g_iou = compute_grid_metrics(grid_pred, grid_gt)
                grid_acc_list.append(g_acc)
                grid_iou_list.append(g_iou)

                # 點預測指標
                cur_pred_val = np.argmax(seg_pred.cpu().data.numpy(), 2)
                target_np = target.cpu().data.numpy()
                total_correct += np.sum(cur_pred_val == target_np)
                total_seen += (cur_batch_size * NUM_POINT)

                for b in range(cur_batch_size):
                    instance_ious = []
                    for l in [0, 1]:
                        I = np.sum((target_np[b] == l) & (cur_pred_val[b] == l))
                        U = np.sum((target_np[b] == l) | (cur_pred_val[b] == l))
                        iou = I / float(U) if U != 0 else 1.0
                        part_ious_total[l].append(iou)
                        instance_ious.append(iou)
                    shape_ious_list.append(np.mean(instance_ious))

            history['test_loss'].append(np.mean(test_loss_list))
            history['test_iou'].append(np.mean(shape_ious_list))
            history['test_acc'].append(total_correct / float(total_seen))
            history['test_body_iou'].append(np.mean(part_ious_total[0]))
            history['test_gate_iou'].append(np.mean(part_ious_total[1]))
            history['test_grid_acc'].append(np.mean(grid_acc_list))
            history['test_grid_iou'].append(np.mean(grid_iou_list))

            # 補回所有指標輸出
            log_string(f'Test Loss: {history["test_loss"][-1]:.4f}, Accuracy: {history["test_acc"][-1]:.4f}, mIoU: {history["test_iou"][-1]:.4f}')
            log_string(f'Diagnostic -> Body IoU: {history["test_body_iou"][-1]:.4f}, Gate IoU: {history["test_gate_iou"][-1]:.4f}')
            log_string(f'Grid Branch -> Acc: {history["test_grid_acc"][-1]:.4f}, IoU: {history["test_grid_iou"][-1]:.4f}')
            
            plot_performance(exp_dir, history)

            if history['test_gate_iou'][-1] >= best_gate_iou:
                best_gate_iou = history['test_gate_iou'][-1]
                torch.save({'model_state_dict': classifier.state_dict(), 'gate_iou': best_gate_iou}, str(checkpoints_dir) + '/best_gate_model.pth')
        
        scheduler.step()

if __name__ == '__main__':
    args = parse_args()
    main(args)