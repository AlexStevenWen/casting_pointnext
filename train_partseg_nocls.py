"""
Author: Benny (Modified for Casting PartSeg NOCLS with Part-wise Diagnostics)
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
import provider
import numpy as np
import json
import matplotlib.pyplot as plt
plt.switch_backend('agg') 

from pathlib import Path
from tqdm import tqdm
from data_utils.ShapeNetDataLoader import PartNormalDataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True

def to_categorical(y, num_classes):
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y

def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='pointnet_part_seg', help='model name')
    parser.add_argument('--batch_size', type=int, default=16, help='batch Size during training')
    parser.add_argument('--epoch', default=251, type=int, help='epoch to run')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--gpu', type=str, default='0', help='specify GPU devices')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD')
    parser.add_argument('--log_dir', type=str, default=None, help='log path')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--npoint', type=int, default=2048, help='point Number')
    parser.add_argument('--normal', action='store_true', default=False, help='use normals')
    parser.add_argument('--step_size', type=int, default=20, help='decay step for lr decay')
    parser.add_argument('--lr_decay', type=float, default=0.5, help='decay rate for lr decay')
    parser.add_argument('--data_dir', type=str, required=True, help='data directory')
    return parser.parse_args()

def plot_performance(exp_dir, history):
    epochs = range(1, len(history['train_loss']) + 1)
    
    # 圖表 1: Loss 曲線
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['train_loss'], 'r-', linewidth=1.5, label='Train Loss')
    if len(history['test_loss']) > 0:
        plt.plot(epochs, history['test_loss'], 'b-', linewidth=1.5, label='Test Loss')
    plt.title('Loss Convergence')
    plt.xlabel('Epochs')
    plt.ylabel('Loss Value')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(os.path.join(exp_dir, 'loss_curves.png'))
    plt.close()

    # 圖表 2: Metrics 曲線 (優化層次感)
    plt.figure(figsize=(12, 7))
    # 先畫背景參考線
    plt.axhline(y=0.9, color='gray', linestyle=':', alpha=0.5, label='90% Threshold')
    
    # 繪製指標
    plt.plot(epochs, history['test_acc'], 'g-', linewidth=1.5, alpha=0.6, label='Test Accuracy')
    plt.plot(epochs, history['test_iou'], 'b-', linewidth=1.5, alpha=0.6, label='Test mIoU')
    plt.plot(epochs, history['test_body_iou'], 'k:', linewidth=1.2, label='Body (0) IoU')
    
    # 將最重要的 Gate IoU 放在最後畫，並加粗，確保不被擋住
    if 'test_gate_iou' in history and len(history['test_gate_iou']) > 0:
        plt.plot(epochs, history['test_gate_iou'], 'm--', linewidth=2.5, label='CRITICAL: Gate (1) IoU')
        
    plt.title('Performance Metrics (Gate-Focused Visualization)')
    plt.xlabel('Epochs')
    plt.ylabel('Score (0.0 - 1.0)')
    plt.ylim(-0.05, 1.05)
    # 圖例放在外面避免遮擋數據
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, 'metrics_curves.png'))
    plt.close('all')
def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    folder_name = f"{args.model if args.log_dir is None else args.log_dir}_{timestr}"
    exp_dir = Path('./log/part_seg/').joinpath(folder_name)
    exp_dir.mkdir(exist_ok=True, parents=True)
    checkpoints_dir, log_dir = exp_dir.joinpath('checkpoints/'), exp_dir.joinpath('logs/')
    checkpoints_dir.mkdir(exist_ok=True); log_dir.mkdir(exist_ok=True)

    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler = logging.FileHandler(f'{log_dir}/{args.model}.txt')
    file_handler.setFormatter(formatter); logger.addHandler(file_handler)

    root = args.data_dir
    TRAIN_DATASET = PartNormalDataset(root=root, npoints=args.npoint, split='trainval', 
                                  normal_channel=args.normal, augment=True)
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    TEST_DATASET = PartNormalDataset(root=root, npoints=args.npoint, split='test', normal_channel=args.normal, augment=False)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    seg_classes = TRAIN_DATASET.seg_classes
    seg_label_to_cat = {label: cat for cat, labels in seg_classes.items() for label in labels}
    num_classes, num_part = len(TRAIN_DATASET.classes), 2 # 強制定義為 2 類

    MODEL = importlib.import_module(args.model)
    shutil.copy('models/%s.py' % args.model, str(exp_dir))
    classifier = MODEL.get_model(num_part, normal_channel=args.normal).cuda()
    
    # 使用 30 倍權重對抗散落澆口
    weights = torch.Tensor([1.0, 50.0]).cuda() 
    base_criterion = torch.nn.NLLLoss(weight=weights).cuda()
    def criterion(pred, target, trans_feat): return base_criterion(pred, target)

    classifier.apply(inplace_relu)
    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_iou': [], 'test_acc': [], 'test_body_iou': [], 'test_gate_iou': []}
    best_instance_avg_iou, start_epoch = 0, 0

    optimizer = torch.optim.Adam(classifier.parameters(), lr=args.learning_rate, weight_decay=args.decay_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch, eta_min=1e-5)

    for epoch in range(start_epoch, args.epoch):
        log_string(f'**** Epoch {epoch + 1} ****')
        log_string(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        classifier.train()
        train_loss_epoch, train_correct_epoch = [], []
        for i, (points, label, target) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
            optimizer.zero_grad()
            points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()
            points = points.transpose(2, 1)
            seg_pred, trans_feat = classifier(points, to_categorical(label, num_classes))
            pred_choice = seg_pred.data.max(2)[1]
            train_correct_epoch.append(pred_choice.eq(target.data).cpu().sum().item() / (args.batch_size * args.npoint))
            loss = criterion(seg_pred.contiguous().view(-1, num_part), target.view(-1), trans_feat)
            loss.backward(); optimizer.step(); train_loss_epoch.append(loss.item())

        history['train_loss'].append(np.mean(train_loss_epoch))
        history['train_acc'].append(np.mean(train_correct_epoch))

        with torch.no_grad():
            classifier.eval()
            test_loss_epoch, total_correct, total_seen = [], 0, 0
            # 建立部件 IoU 統計
            part_ious_total = {0: [], 1: []}
            shape_ious_list = []

            for batch_id, (points, label, target) in enumerate(testDataLoader):
                cur_batch_size, NUM_POINT, _ = points.size()
                points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()
                points = points.transpose(2, 1)
                seg_pred, trans_feat = classifier(points, to_categorical(label, num_classes))
                test_loss_epoch.append(criterion(seg_pred.view(-1, num_part), target.view(-1), trans_feat).item())
                
                cur_pred_val = np.argmax(seg_pred.cpu().data.numpy(), 2)
                target_np = target.cpu().data.numpy()
                total_correct += np.sum(cur_pred_val == target_np); total_seen += (cur_batch_size * NUM_POINT)

                for b in range(cur_batch_size):
                    instance_ious = []
                    for l in [0, 1]: # Body and Gate
                        I = np.sum((target_np[b] == l) & (cur_pred_val[b] == l))
                        U = np.sum((target_np[b] == l) | (cur_pred_val[b] == l))
                        iou = I / float(U) if U != 0 else 1.0
                        part_ious_total[l].append(iou)
                        instance_ious.append(iou)
                    shape_ious_list.append(np.mean(instance_ious))

            history['test_loss'].append(np.mean(test_loss_epoch))
            history['test_iou'].append(np.mean(shape_ious_list))
            history['test_acc'].append(total_correct / float(total_seen))
            history['test_body_iou'].append(np.mean(part_ious_total[0]))
            history['test_gate_iou'].append(np.mean(part_ious_total[1]))

            log_string(f'Test Acc: {history["test_acc"][-1]:.4f}, mIoU: {history["test_iou"][-1]:.4f}')
            log_string(f'Diagnostic -> Body IoU: {history["test_body_iou"][-1]:.4f}, Gate IoU: {history["test_gate_iou"][-1]:.4f}')
            plot_performance(exp_dir, history)

            if (history['test_iou'][-1] >= best_instance_avg_iou):
                best_instance_avg_iou = history['test_iou'][-1]
                torch.save({'epoch': epoch, 'model_state_dict': classifier.state_dict(), 'instance_avg_iou': best_instance_avg_iou}, str(checkpoints_dir) + '/best_model.pth')
        scheduler.step()

if __name__ == '__main__':
    args = parse_args()
    main(args)