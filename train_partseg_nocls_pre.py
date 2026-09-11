"""
Author: Benny (Modified for Casting PartSeg NOCLS with Point-MAE Pretrained Support)
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
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from data_utils.ShapeNetDataLoader import PartNormalDataset

plt.switch_backend('agg') 

# 定義預訓練權重路徑
PRETRAINED_PATH = 'point_mae_part_seg.pth'

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
    parser.add_argument('--model', type=str, default='point_mae_partseg', help='model name')
    parser.add_argument('--batch_size', type=int, default=16, help='batch Size during training')
    parser.add_argument('--epoch', default=501, type=int, help='long run for fine-tuning')
    parser.add_argument('--learning_rate', default=0.0002, type=float, help='lower LR for fine-tuning')
    parser.add_argument('--gpu', type=str, default='0', help='specify GPU devices')
    parser.add_argument('--optimizer', type=str, default='AdamW', help='AdamW is better for Transformers')
    parser.add_argument('--log_dir', type=str, default=None, help='log path')
    parser.add_argument('--decay_rate', type=float, default=0.05, help='weight decay for transformer')
    parser.add_argument('--npoint', type=int, default=4096, help='high density for small gates')
    parser.add_argument('--normal', action='store_true', default=False, help='use normals')
    parser.add_argument('--data_dir', type=str, required=True, help='data directory')
    return parser.parse_args()

def load_pretrained_weights(model, ckpt_path, logger):
    """ 關鍵新增：載入 Point-MAE 權重並自動過濾層名 """
    if os.path.exists(ckpt_path):
        logger.info(f'==> Loading Pretrained Weights from {ckpt_path}...')
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        
        # Point-MAE 的權重通常在 'base_model' 鍵值下
        if 'base_model' in checkpoint:
            state_dict = checkpoint['base_model']
        else:
            state_dict = checkpoint

        # 自動適配層名：移除 'MAE_encoder.' 或 'module.' 等前綴
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace("module.", "")
            if name.startswith('MAE_encoder.'):
                name = name[len('MAE_encoder.'):]
            new_state_dict[name] = v

        # 載入權重，設定 strict=False 忽略不匹配的分類頭
        msg = model.load_state_dict(new_state_dict, strict=False)
        logger.info(f'Missing keys: {len(msg.missing_keys)} (Expected for Head), Unexpected keys: {len(msg.unexpected_keys)}')
        logger.info('==> Successfully Loaded Point-MAE Knowledge!')
    else:
        logger.warning(f'==> Pretrained Weights NOT FOUND at {ckpt_path}. Training from Scratch!')

def plot_performance(exp_dir, history):
    epochs = range(1, len(history['train_loss']) + 1)
    
    # 圖表 1: Loss 曲線
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['train_loss'], 'r-', linewidth=1.5, label='Train Loss')
    if len(history['test_loss']) > 0:
        plt.plot(epochs, history['test_loss'], 'b-', linewidth=1.5, label='Test Loss')
    plt.title('Loss Convergence')
    plt.legend(); plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(os.path.join(exp_dir, 'loss_curves.png'))
    plt.close()

    # 圖表 2: Metrics 曲線 (強化 Gate IoU 可見度)
    plt.figure(figsize=(12, 7))
    plt.axhline(y=0.9, color='gray', linestyle=':', alpha=0.5, label='90% Threshold')
    plt.plot(epochs, history['test_acc'], 'g-', alpha=0.4, label='Accuracy')
    plt.plot(epochs, history['test_iou'], 'b-', alpha=0.4, label='mIoU')
    plt.plot(epochs, history['test_body_iou'], 'k:', linewidth=1.2, label='Body IoU')
    
    if 'test_gate_iou' in history:
        # 使用洋紅色加粗虛線，置於最頂層 (zorder=10)
        plt.plot(epochs, history['test_gate_iou'], 'm--', linewidth=2.5, label='GATE (1) IoU', zorder=10)
        
    plt.title('Performance Metrics (Point-MAE Fine-tuning)')
    plt.ylim(-0.05, 1.05); plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True, linestyle='--', alpha=0.5); plt.tight_layout()
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
    file_handler = logging.FileHandler(f'{log_dir}/{args.model}.txt')
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(file_handler)

    # 數據載入 (4096 點 + Z軸 360度旋轉增強)
    TRAIN_DATASET = PartNormalDataset(root=args.data_dir, npoints=args.npoint, split='trainval', augment=True)
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    TEST_DATASET = PartNormalDataset(root=args.data_dir, npoints=args.npoint, split='test', augment=False)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    num_part = 2 

    # 載入模型定義
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models'))
    MODEL = importlib.import_module(args.model)
    classifier = MODEL.get_model(num_part, normal_channel=args.normal).cuda()
    
    # 關鍵：載入 Point-MAE 權重
    load_pretrained_weights(classifier, PRETRAINED_PATH, logger)

    # 損失函數：Label Smoothing 減緩小樣本過擬合
    weights = torch.Tensor([1.0, 40.0]).cuda() # 增加對澆口的補償
    criterion = torch.nn.NLLLoss(weight=weights).cuda()

    optimizer = torch.optim.AdamW(classifier.parameters(), lr=args.learning_rate, weight_decay=args.decay_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch, eta_min=1e-6)

    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_iou': [], 'test_acc': [], 'test_body_iou': [], 'test_gate_iou': []}
    best_gate_iou = 0

    for epoch in range(args.epoch):
        log_string(f'**** Epoch {epoch + 1} ****')
        classifier.train()
        train_loss_epoch, train_correct_epoch = [], []
        
        for i, (points, label, target) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
            optimizer.zero_grad()
            points, target = points.float().cuda(), target.long().cuda()
            points = points.transpose(2, 1) # B, 3, N
            
            seg_pred, _ = classifier(points)
            loss = criterion(seg_pred.contiguous().view(-1, num_part), target.view(-1))
            loss.backward()
            optimizer.step()
            
            pred_choice = seg_pred.data.max(2)[1]
            train_correct_epoch.append(pred_choice.eq(target.data).cpu().sum().item() / (args.batch_size * args.npoint))
            train_loss_epoch.append(loss.item())

        history['train_loss'].append(np.mean(train_loss_epoch))
        history['train_acc'].append(np.mean(train_correct_epoch))

        # 評估階段
        with torch.no_grad():
            classifier.eval()
            test_loss_epoch, total_correct, total_seen = [], 0, 0
            part_ious_total = {0: [], 1: []}
            shape_ious_list = []

            for batch_id, (points, label, target) in enumerate(testDataLoader):
                cur_batch_size, NUM_POINT, _ = points.size()
                points, target = points.float().cuda(), target.long().cuda()
                points = points.transpose(2, 1)
                
                seg_pred, _ = classifier(points)
                test_loss_epoch.append(criterion(seg_pred.view(-1, num_part), target.view(-1)).item())
                
                cur_pred_val = np.argmax(seg_pred.cpu().data.numpy(), 2)
                target_np = target.cpu().data.numpy()
                total_correct += np.sum(cur_pred_val == target_np); total_seen += (cur_batch_size * NUM_POINT)

                for b in range(cur_batch_size):
                    instance_ious = []
                    for l in [0, 1]:
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

            log_string(f'Test Acc: {history["test_acc"][-1]:.4f}, Body IoU: {history["test_body_iou"][-1]:.4f}, GATE IoU: {history["test_gate_iou"][-1]:.4f}')
            plot_performance(exp_dir, history)

            if history['test_gate_iou'][-1] >= best_gate_iou:
                best_gate_iou = history['test_gate_iou'][-1]
                torch.save({'model_state_dict': classifier.state_dict(), 'gate_iou': best_gate_iou}, str(checkpoints_dir) + '/best_gate_model.pth')
        
        scheduler.step()

if __name__ == '__main__':
    args = parse_args()
    main(args)