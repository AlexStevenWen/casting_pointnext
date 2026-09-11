"""
Author: Benny (Modified for Casting PartSeg NOCLS)
Date: 2025 Revised
"""
import argparse
import os
import torch
import datetime
import logging
from pathlib import Path
import sys
import importlib
import shutil
from tqdm import tqdm
import provider
import numpy as np
import time

# 為了讓此腳本能讀取你目前的 PartNormalDataset
from data_utils.ShapeNetDataLoader import PartNormalDataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True

def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='pointnet_sem_seg_nocls', help='model name')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training')
    parser.add_argument('--epoch', default=251, type=int, help='Epoch to run')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='Initial learning rate')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD')
    parser.add_argument('--log_dir', type=str, default=None, help='Log path')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--npoint', type=int, default=2048, help='Point Number')
    parser.add_argument('--step_size', type=int, default=20, help='Decay step for lr decay')
    parser.add_argument('--lr_decay', type=float, default=0.5, help='Decay rate for lr decay')
    parser.add_argument('--data_dir', type=str, required=True, help='Data directory')
    parser.add_argument('--normal', action='store_true', default=False, help='use normals')

    return parser.parse_args()

def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path('./log/')
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath('sem_seg')
    experiment_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        experiment_dir = experiment_dir.joinpath(timestr)
    else:
        experiment_dir = experiment_dir.joinpath(args.log_dir)
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    root = args.data_dir
    NUM_POINT = args.npoint
    BATCH_SIZE = args.batch_size

    # --- 1. 資料集動態加載 (使用你目前的 PartNormalDataset) ---
    print("start loading training data ...")
    TRAIN_DATASET = PartNormalDataset(root=root, npoints=NUM_POINT, split='trainval', normal_channel=args.normal)
    print("start loading test data ...")
    TEST_DATASET = PartNormalDataset(root=root, npoints=NUM_POINT, split='test', normal_channel=args.normal)

    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=BATCH_SIZE, shuffle=True, num_workers=10, drop_last=True)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=BATCH_SIZE, shuffle=False, num_workers=10)

    # --- 2. 動態獲取類別資訊 ---
    seg_classes = TRAIN_DATASET.seg_classes
    seg_label_to_cat = {}
    for cat in seg_classes.keys():
        for label in seg_classes[cat]:
            seg_label_to_cat[label] = cat
    
    # 部件總數 (例如 2: 本體與澆口)
    all_parts = []
    for cat in seg_classes:
        all_parts.extend(seg_classes[cat])
    NUM_CLASSES = len(set(all_parts)) 

    # 設定損失函數權重 (解決樣本不平衡)
    # 這裡手動設定為 澆口(1) 是 本體(0) 的 10 倍權重
    weights = torch.Tensor([1.0, 10.0]).cuda()

    log_string("The number of training data is: %d" % len(TRAIN_DATASET))
    log_string("The number of test data is: %d" % len(TEST_DATASET))
    log_string(f"Detected Parts: {NUM_CLASSES}")

    '''MODEL LOADING'''
    MODEL = importlib.import_module(args.model)
    shutil.copy('models/%s.py' % args.model, str(experiment_dir))

    # 初始化模型時傳入 NUM_CLASSES (即部件數)
    classifier = MODEL.get_model(NUM_CLASSES, normal_channel=args.normal).cuda()
    criterion = MODEL.get_loss(weight=weights).cuda()
    classifier.apply(inplace_relu)

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv1d') != -1 or classname.find('Linear') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)

    try:
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0
        classifier = classifier.apply(weights_init)

    optimizer = torch.optim.Adam(classifier.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.decay_rate)

    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size

    global_epoch = 0
    best_iou = 0

    for epoch in range(start_epoch, args.epoch):
        log_string('**** Epoch %d (%d/%s) ****' % (global_epoch + 1, epoch + 1, args.epoch))
        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        momentum = max(MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP)), 0.01)
        classifier = classifier.apply(lambda x: bn_momentum_adjust(x, momentum))
        
        num_batches = len(trainDataLoader)
        total_correct = 0
        total_seen = 0
        loss_sum = 0
        classifier = classifier.train()

        for i, (points, label, target) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
            optimizer.zero_grad()
            points, target = points.float().cuda(), target.long().cuda()
            points = points.transpose(2, 1)

            seg_pred, trans_feat = classifier(points)
            seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)
            target = target.view(-1, 1)[:, 0]
            
            # 使用包含權重的損失函數
            loss = criterion(seg_pred, target, trans_feat)
            loss.backward()
            optimizer.step()

            pred_choice = seg_pred.cpu().data.max(1)[1].numpy()
            target_np = target.cpu().data.numpy()
            correct = np.sum(pred_choice == target_np)
            total_correct += correct
            total_seen += (BATCH_SIZE * NUM_POINT)
            loss_sum += loss
            
        log_string('Training mean loss: %f' % (loss_sum / num_batches))
        log_string('Training accuracy: %f' % (total_correct / float(total_seen)))

        '''Evaluation'''
        with torch.no_grad():
            num_batches = len(testDataLoader)
            total_correct = 0
            total_seen = 0
            
            # 傳統寫法：先定義長度，再建立列表
            num_part_count = NUM_CLASSES 
            total_seen_class = [0] * num_part_count
            total_correct_class = [0] * num_part_count
            total_iou_deno_class = [0] * num_part_count
            
            classifier = classifier.eval()
            log_string('---- EPOCH %03d EVALUATION ----' % (global_epoch + 1))
            for i, (points, label, target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
                points, target = points.float().cuda(), target.long().cuda()
                points = points.transpose(2, 1)

                seg_pred, _ = classifier(points)
                pred_val = seg_pred.contiguous().cpu().data.numpy()
                target_np = target.cpu().data.numpy()
                
                pred_val = np.argmax(pred_val, 2)
                correct = np.sum((pred_val == target_np))
                total_correct += correct
                total_seen += (cur_batch_size := points.size(0)) * NUM_POINT

                for l in range(NUM_CLASSES):
                    total_seen_class[l] += np.sum((target_np == l))
                    total_correct_class[l] += np.sum((pred_val == l) & (target_np == l))
                    total_iou_deno_class[l] += np.sum(((pred_val == l) | (target_np == l)))

            mIoU = np.mean(np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=float) + 1e-6))
            log_string('eval point avg class IoU: %f' % (mIoU))
            log_string('eval point accuracy: %f' % (total_correct / float(total_seen)))

            if mIoU >= best_iou:
                best_iou = mIoU
                savepath = str(checkpoints_dir) + '/best_model.pth'
                state = {'epoch': epoch, 'class_avg_iou': mIoU, 'model_state_dict': classifier.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}
                torch.save(state, savepath)
                log_string('Saving model....')
            log_string('Best mIoU: %f' % best_iou)
        global_epoch += 1

if __name__ == '__main__':
    args = parse_args()
    main(args)