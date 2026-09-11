"""
用於 PVN3D (關鍵點回歸) 的訓練腳本
"""

import os
import sys
import torch
import numpy as np
import datetime
import logging
import provider # (我們仍然需要它進行資料增強)
import importlib
import shutil
import argparse

from pathlib import Path
from tqdm import tqdm
from data_utils.PVN3DDataLoader import PVN3DDataLoader # (*** 新 ***)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
    
    # --- (*** 新 ***) ---
    parser.add_argument('--model', default='PVN3D_model', help='model name [default: PVN3D_model]')
    parser.add_argument('--loss', default='PVN3D_loss', help='loss function name [default: PVN3D_loss]')
    # (我們使用 output_dim 來告訴模型關鍵點的數量)
    parser.add_argument('--output_dim', default=8, type=int, help='Number of keypoints [default: 8]')
    # --- (*** 結束 ***) ---
    
    parser.add_argument('--epoch', default=200, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    
    # (*** 新 ***) 指向 pvn3d_dataset
    parser.add_argument('--data_dir', type=str, default='data/pvn3d_dataset/', help='path to dataset') 
    
    return parser.parse_args()


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True

# --- (*** 新的 TEST 函數 ***) ---
def test(model, loader, criterion, args):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    # 用於儲存所有 *L2 誤差*
    all_keypoint_l2_errors = []

    with torch.no_grad(): 
        for j, (points, target) in tqdm(enumerate(loader), total=len(loader)):
            if not args.use_cpu:
                points, target = points.cuda(), target.cuda()

            target = target.float() # (B, 8, 3)
            points = points.transpose(2, 1) # (B, 3, N)
            
            pred, trans_feat = model(points) # pred: (B, 8, 3)
            
            loss = criterion(pred, target, trans_feat)
            
            total_loss += loss.item()
            num_batches += 1
            
            # --- (新) 計算 L2 誤差 (歐幾里得距離) ---
            # (B, 8, 3) -> (B, 8)
            l2_dist_per_keypoint = torch.norm(pred - target, dim=2)
            
            # (B, 8) -> (B,)
            mean_l2_dist_per_object = torch.mean(l2_dist_per_keypoint, dim=1)
            
            all_keypoint_l2_errors.append(mean_l2_dist_per_object.cpu().numpy())
            # --- (結束) ---

    # 1. 平均 Loss
    mean_loss = total_loss / float(num_batches) if num_batches > 0 else 0.0 

    # 2. 平均 L2 關鍵點誤差 (MAE)
    all_errors = np.concatenate(all_keypoint_l2_errors, axis=0)
    mae_l2_dist = np.mean(all_errors)

    # 3. 準確率 @ 5% 尺度 (Acc @ 0.05)
    # (假設我們的資料被歸一化到 1.0 的尺度)
    threshold = 0.05
    acc_at_5_percent = np.mean(all_errors <= threshold)

    return mean_loss, mae_l2_dist, acc_at_5_percent
# --- (*** TEST 函數結束 ***) ---


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('pvn3d') # (新日誌資料夾)
    exp_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args() 
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    data_path = args.data_dir

    # (*** 新 ***)
    train_dataset = PVN3DDataLoader(root=data_path, args=args, split='train')
    test_dataset = PVN3DDataLoader(root=data_path, args=args, split='test')
    # (*** 結束 ***)

    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=10, drop_last=True)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10)

    '''MODEL LOADING'''
    num_keypoints = args.output_dim
    model = importlib.import_module(args.model)
    loss_fn = importlib.import_module(args.loss) # (新)
    
    shutil.copy('./models/%s.py' % args.model, str(exp_dir))
    shutil.copy('./models/%s.py' % args.loss, str(exp_dir))
    shutil.copy('./train_pvn3d.py', str(exp_dir)) # (新)
    shutil.copy('./models/pointnet2_reg_utils.py', str(exp_dir)) # (新)

    classifier = model.get_model(num_keypoints, normal_channel=args.use_normals)
    criterion = loss_fn.get_loss() # (新)
    classifier.apply(inplace_relu)

    if not args.use_cpu:
        classifier = classifier.cuda()
        criterion = criterion.cuda()

    try:
        checkpoint = torch.load(str(exp_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    global_epoch = 0
    global_step = 0
    
    # (*** 新 ***)
    best_loss = float('inf') # (我們現在用 Mean L2 Error)
    best_epoch = 0 

    '''TRANING'''
    logger.info('Start training...')
    for epoch in range(start_epoch, args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        
        train_loss_list = [] 
        classifier = classifier.train()

        scheduler.step()
 
        for batch_id, (points, target) in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
            optimizer.zero_grad()

            # (*** 新 ***) 點雲抖動和增強
            # (我們只抖動點雲，不抖動關鍵點標籤)
            points = points.data.numpy()
            points = provider.random_point_dropout(points)
            points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            points = torch.Tensor(points)
            # (*** 結束 ***)
            
            points = points.transpose(2, 1)

            if not args.use_cpu:
                points, target = points.cuda(), target.cuda()

            # (target shape 預期是 B, 8, 3)
            # (points shape 預期是 B, 3, N)
            
            pred, trans_feat = classifier(points) # pred: (B, 8, 3)
            
            loss = criterion(pred, target.float(), trans_feat) 
            
            train_loss_list.append(loss.item()) 
            
            loss.backward()
            optimizer.step()
            global_step += 1

        train_mean_loss = np.mean(train_loss_list) 
        log_string('Train Mean Loss (L1): %f' % train_mean_loss) 

        # --- (*** 新的驗證迴圈 ***) ---
        with torch.no_grad():
            val_loss, val_mae_l2_dist, val_acc_5p = test(classifier.eval(), testDataLoader, criterion, args)

            is_best_model = val_mae_l2_dist < best_loss 
            if is_best_model:
                best_loss = val_mae_l2_dist # 更新最佳 L2 誤差
                best_epoch = epoch + 1
            
            log_string('Validation Loss (L1): %f' % val_loss)
            log_string('Validation MAE (L2 Norm): %.6f' % val_mae_l2_dist)
            log_string('Accuracy (Keypoints < 5%% scale): %.4f' % val_acc_5p)
            log_string('Best MAE (L2 Norm): %.6f (at epoch %d)' % (best_loss, best_epoch))

            if (is_best_model):
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': best_epoch,
                    'val_loss': val_loss, 
                    'val_mae_l2': val_mae_l2_dist,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
            global_epoch += 1

    logger.info('End of training...')


if __name__ == '__main__':
    args = parse_args()
    main(args)