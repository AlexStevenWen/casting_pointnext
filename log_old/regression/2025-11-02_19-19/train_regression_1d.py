"""
Author: Benny
Date: Nov 2019
(Modified for Regression)
"""

import os
import sys
import torch
import numpy as np

import datetime
import logging
import provider
import importlib
import shutil
import argparse

from pathlib import Path
from tqdm import tqdm

# --- 變更 1: 匯入正確的 Data Loader ---
# 移除: from data_utils.ModelNetDataLoader import ModelNetDataLoader 
# 替換為:
from data_utils.RegressionDataLoader import RegressionDataLoader
# ------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
    parser.add_argument('--model', default='pointnet_reg_1d', help='model name [default: pointnet_reg_1d]') 
    
    ### 1. 修改參數 (Parameter Change) ###
    parser.add_argument('--output_dim', default=6, type=int, help='regression output dimension') 
    
    parser.add_argument('--epoch', default=200, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    
    parser.add_argument('--data_dir', type=str, default='data/your_regression_data/', help='path to dataset') 
    
    return parser.parse_args()


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True


### 2. 重寫 Test 函數 (Rewrite Test Function) ###
def test(model, loader, criterion, args):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    # (*** 新增 ***) 用於儲存所有誤差值
    all_abs_errors_normalized = []

    with torch.no_grad(): 
        for j, (points, target) in tqdm(enumerate(loader), total=len(loader)):
            if not args.use_cpu:
                points, target = points.cuda(), target.cuda()

            target = target.float() 
            
            points = points.transpose(2, 1)
            pred, trans_feat = model(points)
            
            # (可選) 處理 1D 標籤的維度問題
            if args.output_dim == 1 and len(target.shape) == 1:
                target = target.unsqueeze(1) 
            if args.output_dim == 1 and len(pred.shape) == 1:
                pred = pred.unsqueeze(1) 

            loss = criterion(pred, target, trans_feat)
            
            total_loss += loss.item()
            num_batches += 1
            
            # (*** 新增 ***) 計算標準化的絕對誤差
            # 假設 pred 和 target 都在 [-1, 1] 範圍 (rad/pi)
            abs_error_normalized = torch.abs(pred - target)
            all_abs_errors_normalized.append(abs_error_normalized.cpu().numpy())
    # --- (*** 新增：計算最終指標 ***) ---
    
    # 1. 計算平均 Loss (您原本的指標)
    mean_loss = total_loss / float(num_batches) if num_batches > 0 else 0.0 

    # 2. 計算 MAE (in Degrees) - 平均絕對誤差 (度)
    # 將所有批次的誤差合併
    all_abs_errors = np.concatenate(all_abs_errors_normalized, axis=0)
    # 將標準化誤差 [-1, 1] 轉換回角度
    all_abs_errors_deg = all_abs_errors * 180.0
    mae_deg = np.mean(all_abs_errors_deg)

    # 3. 計算 準確率@5度 (Accuracy @ 5 degrees)
    # 這是指所有預測軸中，有多少比例的誤差小於 5 度
    threshold_deg = 5.0
    acc_axis_5_deg = np.mean(all_abs_errors_deg <= threshold_deg)

    # 4. (可選) 計算 準確率@5度 (Per Object)
    # 這是指有多少個物件，其 *所有* 軸的誤差都小於 5 度 (更嚴格的指標)
    acc_obj_5_deg = np.mean(np.all(all_abs_errors_deg <= threshold_deg, axis=1))

    # 返回所有指標
    return mean_loss, mae_deg, acc_axis_5_deg, acc_obj_5_deg


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
    exp_dir = exp_dir.joinpath('regression') 
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
    args = parse_args() # <--- 修正：main 函數開頭不應有 args，應在這裡解析
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

    # --- 變更 2 & 3: 使用 RegressionDataLoader ---
    train_dataset = RegressionDataLoader(root=data_path, args=args, split='train', process_data=args.process_data)
    test_dataset = RegressionDataLoader(root=data_path, args=args, split='test', process_data=args.process_data)
    # ------------------------------------------

    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=10, drop_last=True)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10)

    '''MODEL LOADING'''
    output_dim = args.output_dim 
    model = importlib.import_module(args.model)
    
    # --- 變更 4 & 5: 複製正確的依賴檔案 ---
    shutil.copy('./models/%s.py' % args.model, str(exp_dir))
    shutil.copy('models/pointnet_utils.py', str(exp_dir)) # <--- 修正：複製回歸模型需要的 utils
    shutil.copy('./train_regression_1d.py', str(exp_dir)) # <--- 修正：複製正確的腳本名稱
    # ------------------------------------

    classifier = model.get_model(output_dim, normal_channel=args.use_normals) 
    criterion = model.get_loss() 
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
    
    best_loss = float('inf') 
    best_epoch = 0 


    '''TRANING'''
    logger.info('Start training...')
    for epoch in range(start_epoch, args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        
        train_loss_list = [] 
        classifier = classifier.train()

        # scheduler.step() # <--- 修正： scheduler.step() 應在 optimizer.step() 之後，或在 epoch 結尾
 
        for batch_id, (points, target) in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
            optimizer.zero_grad()

            points = points.data.numpy()
            #points = provider.random_point_dropout(points)
            #points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            #points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            points = torch.Tensor(points)
            points = points.transpose(2, 1)

            if not args.use_cpu:
                points, target = points.cuda(), target.cuda()

            pred, trans_feat = classifier(points)
            
            target = target.float() # <--- 關鍵修改
            
            # (可選) 處理 1D 標籤的維度問題
            if args.output_dim == 1 and len(target.shape) == 1:
                target = target.unsqueeze(1) # 確保 target 是 [B, 1] 而不是 [B]
            if args.output_dim == 1 and len(pred.shape) == 1:
                pred = pred.unsqueeze(1) # 確保 pred 是 [B, 1]

            loss = criterion(pred, target, trans_feat) 
            
            train_loss_list.append(loss.item()) 
            
            loss.backward()
            optimizer.step()
            global_step += 1

        scheduler.step() # <--- 修正：移到 epoch 迴圈的末尾

        train_mean_loss = np.mean(train_loss_list) 
        log_string('Train Mean Loss: %f' % train_mean_loss) 

        ### 6. 修改驗證迴圈 (Validation Loop Change) ###
        with torch.no_grad():
            val_loss, val_mae_deg, val_acc_axis, val_acc_obj = test(classifier.eval(), testDataLoader, criterion, args) # <--- 修正：傳入 args

            if (val_loss < best_loss): # <--- 修正：使用 '<'
                best_loss = val_loss 
                best_epoch = epoch + 1
            
            log_string('Validation Loss: %f' % val_loss)
            log_string('Validation MAE (Degrees): %.2f' % val_mae_deg)
            log_string('Accuracy (Axis < 5°): %.4f' % val_acc_axis)
            log_string('Accuracy (Object < 5°): %.4f' % val_acc_obj)
            log_string('Best MAE (Degrees): %.2f (at epoch %d)' % (best_loss, best_epoch))

            # 儲存模型的邏輯也改為 val_loss
            if (val_loss <= best_loss): # <--- 修正：這裡應該是 val_loss <= (或 <) best_loss
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': best_epoch,
                    'val_loss': val_loss, 
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
            global_epoch += 1

    logger.info('End of training...')


if __name__ == '__main__':
    args = parse_args() # <--- 修正：在 main 之外先解析一次 args 是多餘的
    main(args) # <--- 修正：直接呼叫 main(args)