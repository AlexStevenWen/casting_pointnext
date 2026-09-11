"""
Author: Benny (Modified with Deep Error Analysis & Visualization)
Date: 2025 Revised
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
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

from pathlib import Path
from tqdm import tqdm
from data_utils.ModelNetDataLoader import ModelNetDataLoader

# 強制使用 Agg 後端，解決無顯示器環境報錯
plt.switch_backend('agg')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
    parser.add_argument('--model', default='pointnet_cls', help='model name [default: pointnet_cls]')
    parser.add_argument('--num_category', default=40, type=int,  help='training on ModelNet10/40')
    parser.add_argument('--epoch', default=200, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    parser.add_argument('--data_dir', type=str, default='data/modelnet40_normal_resampled/', help='path to dataset')
    return parser.parse_args()

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def plot_performance(exp_dir, history):
    """ 繪製 Loss 與 Accuracy 曲線 """
    epochs = range(1, len(history['train_loss']) + 1)
    
    # 1. Loss 曲線
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['train_loss'], 'r-', label='Train Loss')
    plt.title('Training Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(exp_dir, 'loss_curves.png'))
    plt.close()

    # 2. Accuracy 曲線
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['test_instance_acc'], 'b-', label='Instance Acc')
    plt.plot(epochs, history['test_class_acc'], 'g--', label='Class Acc')
    plt.title('Test Accuracy Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(exp_dir, 'accuracy_curves.png'))
    plt.close('all')

def load_class_names(data_dir):
    """ 嘗試讀取類別名稱 """
    try:
        # 假設 modelnet40_shape_names.txt 在資料夾中
        shape_names_path = os.path.join(data_dir, 'modelnet40_shape_names.txt')
        if not os.path.exists(shape_names_path):
             shape_names_path = os.path.join(data_dir, 'shape_names.txt')
        
        if os.path.exists(shape_names_path):
            with open(shape_names_path, 'r') as f:
                class_names = [line.strip() for line in f]
            return class_names
    except Exception as e:
        print(f"Warning: Could not load class names: {e}")
    
    # 如果找不到檔案，回傳數字字串
    return [str(i) for i in range(40)]

def analyze_errors(model, loader, num_class, exp_dir, class_names):
    """
    深度錯誤分析函數：
    1. 計算混淆矩陣
    2. 儲存錯誤清單 (哪一個樣本錯了)
    3. 繪製錯誤率分析圖
    """
    print("\n[Analysis] Starting detailed error analysis...")
    model.eval()
    all_preds = []
    all_targets = []
    
    # 用於儲存錯誤的具體資訊
    error_list = [] # [Index, True_Label, Pred_Label]
    
    global_idx = 0 # 用於追蹤是第幾個測試樣本
    
    for j, (points, target) in tqdm(enumerate(loader), total=len(loader), desc="Analyzing"):
        if not next(model.parameters()).is_cpu:
            points, target = points.cuda(), target.cuda()

        points = points.transpose(2, 1)
        pred, _ = model(points)
        pred_choice = pred.data.max(1)[1]
        
        # 轉回 numpy
        pred_np = pred_choice.cpu().numpy()
        target_np = target.cpu().numpy()
        
        all_preds.extend(pred_np)
        all_targets.extend(target_np)
        
        # 記錄每一個錯誤
        for i in range(len(pred_np)):
            if pred_np[i] != target_np[i]:
                error_info = {
                    "Test_Index": global_idx,
                    "True_ID": target_np[i],
                    "True_Name": class_names[int(target_np[i])],
                    "Pred_ID": pred_np[i],
                    "Pred_Name": class_names[int(pred_np[i])]
                }
                error_list.append(error_info)
            global_idx += 1

    # --- 1. 儲存詳細錯誤清單 CSV ---
    df_errors = pd.DataFrame(error_list)
    csv_path = os.path.join(exp_dir, 'inference_error_details.csv')
    df_errors.to_csv(csv_path, index=False)
    print(f"[Analysis] Detailed error list saved to: {csv_path}")

    # --- 2. 混淆矩陣 (Confusion Matrix) ---
    cm = confusion_matrix(all_targets, all_preds)
    
    # 計算正規化混淆矩陣 (看比例)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(20, 18))
    sns.heatmap(cm_norm, annot=False, fmt=".2f", cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Normalized Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, 'confusion_matrix_heatmap.png'))
    plt.close()

    # --- 3. 各類別錯誤率 (Error Rate per Class) ---
    # 準確率 = 對角線 / 該列總和
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    per_class_error = 1.0 - per_class_acc
    
    # 建立 Dataframe 方便繪圖
    df_class_perf = pd.DataFrame({
        'Class': class_names,
        'Error_Rate': per_class_error,
        'Accuracy': per_class_acc
    })
    # 依照錯誤率排序 (高的在上面)
    df_class_perf = df_class_perf.sort_values(by='Error_Rate', ascending=False)

    plt.figure(figsize=(12, 10))
    sns.barplot(x='Error_Rate', y='Class', data=df_class_perf, palette='Reds_r')
    plt.title('Error Rate by Category (Which parts are hardest?)')
    plt.xlabel('Error Rate (0.0 - 1.0)')
    plt.xlim(0, 1.0)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, 'class_error_ranking.png'))
    plt.close()

    # --- 4. 文字報告 (Top Confusions) ---
    report_path = os.path.join(exp_dir, 'final_analysis_report.txt')
    with open(report_path, 'w') as f:
        f.write("=== Inference Analysis Report ===\n")
        f.write(f"Total Test Samples: {len(all_targets)}\n")
        f.write(f"Total Errors: {len(error_list)}\n")
        f.write(f"Overall Error Rate: {len(error_list)/len(all_targets):.4f}\n\n")
        
        f.write("--- Top 5 Hardest Classes (High Error Rate) ---\n")
        for i in range(5):
            row = df_class_perf.iloc[i]
            f.write(f"{i+1}. {row['Class']}: {row['Error_Rate']*100:.2f}% Error\n")
        
        f.write("\n--- Major Confusions (Where did it go wrong?) ---\n")
        # 找出非對角線中最大的值 (最常被誤判成什麼)
        np.fill_diagonal(cm, 0) # 忽略正確預測
        # Flatten 並排序
        indices = np.dstack(np.unravel_index(np.argsort(cm.ravel()), cm.shape))[0][::-1]
        
        count = 0
        for idx in indices:
            true_idx, pred_idx = idx
            if cm[true_idx, pred_idx] > 0 and count < 10:
                f.write(f"True: '{class_names[true_idx]}' was predicted as '{class_names[pred_idx]}' -> {cm[true_idx, pred_idx]} times\n")
                count += 1
                
    print(f"[Analysis] Text report saved to: {report_path}")


def test(model, loader, num_class=40):
    mean_correct = []
    class_acc = np.zeros((num_class, 3))  # [sum_correct_ratio, class_count, final_acc]
    classifier = model.eval()

    for j, (points, target) in tqdm(enumerate(loader), total=len(loader), leave=False):
        if not next(model.parameters()).is_cpu:
            points, target = points.cuda(), target.cuda()

        points = points.transpose(2, 1)
        pred, _ = classifier(points)
        pred_choice = pred.data.max(1)[1]

        for cat in np.unique(target.cpu()):
            cat = int(cat)
            mask = (target == cat)
            num = mask.sum().item()
            if num > 0:
                correct = pred_choice[mask].eq(target[mask]).sum().item()
                class_acc[cat, 0] += correct / float(num)
                class_acc[cat, 1] += 1

        correct = pred_choice.eq(target).sum().item()
        mean_correct.append(correct / float(points.size(0)))

    with np.errstate(divide='ignore', invalid='ignore'):
        class_acc[:, 2] = np.divide(class_acc[:, 0], class_acc[:, 1], out=np.zeros_like(class_acc[:, 0]), where=class_acc[:, 1] != 0)

    valid_class_mask = class_acc[:, 1] != 0
    mean_class_acc = np.mean(class_acc[valid_class_mask, 2]) if np.any(valid_class_mask) else float('nan')
    instance_acc = np.mean(mean_correct)

    return instance_acc, mean_class_acc

def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR WITH MODEL NAME AND TIMESTAMP'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    if args.log_dir is None:
        folder_name = f"{args.model}_{timestr}"
    else:
        folder_name = f"{args.log_dir}_{timestr}"
    
    exp_dir = Path('./log/classification/').joinpath(folder_name)
    exp_dir.mkdir(exist_ok=True, parents=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG WITH TIMESTAMPS'''
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    log_string(f'實驗資料夾建立於: {exp_dir}')
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    train_dataset = ModelNetDataLoader(root=args.data_dir, args=args, split='train', process_data=args.process_data)
    test_dataset = ModelNetDataLoader(root=args.data_dir, args=args, split='test', process_data=args.process_data)
    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=10, drop_last=True)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10)

    # 載入類別名稱以供後續分析使用
    class_names = load_class_names(args.data_dir)
    log_string(f"Loaded {len(class_names)} classes: {class_names[:5]}...")

    '''MODEL LOADING'''
    model = importlib.import_module(args.model)
    shutil.copy('./models/%s.py' % args.model, str(exp_dir))
    shutil.copy(__file__, str(exp_dir)) # 備份此執行檔

    classifier = model.get_model(args.num_category, normal_channel=args.use_normals)
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
        optimizer = torch.optim.Adam(classifier.parameters(), lr=args.learning_rate, weight_decay=args.decay_rate)
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    
    history = {'train_loss': [], 'test_instance_acc': [], 'test_class_acc': []}
    best_instance_acc = 0.0
    best_class_acc = 0.0

    '''TRAINING'''
    log_string('Start training...')
    for epoch in range(start_epoch, args.epoch):
        log_string(f'--- Epoch {epoch + 1} ({epoch + 1}/{args.epoch}) ---')
        mean_correct = []
        epoch_loss = []
        classifier = classifier.train()

        for batch_id, (points, target) in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
            optimizer.zero_grad()
            points = points.data.numpy()
            points = provider.random_point_dropout(points)
            points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            points = torch.Tensor(points).transpose(2, 1)

            if not args.use_cpu:
                points, target = points.cuda(), target.cuda()

            pred, trans_feat = classifier(points)
            loss = criterion(pred, target.long(), trans_feat)
            loss.backward()
            optimizer.step()
            
            epoch_loss.append(loss.item())
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))

        train_instance_acc = np.mean(mean_correct)
        avg_loss = np.mean(epoch_loss)
        history['train_loss'].append(avg_loss)
        
        log_string(f'Train Instance Accuracy: {train_instance_acc:.6f}, Loss: {avg_loss:.6f}')

        with torch.no_grad():
            instance_acc, class_acc = test(classifier.eval(), testDataLoader, num_class=args.num_category)
            history['test_instance_acc'].append(instance_acc)
            history['test_class_acc'].append(class_acc)

            if (instance_acc >= best_instance_acc):
                best_instance_acc = instance_acc
                best_epoch = epoch + 1
                savepath = str(checkpoints_dir) + '/best_model.pth'
                state = {'epoch': best_epoch, 'instance_acc': instance_acc, 'class_acc': class_acc, 'model_state_dict': classifier.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}
                torch.save(state, savepath)
                log_string('Saving Best Model...')

            if (class_acc >= best_class_acc):
                best_class_acc = class_acc
                
            log_string(f'Test Instance Accuracy: {instance_acc:.6f}, Class Accuracy: {class_acc:.6f}')
            
            # 更新圖表
            plot_performance(exp_dir, history)
            
        scheduler.step()

    log_string('End of training...')
    log_string('Starting Final Error Analysis on Best Model...')

    # --- 最後一步：載入最佳模型並生成詳細報告 ---
    # 重新載入最佳權重
    best_checkpoint = torch.load(str(checkpoints_dir) + '/best_model.pth')
    classifier.load_state_dict(best_checkpoint['model_state_dict'])
    
    # 執行分析
    with torch.no_grad():
        analyze_errors(classifier, testDataLoader, args.num_category, exp_dir, class_names)
    
    log_string(f'Analysis Complete. Check results in: {exp_dir}')

if __name__ == '__main__':
    args = parse_args()
    main(args)