"""
Author: Benny (V8: Added Filename Mapping Support & Class Error Rate Ranking)
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
from sklearn.metrics import confusion_matrix

from pathlib import Path
from tqdm import tqdm
from data_utils.ModelNetDataLoader import ModelNetDataLoader

# 強制使用 Agg 後端，避免無顯示器報錯
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
    if 'test_loss' in history and len(history['test_loss']) > 0:
        plt.plot(epochs, history['test_loss'], 'b-', label='Test Loss')
        
    plt.title('Loss Curve (Train vs Test)')
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
        shape_names_path = os.path.join(data_dir, 'modelnet40_shape_names.txt')
        if not os.path.exists(shape_names_path):
             shape_names_path = os.path.join(data_dir, 'shape_names.txt')
        
        if os.path.exists(shape_names_path):
            with open(shape_names_path, 'r', encoding='utf-8') as f:
                class_names = [line.strip() for line in f]
            return class_names
    except Exception as e:
        print(f"Warning: Could not load class names: {e}")
    
    return [str(i) for i in range(40)]

# *** 新增：讀取檔名對照表 ***
def load_filename_mapping(data_dir):
    """
    讀取 filename_mapping.csv 並轉換為字典
    Key: modelnet_filename (e.g., chair_0001)
    Value: original_filename (e.g., 椅子_A款)
    """
    mapping_path = os.path.join(data_dir, 'filename_mapping.csv')
    if not os.path.exists(mapping_path):
        print(f"Warning: No filename mapping found at {mapping_path}. Original filenames will be unavailable.")
        return {}
    
    try:
        df = pd.read_csv(mapping_path, encoding='utf-8-sig')
        # 建立查表字典：移除 .txt 副檔名來做對應 (假設 mapping 裡不含路徑，只有檔名)
        mapping_dict = dict(zip(df['modelnet_filename'], df['original_filename']))
        return mapping_dict
    except Exception as e:
        print(f"Error loading filename mapping: {e}")
        return {}

def test(model, loader, criterion, num_class=40):
    """ 測試函數：回傳 Acc 與 Test Loss """
    mean_correct = []
    class_acc = np.zeros((num_class, 3))
    classifier = model.eval()
    test_losses = [] 

    for j, (points, target) in tqdm(enumerate(loader), total=len(loader), leave=False):
        if not next(model.parameters()).is_cpu:
            points, target = points.cuda(), target.cuda()

        points = points.transpose(2, 1)
        pred, trans_feat = classifier(points)
        
        loss = criterion(pred, target.long(), trans_feat)
        test_losses.append(loss.item())
        
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
    mean_test_loss = np.mean(test_losses) if len(test_losses) > 0 else 0.0

    return instance_acc, mean_class_acc, mean_test_loss

def run_inference_analysis(model, loaders_dict, exp_dir, class_names, mapping_dict):
    """
    執行完整的推論分析 (包含檔名還原)
    """
    print("\n[Analysis] Generating consolidated Train/Test report with original filenames...")
    model.eval()
    
    full_results_list = [] 
    test_targets = []
    test_preds = []
    
    # 遍歷所有 DataLoader
    for split_name, loader in loaders_dict.items():
        print(f"Processing {split_name} set...")
        
        # 取得 dataset 中的檔案列表 (通常是 self.datapath)
        # ModelNetDataLoader 一般格式: self.datapath = [(cls_name, file_path), ...]
        dataset_files = []
        if hasattr(loader.dataset, 'datapath'):
            dataset_files = loader.dataset.datapath
        else:
            print(f"Warning: Could not access file list in {split_name} loader. Filename mapping may fail.")
        
        global_idx = 0
        
        for j, (points, target) in tqdm(enumerate(loader), total=len(loader), desc=f"Analyzing {split_name}"):
            if not next(model.parameters()).is_cpu:
                points, target = points.cuda(), target.cuda()

            points = points.transpose(2, 1)
            pred, _ = model(points)
            pred_choice = pred.data.max(1)[1]
            
            pred_np = pred_choice.cpu().numpy()
            target_np = target.cpu().numpy()
            
            if split_name == 'Test':
                test_targets.extend(target_np)
                test_preds.extend(pred_np)
            
            batch_size = len(pred_np)
            
            for i in range(batch_size):
                # 嘗試取得對應的檔名
                modelnet_filename = "Unknown"
                original_filename = "Unknown"
                
                if dataset_files and (global_idx + i) < len(dataset_files):
                    # dataset_files[idx] 通常是 (class_name, full_path)
                    file_path = dataset_files[global_idx + i][1]
                    # 取得檔名 (不含路徑，不含副檔名)
                    base_name = os.path.splitext(os.path.basename(file_path))[0] # e.g., chair_0001
                    modelnet_filename = base_name
                    
                    # 透過 mapping 還原
                    # 嘗試各種 key 格式 (因為不確定 mapping 裡的存法是否有副檔名)
                    if base_name in mapping_dict:
                        original_filename = mapping_dict[base_name]
                    elif f"{base_name}.txt" in mapping_dict:
                        original_filename = mapping_dict[f"{base_name}.txt"]
                    elif f"{base_name}.h5" in mapping_dict:
                         original_filename = mapping_dict[f"{base_name}.h5"]

                is_correct = (pred_np[i] == target_np[i])
                status_str = "Correct" if is_correct else "Wrong"
                
                result_info = {
                    "Split": split_name,
                    "Sample_Index": global_idx + i,
                    "ModelNet_Filename": modelnet_filename, # 新增
                    "Original_Filename": original_filename, # 新增
                    "Status": status_str,
                    "True_Name": class_names[int(target_np[i])],
                    "Pred_Name": class_names[int(pred_np[i])],
                    "True_ID": target_np[i],
                    "Pred_ID": pred_np[i]
                }
                full_results_list.append(result_info)
            
            global_idx += batch_size

    # --- 1. 儲存大表 (CSV) ---
    df_results = pd.DataFrame(full_results_list)
    # 調整欄位順序
    cols = ["Split", "Status", "Original_Filename", "ModelNet_Filename", "True_Name", "Pred_Name", "True_ID", "Pred_ID"]
    # 確保只取存在的欄位
    existing_cols = [c for c in cols if c in df_results.columns]
    df_results = df_results[existing_cols]
    
    csv_path = os.path.join(exp_dir, 'inference_full_report.csv')
    df_results.to_csv(csv_path, index=False, encoding='utf-8-sig') # 使用 utf-8-sig 支援中文
    print(f"[Analysis] Full report saved to: {csv_path}")

    # ==========================================
    #  詳細的各類別錯誤率計算與繪圖
    # ==========================================
    if len(test_targets) > 0:
        # 計算 Confusion Matrix
        cm = confusion_matrix(test_targets, test_preds, labels=range(len(class_names)))
        
        # 取得每個類別的總數與正確數
        per_class_total = cm.sum(axis=1)
        per_class_correct = cm.diagonal()
        
        # 避免除以零
        with np.errstate(divide='ignore', invalid='ignore'):
            per_class_acc = per_class_correct / per_class_total
        per_class_acc = np.nan_to_num(per_class_acc) # 若該類別無樣本，設為0
        per_class_error = 1.0 - per_class_acc

        # 建立統計摘要表
        df_summary = pd.DataFrame({
            'Class': class_names,
            'Total_Samples': per_class_total,
            'Correct_Samples': per_class_correct,
            'Accuracy': per_class_acc,
            'Error_Rate': per_class_error
        })
        
        # 儲存摘要表 CSV
        summary_path = os.path.join(exp_dir, 'class_error_summary.csv')
        df_summary.to_csv(summary_path, index=False)
        print(f"[Analysis] Class error summary saved to: {summary_path}")

        # --- 2. 繪製「錯誤率排行榜」長條圖 (Bar Chart) ---
        df_plot = df_summary.sort_values(by='Error_Rate', ascending=False)
        
        plt.figure(figsize=(12, 10))
        sns.barplot(data=df_plot, x='Error_Rate', y='Class', palette='Reds_r')
        
        plt.title('Error Rate by Class (Hardest Classes on Top)', fontsize=15)
        plt.xlabel('Error Rate (0.0 - 1.0)', fontsize=12)
        plt.ylabel('Category', fontsize=12)
        plt.xlim(0, 1.0) 
        plt.grid(axis='x', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(exp_dir, 'class_error_ranking.png'))
        plt.close()
        
        # --- 3. 繪製混淆矩陣熱力圖 ---
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        plt.figure(figsize=(20, 18))
        sns.heatmap(cm_norm, annot=False, fmt=".2f", cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Normalized Confusion Matrix (Test Set)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(exp_dir, 'confusion_matrix_heatmap.png'))
        plt.close()

        # --- 4. 文字報告更新 ---
        report_path = os.path.join(exp_dir, 'final_analysis_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=== Final Analysis Report ===\n")
            f.write(f"Overall Accuracy: {(np.array(test_targets) == np.array(test_preds)).mean():.4f}\n\n")
            f.write("--- Top 5 High Error Classes ---\n")
            for i in range(min(5, len(df_plot))):
                row = df_plot.iloc[i]
                f.write(f"{i+1}. {row['Class']} : {row['Error_Rate']*100:.1f}% Error ({row['Correct_Samples']}/{row['Total_Samples']})\n")
            
            f.write("\nFor full details, see 'class_error_summary.csv' and 'inference_full_report.csv'\n")

def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
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

    '''LOG'''
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    log_string(f'Experiment Dir: {exp_dir}')
    log_string(str(args))

    '''DATA LOADING'''
    log_string('Load dataset ...')
    train_dataset = ModelNetDataLoader(root=args.data_dir, args=args, split='train', process_data=args.process_data)
    test_dataset = ModelNetDataLoader(root=args.data_dir, args=args, split='test', process_data=args.process_data)
    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=10, drop_last=True)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10)

    class_names = load_class_names(args.data_dir)
    # *** 載入 Mapping ***
    mapping_dict = load_filename_mapping(args.data_dir)

    '''MODEL LOADING'''
    model = importlib.import_module(args.model)
    shutil.copy('./models/%s.py' % args.model, str(exp_dir))
    shutil.copy(__file__, str(exp_dir))

    classifier = model.get_model(args.num_category, normal_channel=args.use_normals)
    criterion = model.get_loss()
    classifier.apply(inplace_relu)

    if not args.use_cpu:
        classifier = classifier.cuda()
        criterion = criterion.cuda()

    try:
        # Safe load
        checkpoint = torch.load(str(exp_dir) + '/checkpoints/best_model.pth', weights_only=False)
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
    
    history = {'train_loss': [], 'test_loss': [], 'test_instance_acc': [], 'test_class_acc': []}
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
        avg_train_loss = np.mean(epoch_loss)
        history['train_loss'].append(avg_train_loss)
        
        log_string(f'Train Instance Accuracy: {train_instance_acc:.6f}, Loss: {avg_train_loss:.6f}')

        with torch.no_grad():
            instance_acc, class_acc, test_loss = test(classifier.eval(), testDataLoader, criterion, num_class=args.num_category)
            
            history['test_instance_acc'].append(instance_acc)
            history['test_class_acc'].append(class_acc)
            history['test_loss'].append(test_loss)

            if (instance_acc >= best_instance_acc):
                best_instance_acc = instance_acc
                best_epoch = epoch + 1
                savepath = str(checkpoints_dir) + '/best_model.pth'
                state = {
                    'epoch': best_epoch, 
                    'instance_acc': instance_acc, 
                    'class_acc': class_acc, 
                    'model_state_dict': classifier.state_dict(), 
                    'optimizer_state_dict': optimizer.state_dict()
                }
                torch.save(state, savepath)
                log_string('Saving Best Model...')

            if (class_acc >= best_class_acc):
                best_class_acc = class_acc
                
            log_string(f'Test Instance Accuracy: {instance_acc:.6f}, Class Accuracy: {class_acc:.6f}, Test Loss: {test_loss:.6f}')
            
            plot_performance(exp_dir, history)
            
        scheduler.step()

    log_string('End of training...')
    log_string('Starting Final Detailed Analysis (Train + Test)...')

    # --- Load Best Model ---
    best_checkpoint = torch.load(str(checkpoints_dir) + '/best_model.pth', weights_only=False)
    classifier.load_state_dict(best_checkpoint['model_state_dict'])
    
    # *** 關鍵修改：重新建立 shuffle=False 的 Train DataLoader ***
    # 這是為了讓 inference 時的 index 順序能與 dataset.datapath 列表對齊，
    # 否則隨機打亂後，我們無法得知哪個預測結果對應哪個檔名。
    analysis_train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10)
    
    analysis_loaders = {'Train': analysis_train_loader, 'Test': testDataLoader}

    with torch.no_grad():
        run_inference_analysis(classifier, analysis_loaders, exp_dir, class_names, mapping_dict)
    
    log_string(f'All Done. Check results in: {exp_dir}')

if __name__ == '__main__':
    args = parse_args()
    main(args)