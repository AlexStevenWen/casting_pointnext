"""
Author: Benny (Modified for Comprehensive Evaluation & Visualization)
Date: 2025 Revised
"""
import argparse
import numpy as np
import os
import torch
import logging
from tqdm import tqdm
import sys
import importlib
import datetime
import matplotlib.pyplot as plt
from pathlib import Path
from data_utils.ModelNetDataLoader import ModelNetDataLoader

# 強制使用 Agg 後端，解決無 GUI 環境報錯
plt.switch_backend('agg')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Testing')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in testing')
    parser.add_argument('--num_category', default=40, type=int, help='ModelNet10/40')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--log_dir', type=str, required=True, help='Experiment root')
    parser.add_argument('--data_dir', type=str, default='data/modelnet40_normal_resampled/', help='Path to dataset')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampling')
    parser.add_argument('--num_votes', type=int, default=3, help='Voting number')
    parser.add_argument('--export_onnx', action='store_true', help='Export model to ONNX for Netron')
    return parser.parse_args()

def plot_class_acc(exp_dir, class_acc_values, class_names):
    """ 繪製詳細的各類別準確率圖表（自動適應長標籤與類別數量） """
    num_classes = len(class_names)
    
    # 1. 根據類別數量與標籤長度，動態調整畫布大小 (類別多或字串長就自動放大)
    max_label_len = max([len(name) for name in class_names])
    fig_width = max(12, max_label_len * 0.4)
    fig_height = max(10, num_classes * 0.35)
    
    plt.figure(figsize=(fig_width, fig_height))
    y_pos = np.arange(num_classes)
    
    # 2. 動態調整字型大小，避免擁擠
    font_size = max(8, min(12, int(200 / num_classes)))
    
    # 建立漸層色效果
    colors = plt.cm.viridis(np.linspace(0, 1, num_classes))
    
    bars = plt.barh(y_pos, class_acc_values, align='center', color=colors, alpha=0.8)
    
    # 設置 Y 軸標籤與字型大小
    plt.yticks(y_pos, class_names, fontsize=font_size)
    plt.xlabel('Accuracy Score', fontsize=font_size + 2)
    
    # 3. 增大標頭字體，使其更醒目
    plt.title('Detailed Accuracy per Class (Test Set)', fontsize=font_size + 4, pad=15)
    plt.xlim(0, 1.1)
    
    # 在柱狀圖末端標註數值
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                 f'{width:.2f}', va='center', fontsize=font_size - 1)

    plt.grid(axis='x', linestyle='--', alpha=0.6)
    
    # 4. 強制自動調整邊距，確保長標籤 (如 T_cross_and_Round_Square) 完美顯示不被切到
    plt.tight_layout()
    
    plt.savefig(os.path.join(exp_dir, 'class_accuracy_report.png'), dpi=300) # 提升儲存解析度
    plt.close()

def test(model, loader, num_class=40, vote_num=1):
    mean_correct = []
    classifier = model.eval()
    class_acc = np.zeros((num_class, 3))  # [sum_acc, count, mean_acc]

    for j, (points, target) in tqdm(enumerate(loader), total=len(loader)):
        if not next(model.parameters()).is_cpu:
            points, target = points.cuda(), target.cuda()

        # 維度處理: (B, N, C) -> (B, C, N)
        points = points.transpose(2, 1) 
        device = 'cuda' if not next(model.parameters()).is_cpu else 'cpu'
        vote_pool = torch.zeros(target.size(0), num_class).to(device)

        for _ in range(vote_num):
            pred, _ = classifier(points)
            vote_pool += pred

        pred = vote_pool / vote_num
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

    # 計算各類平均準確率
    for i in range(num_class):
        if class_acc[i, 1] > 0:
            class_acc[i, 2] = class_acc[i, 0] / class_acc[i, 1]

    valid_mask = class_acc[:, 1] > 0
    mean_class_acc = np.mean(class_acc[valid_mask, 2])
    instance_acc = np.mean(mean_correct)

    return instance_acc, mean_class_acc, class_acc[valid_mask, 2]

def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # 1. 建立獨立評估目錄 (包含原始日誌目錄名稱與目前時間)
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    raw_exp_dir = Path('./log/classification/').joinpath(args.log_dir)
    eval_dir = raw_exp_dir.joinpath(f'eval_{timestr}')
    eval_dir.mkdir(exist_ok=True, parents=True)

    # 2. 日誌設定 (帶精確時間戳記)
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler = logging.FileHandler(f'{eval_dir}/evaluation_log.txt')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    log_string(f'評估目錄已建立: {eval_dir}')
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Loading Test Dataset...')
    test_dataset = ModelNetDataLoader(root=args.data_dir, args=args, split='test', process_data=False)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    '''MODEL LOADING'''
    model_log_dir = raw_exp_dir.joinpath('logs')
    # 讀取 logs 資料夾下的第一個檔名作為基礎模型名稱
    try:
        raw_model_name = os.listdir(str(model_log_dir))[0].split('.')[0]
    except IndexError:
        raw_model_name = 'pointnet_cls' # 預設回退
        
    model_name = raw_model_name + '_keypoint'
    
    try:
        model = importlib.import_module(model_name)
    except ImportError:
        model = importlib.import_module(raw_model_name)

    classifier = model.get_model(args.num_category, normal_channel=args.use_normals)
    if not args.use_cpu:
        classifier = classifier.cuda()

    # 載入 Best Checkpoint
    checkpoint_path = raw_exp_dir.joinpath('checkpoints/best_model.pth')
    checkpoint = torch.load(str(checkpoint_path), map_location='cuda' if not args.use_cpu else 'cpu')
    classifier.load_state_dict(checkpoint['model_state_dict'], strict=False)

    log_string(f'Successfully loaded model: {model_name} from {checkpoint_path}')

    # 可選功能：導出 ONNX 以供 Netron 可視化
    if args.export_onnx:
        try:
            onnx_path = eval_dir.joinpath('model_structure.onnx')
            dummy_input = torch.randn(1, 6 if args.use_normals else 3, args.num_point).cuda()
            torch.onnx.export(classifier, dummy_input, str(onnx_path), opset_version=11)
            log_string(f'ONNX 模型已導出至: {onnx_path} (可用 Netron 開啟)')
        except Exception as e:
            log_string(f'導出 ONNX 失敗: {e}')

    '''TESTING'''
    with torch.no_grad():
        instance_acc, mean_class_acc, all_class_accs = test(classifier.eval(), testDataLoader, 
                                                           vote_num=args.num_votes, num_class=args.num_category)
        
        log_string(f'Evaluation Results:')
        log_string(f'Overall Instance Accuracy: {instance_acc:.6f}')
        log_string(f'Mean Class Accuracy: {mean_class_acc:.6f}')
        
        # 取得類別名稱檔案
        try:
            class_names = [line.rstrip() for line in open(os.path.join(args.data_dir, 'modelnet40_shape_names.txt'))]
            class_names = class_names[:len(all_class_accs)]
            plot_class_acc(eval_dir, all_class_accs, class_names)
            log_string(f'各類別準確率圖表已儲存至: {eval_dir}/class_accuracy_report.png')
        except Exception as e:
            log_string(f'繪圖失敗: {e}')

if __name__ == '__main__':
    args = parse_args()
    main(args)