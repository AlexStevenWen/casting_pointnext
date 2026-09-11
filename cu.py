from data_utils.ModelNetDataLoader6 import ModelNetDataLoader
import argparse
import numpy as np
import os
import torch
import logging
from tqdm import tqdm
import sys
import importlib

# --- Captum Import ---
from captum.attr import IntegratedGradients
# ---------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Testing')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
    parser.add_argument('--num_category', default=40, type=int ,help='training on ModelNet10/40')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--log_dir', type=str, required=True, help='Experiment root')
    parser.add_argument('--data_dir', type=str, default='data/modelnet40_normal_resampled/', help='Path to dataset')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    parser.add_argument('--num_votes', type=int, default=3, help='Aggregate classification scores with voting')
    return parser.parse_args()


# --- Captum Model Wrapper ---
class PointNetWrapper(torch.nn.Module):
    """
    包裝器類別，使模型 forward 只返回 logits
    Captum 要求模型的 forward 方法只返回一個輸出 (logits)
    而 PointNet/PointNet++ 通常返回 (pred, trans_feat)
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inputs):
        # 呼叫原始模型
        pred, _ = self.model(inputs)
        # 只返回 Captum 需要的 pred (logits)
        return pred
# ----------------------------


def test(model, loader, num_class=40, vote_num=1):
    mean_correct = []
    classifier = model.eval()
    class_acc = np.zeros((num_class, 3))  # [sum_class_acc, class_count, mean_class_acc]

    for j, (points, target) in tqdm(enumerate(loader), total=len(loader)):
        if not args.use_cpu:
            points, target = points.cuda(), target.cuda()

        points = points.transpose(2, 1)
        vote_pool = torch.zeros(target.size(0), num_class).cuda()

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

    # 避免除以 0（除錯保險）
    with np.errstate(divide='ignore', invalid='ignore'):
        class_acc[:, 2] = np.divide(class_acc[:, 0], class_acc[:, 1], out=np.zeros_like(class_acc[:, 0]), where=class_acc[:, 1] != 0)

    valid_class_mask = class_acc[:, 1] != 0
    mean_class_acc = np.mean(class_acc[valid_class_mask, 2]) if np.any(valid_class_mask) else float('nan')
    instance_acc = np.mean(mean_correct)

    return instance_acc, mean_class_acc


def main(args):
    def log_string(str_val):
        logger.info(str_val)
        print(str_val)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    experiment_dir = 'log/classification/' + args.log_dir

    '''LOG'''
    # args = parse_args() # 這行在原始碼中，但似乎是多餘的，因為 main 已經傳入了 args
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # 檢查 eval.txt 是否已經有 handler，避免重複加入
    if not logger.hasHandlers():
        file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    data_path = args.data_dir

    test_dataset = ModelNetDataLoader(root=data_path, args=args, split='test', process_data=False)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10)

    '''MODEL LOADING'''
    num_class = args.num_category
    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    model = importlib.import_module(model_name)

    classifier = model.get_model(num_class, normal_channel=args.use_normals)
    if not args.use_cpu:
        classifier = classifier.cuda()

    checkpoint_path = str(experiment_dir) + '/checkpoints/best_model.pth'
    log_string(f'Loading model from {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path, map_location='cuda' if not args.use_cpu else 'cpu', weights_only=False)
    
    # 1. 正常載入整個預訓練模型的參數字典
    pretrained_dict = checkpoint['model_state_dict']

    # 2. 獲取您當前新建模型 (為15個類別設計) 的參數字典
    model_dict = classifier.state_dict()

    # 3. 過濾掉預訓練模型中尺寸不匹配的層 (也就是最後的分類層)
    #    我們只保留那些名稱存在於當前模型中，且形狀也完全匹配的層。
    filtered_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
    log_string(f'Loaded {len(filtered_dict)} matching layers from checkpoint.')

    # 4. 用過濾後的參數字典，來更新當前模型的參數字典
    model_dict.update(filtered_dict)

    # 5. 將更新後的參數字典載入到您的模型中
    classifier.load_state_dict(model_dict)
    log_string('Model loaded successfully.')


    # ======================================================
    #               !!! Captum 整合開始 !!!
    # ======================================================
    log_string('Running Captum attribution...')

    # 1. 將模型設為評估模式
    classifier.eval()
    
    # 2. 建立 Captum 需要的包裝器
    model_wrapper = PointNetWrapper(classifier)

    # 3. 獲取一個批次 (batch) 的數據來進行解釋
    #    我們使用 iter 從 testDataLoader 中手動取出一筆
    data_iter = iter(testDataLoader)
    try:
        points, target = next(data_iter)
        log_string(f'Loaded one batch for attribution (Batch size: {points.shape[0]})')
    except StopIteration:
        log_string("Test data loader is empty. Cannot run Captum.")
        # 如果為空，直接跳到後續的 test
        points = None

    if points is not None:
        if not args.use_cpu:
            points, target = points.cuda(), target.cuda()

        # 記得進行與 test 函數中相同的轉置！
        # (B, N, C) -> (B, C, N)
        points = points.transpose(2, 1)
        
        # 確保 points 需要計算梯度 (Captum 的要求)
        points.requires_grad_()

        # 4. 選擇一個樣本進行解釋
        #    我們選擇這個 batch 中的第 0 個樣本
        input_tensor = points[0:1]  # Shape: (1, C, N) e.g., (1, 3, 1024)
        target_label = target[0].item() # 這筆資料的真實標籤
        
        # 獲取模型對此樣本的預測
        with torch.no_grad():
             pred_logits, _ = classifier(input_tensor)
             pred_label = torch.argmax(pred_logits, dim=1).item()
        
        log_string(f'Explaining sample 0 (Target class: {target_label}, Predicted class: {pred_label})')

        # 5. 初始化歸因演算法 (Integrated Gradients)
        ig = IntegratedGradients(model_wrapper)

        # 6. 定義基線 (baseline)
        #    對於點雲，一個全零的張量是常見的基線
        baseline = torch.zeros_like(input_tensor)

        # 7. 執行歸因
        #    我們想知道模型為什麼將其預測為 "target_label"
        #    注意：您也可以將 target 設為 pred_label，來解釋 *模型實際預測* 的原因
        attributions, delta = ig.attribute(
            input_tensor,
            baseline,
            target=target_label, # 解釋 "真實標籤"
            # target=pred_label, # 或者: 解釋 "預測標籤"
            return_convergence_delta=True,
            n_steps=50 # n_steps 建議 50-200，數字越大越準確但越慢
        )
        
        log_string(f'Captum Convergence Delta: {delta.item()} (越接近0越好)')

        # 8. 處理結果
        # attributions 的形狀將與 input_tensor 相同: (1, C, N)
        # C 通常是 3 (xyz) 或 6 (xyz + normals)
        # 我們可以將特徵維度(dim=1)上的貢獻度加總(或取絕對值總和)，得到每個點(Point)的總貢獻度
        
        # (1, C, N) -> (1, N) -> (N,)
        point_attributions = attributions.sum(dim=1).squeeze(0) 
        
        log_string(f'Attributions shape (per feature): {attributions.shape}')
        log_string(f'Attributions shape (per point): {point_attributions.shape}')

        # 顯示貢獻度最高的 5 個點的索引
        # .cpu().detach().numpy() 是為了離開 GPU 並轉成 numpy
        sorted_indices = torch.argsort(point_attributions, descending=True)
        log_string(f'Top 5 most important point indices (highest positive attribution):')
        log_string(f'{sorted_indices[:5].cpu().detach().numpy()}')

        # (可選) 顯示貢獻度最低的 5 個點的索引 (最強烈反對的點)
        log_string(f'Top 5 least important point indices (highest negative attribution):')
        log_string(f'{sorted_indices[-5:].cpu().detach().numpy()}')

        # (可選) 儲存結果以便視覺化
        # 建議您將 input_tensor 和 point_attributions 存為 .npy 或 .ply 檔案
        # 以便在 Open3D 或 MeshLab 中進行視覺化
        
        # 獲取原始點雲 (N, C)
        original_points_to_save = input_tensor.squeeze(0).transpose(0, 1).cpu().detach().numpy() # (N, C)
        # 獲取貢獻度 (N,)
        attributions_to_save = point_attributions.cpu().detach().numpy() # (N,)
        
        np.savez(f'./captum_sample_0.npz', 
                  points=original_points_to_save, 
                  attributions=attributions_to_save,
                  target_label=target_label,
                  pred_label=pred_label)
        log_string(f'Saved sample 0 data to ./captum_sample_0.npz')

    # ======================================================
    #               !!! Captum 整合結束 !!!
    # ======================================================


    # 繼續執行原本的測試
    log_string('Starting final evaluation...')
    with torch.no_grad():
        instance_acc, class_acc = test(classifier.eval(), testDataLoader, vote_num=args.num_votes, num_class=num_class)
        log_string('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))


if __name__ == '__main__':
    args = parse_args()
    main(args)