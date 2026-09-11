"""
Author: Benny (Modified for Casting NOCLS Test with Diagnostic Metrics)
Date: 2025 Revised
"""
import argparse
import os
from data_utils.ShapeNetDataLoader import PartNormalDataset
import torch
import logging
import sys
import importlib
from tqdm import tqdm
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size in testing')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--num_point', type=int, default=2048, help='point Number')
    parser.add_argument('--log_dir', type=str, required=True, help='experiment root (e.g., sem_run)')
    parser.add_argument('--normal', action='store_true', default=False, help='use normals')
    parser.add_argument('--num_votes', type=int, default=3, help='aggregate segmentation scores with voting')
    parser.add_argument('--data_dir', type=str, required=True, help='data directory')
    return parser.parse_args()

def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    experiment_dir = 'log/part_seg/' + args.log_dir

    '''LOG'''
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    # --- 1. 動態加載資料集 ---
    root = args.data_dir
    TEST_DATASET = PartNormalDataset(root=root, npoints=args.num_point, split='test', normal_channel=args.normal)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=4)
    log_string("The number of test data is: %d" % len(TEST_DATASET))

    # --- 2. 動態獲取類別資訊 ---
    seg_classes = TEST_DATASET.seg_classes
    num_classes = len(TEST_DATASET.classes)
    
    all_parts = []
    for cat in seg_classes:
        all_parts.extend(seg_classes[cat])
    num_part = len(set(all_parts))

    seg_label_to_cat = {}
    for cat in seg_classes.keys():
        for label in seg_classes[cat]:
            seg_label_to_cat[label] = cat

    '''MODEL LOADING'''
    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    MODEL = importlib.import_module(model_name)
    
    try:
        classifier = MODEL.get_model(num_part, normal_channel=args.normal).cuda()
    except TypeError:
        classifier = MODEL.get_model(num_part, num_classes=num_classes, normal_channel=args.normal).cuda()
        
    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth', weights_only=False)
    classifier.load_state_dict(checkpoint['model_state_dict'])

    with torch.no_grad():
        test_metrics = {}
        total_correct = 0
        total_seen = 0
        
        # 用於診斷散落澆口的關鍵數據結構
        part_ious = {l: [] for l in range(num_part)}
        shape_ious = {cat: [] for cat in seg_classes.keys()}

        classifier = classifier.eval()
        for batch_id, (points, label, target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
            cur_batch_size, NUM_POINT, _ = points.size()
            points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()
            points = points.transpose(2, 1)
            
            # 投票機制 (Voting)
            vote_pool = torch.zeros(cur_batch_size, NUM_POINT, num_part).cuda()
            for _ in range(args.num_votes):
                try:
                    seg_pred, _ = classifier(points)
                except TypeError:
                    # 兼容需要標籤輸入的模型
                    from train_partseg_nocls import to_categorical
                    seg_pred, _ = classifier(points, to_categorical(label, num_classes))
                vote_pool += seg_pred

            seg_pred = vote_pool / args.num_votes
            cur_pred_val = seg_pred.cpu().data.numpy()
            cur_pred_val = np.argmax(cur_pred_val, 2) # [B, N]
            target_np = target.cpu().data.numpy()

            # 計算點準確率
            total_correct += np.sum(cur_pred_val == target_np)
            total_seen += (cur_batch_size * NUM_POINT)

            # 計算 IoU
            for i in range(cur_batch_size):
                segp = cur_pred_val[i, :]
                segl = target_np[i, :]
                cat = seg_label_to_cat[segl[0]]
                
                instance_part_ious = []
                for l in seg_classes[cat]:
                    # 計算該部件的 Intersection 與 Union
                    I = np.sum((segl == l) & (segp == l))
                    U = np.sum((segl == l) | (segp == l))
                    
                    if U == 0:
                        iou = 1.0  # 如果該樣本中沒有這個部件且沒誤判
                    else:
                        iou = I / float(U)
                    
                    instance_part_ious.append(iou)
                    part_ious[l].append(iou) # 紀錄各部件獨立表現
                
                shape_ious[cat].append(np.mean(instance_part_ious))

        # 彙整結果
        all_shape_ious = []
        for cat in shape_ious.keys():
            all_shape_ious.extend(shape_ious[cat])
            shape_ious[cat] = np.mean(shape_ious[cat])
        
        test_metrics['accuracy'] = total_correct / float(total_seen)
        test_metrics['class_avg_iou'] = np.mean(list(shape_ious.values()))
        test_metrics['instance_avg_iou'] = np.mean(all_shape_ious)

    # --- 診斷報告輸出 ---
    log_string('='*30)
    log_string('FINAL DIAGNOSTIC REPORT')
    log_string('='*30)
    log_string('Overall Test Accuracy: %.5f' % test_metrics['accuracy'])
    log_string('Overall Instance mIOU: %.5f' % test_metrics['instance_avg_iou'])
    log_string('-'*30)
    
    # 重點：輸出 Body vs Gate 的獨立表現
    # 假設零件編號 0 為本體，1 為澆口
    for l in range(num_part):
        part_name = "Body (0)" if l == 0 else "Gate (1)"
        avg_p_iou = np.mean(part_ious[l])
        log_string('Part %s IoU: %.5f' % (part_name.ljust(10), avg_p_iou))
    
    log_string('-'*30)
    for cat in sorted(shape_ious.keys()):
        log_string('mIoU of %s: %.5f' % (cat.ljust(14), shape_ious[cat]))

if __name__ == '__main__':
    args = parse_args()
    main(args)