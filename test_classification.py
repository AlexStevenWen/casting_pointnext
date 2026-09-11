"""
Author: Benny
Date: Nov 2019
"""
from data_utils.ModelNetDataLoader import ModelNetDataLoader
import argparse
import numpy as np
import os
import torch
import logging
from tqdm import tqdm
import sys
import importlib

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
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    experiment_dir = 'log/classification/' + args.log_dir

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    #data_path = 'data/modelnet40_normal_resampled/'
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
    """
    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth', weights_only=False)
    classifier.load_state_dict(checkpoint['model_state_dict'])
    """
    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth', weights_only=False)
    # 1. 正常載入整個預訓練模型的參數字典
    pretrained_dict = checkpoint['model_state_dict']

    # 2. 獲取您當前新建模型 (為15個類別設計) 的參數字典
    model_dict = classifier.state_dict()

    # 3. 過濾掉預訓練模型中尺寸不匹配的層 (也就是最後的分類層)
    #    我們只保留那些名稱存在於當前模型中，且形狀也完全匹配的層。
    filtered_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}

    # 4. 用過濾後的參數字典，來更新當前模型的參數字典
    model_dict.update(filtered_dict)

    # 5. 將更新後的參數字典載入到您的模型中
    classifier.load_state_dict(model_dict)

    with torch.no_grad():
        instance_acc, class_acc = test(classifier.eval(), testDataLoader, vote_num=args.num_votes, num_class=num_class)
        log_string('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))


if __name__ == '__main__':
    args = parse_args()
    main(args)
