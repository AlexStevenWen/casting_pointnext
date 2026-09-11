# *_*coding:utf-8 *_*
"""
Author: Benny (Modified for Soft Label / Probability Distribution)
Date: 2025 Revised
Description: 
    - 加入 Jitter (噪聲) 增強策略
"""
import os
import json
import warnings
import numpy as np
import torch
from torch.utils.data import Dataset

warnings.filterwarnings('ignore')

def pc_normalize(pc):
    """ 將點雲歸一化到單位球體內 [-1, 1] """
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc

# [新增] Jitter 增強函數
def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """ 
    對點雲加入高斯噪聲 
    sigma: 噪聲標準差 (建議 0.005 ~ 0.01)
    clip: 噪聲最大值限制
    """
    N, C = batch_data.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    batch_data += jittered_data
    return batch_data

class PartNormalDataset(Dataset):
    def __init__(self, root='./data/shapenetcore_partanno_segmentation_benchmark_v0_normal', 
                 npoints=2500, split='train', class_choice=None, normal_channel=False,
                 augment=False, grid_num=4): 
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}
        self.normal_channel = normal_channel
        self.augment = augment 
        self.grid_num = grid_num 

        # 1. 讀取類別對應表
        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        
        self.classes_original = dict(zip(self.cat, range(len(self.cat))))

        if class_choice is not None:
            self.cat = {k: v for k, v in self.cat.items() if k in class_choice}

        # 2. 讀取訓練/測試文件列表
        self.meta = {}
        # 請確認您的 json 路徑是否正確
        train_path = os.path.join(self.root, 'train_test_split', 'shufflenet_train_FileList.json')
        test_path = os.path.join(self.root, 'train_test_split', 'shufflenet_test_FileList.json')
        
        with open(train_path, 'r') as f:
            train_ids = set(json.load(f))
        with open(test_path, 'r') as f:
            test_ids = set(json.load(f))
            
        for item in self.cat:
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item])
            if not os.path.exists(dir_point):
                continue
            fns = sorted(os.listdir(dir_point))
            
            if split == 'trainval':
                fns = [fn for fn in fns if ((item + '/' + fn[0:-4]) in train_ids)]
            elif split == 'train':
                fns = [fn for fn in fns if (item + '/' + fn[0:-4]) in train_ids]
            elif split == 'test':
                fns = [fn for fn in fns if (item + '/' + fn[0:-4]) in test_ids]
            else:
                exit(-1)

            for fn in fns:
                token = (os.path.splitext(os.path.basename(fn))[0])
                self.meta[item].append(os.path.join(dir_point, token + '.txt'))

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn))

        self.classes = {i: self.classes_original[i] for i in self.cat.keys()}

        # 3. 設定 Seg Classes
        self.seg_classes = {}
        for cat_name in self.cat.keys():
            self.seg_classes[cat_name] = [0, 1]
        
        print(f"Loaded {len(self.datapath)} clouds. Grid: {self.grid_num}^3. Augment: {self.augment}")

        self.cache = {}  
        self.cache_size = 20000

    def __getitem__(self, index):
        if index in self.cache:
            point_set, cls, seg = self.cache[index]
        else:
            fn = self.datapath[index]
            cat = self.datapath[index][0]
            cls = np.array([self.classes[cat]]).astype(np.int32)
            data = np.loadtxt(fn[1]).astype(np.float32)
            point_set = data[:, 0:6] if self.normal_channel else data[:, 0:3]
            
            # 讀取標籤時轉為 float32
            seg = data[:, -1].astype(np.float32)
            
            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls, seg)
        
        # 1. 點雲歸一化 (暫時註解，保持你原本的設定)
        #point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

        # 2. 數據增強
        if self.augment:
            # A. 既有的旋轉邏輯 (保持不變)
            angle = np.random.uniform(-180, 180) * (np.pi / 180.0)
            cos_val, sin_val = np.cos(angle), np.sin(angle)
            rot_matrix = np.array([[cos_val, -sin_val, 0],
                                   [sin_val,  cos_val, 0],
                                   [0,        0,       1]])
            point_set[:, 0:3] = np.dot(point_set[:, 0:3], rot_matrix)
            if self.normal_channel:
                point_set[:, 3:6] = np.dot(point_set[:, 3:6], rot_matrix)
            
            # B. [新增] Jitter (隨機噪聲)
            # sigma=0.005 代表噪聲強度，clip=0.02 限制最大偏差
            point_set[:, 0:3] = jitter_point_cloud(point_set[:, 0:3], sigma=0.005, clip=0.02)

        # 3. --- 生成軟性格子標籤 (Soft Grid Labels) ---
        xyz = point_set[:, 0:3]
        min_bound = np.min(xyz, axis=0)
        max_bound = np.max(xyz, axis=0)
        denom = (max_bound - min_bound) + 1e-8
        norm_xyz = (xyz - min_bound) / denom
        
        # 映射到 grid 座標
        grid_coords = (norm_xyz * (self.grid_num - 1e-5)).astype(np.int32)
        
        # 將三維座標轉為一維索引
        grid_idx = grid_coords[:, 0] * (self.grid_num**2) + \
                   grid_coords[:, 1] * (self.grid_num) + \
                   grid_coords[:, 2]
        
        # 使用 Max Pooling 生成 Grid Label
        grid_labels = np.zeros(self.grid_num**3, dtype=np.float32)
        np.maximum.at(grid_labels, grid_idx, seg)

        # 4. 採樣到固定點數
        choice = np.random.choice(len(seg), self.npoints, replace=True)
        point_set = point_set[choice, :]
        seg = seg[choice]

        return point_set, cls, seg, grid_labels

    def __len__(self):
        return len(self.datapath)