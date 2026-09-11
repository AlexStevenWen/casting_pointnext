# *_*coding:utf-8 *_*
"""
Author: Benny (Modified for Soft Label / Probability Distribution)
Date: 2025 Revised
Description: 
    針對概率分佈標籤 (0.0~1.0) 的資料集讀取器。
    1. 點雲標籤 (seg) 會保持 float32 格式。
    2. Grid 標籤會計算該格內所有點的「最大概率值」。
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

class PartNormalDataset(Dataset):
    def __init__(self, root='./data/shapenetcore_partanno_segmentation_benchmark_v0_normal', 
                 npoints=2500, split='train', class_choice=None, normal_channel=False,
                 augment=False, grid_num=4): # 建議 grid_num 設為 4 或 8
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

        # 3. [修改] 設定 Seg Classes
        # 因為現在標籤是連續機率 (Float)，不能用 unique 來找類別。
        # 我們直接假定這是一個二元分割任務 (背景 vs 澆口)，所以類別 ID 為 [0, 1]
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
            
            # [關鍵修改 1] 讀取標籤時轉為 float32，保留概率值 (0.0 ~ 1.0)
            seg = data[:, -1].astype(np.float32)
            
            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls, seg)
        
        # 1. 點雲歸一化
        #point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

        # 2. 數據增強 (Z軸 360度全旋轉)
        # 注意：旋轉只改變座標，不會改變點的「概率標籤」，所以 seg 不用動
        if self.augment:
            angle = np.random.uniform(-180, 180) * (np.pi / 180.0)
            cos_val, sin_val = np.cos(angle), np.sin(angle)
            rot_matrix = np.array([[cos_val, -sin_val, 0],
                                   [sin_val,  cos_val, 0],
                                   [0,        0,       1]])
            point_set[:, 0:3] = np.dot(point_set[:, 0:3], rot_matrix)
            if self.normal_channel:
                point_set[:, 3:6] = np.dot(point_set[:, 3:6], rot_matrix)

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
        
        # [關鍵修改 2] 使用 Max Pooling 生成 Grid Label
        # 邏輯：一個格子的標籤 = 該格子內部所有點的最大概率值
        # 這樣如果格子裡有澆口點(0.9)，格子就是 0.9；如果只有邊緣點(0.2)，格子就是 0.2
        
        grid_labels = np.zeros(self.grid_num**3, dtype=np.float32)
        
        # 使用 numpy 的 ufunc.at 來進行快速的最大值聚合
        # 這行代碼的意思是：對於 grid_idx 裡的每個位置，用 seg 的值去更新 grid_labels，取最大值
        np.maximum.at(grid_labels, grid_idx, seg)

        # 4. 採樣到固定點數
        # 注意：seg 現在是 float，這沒問題
        choice = np.random.choice(len(seg), self.npoints, replace=True)
        point_set = point_set[choice, :]
        seg = seg[choice]

        return point_set, cls, seg, grid_labels

    def __len__(self):
        return len(self.datapath)