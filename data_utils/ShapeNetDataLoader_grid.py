# *_*coding:utf-8 *_*
"""
Author: Benny (Modified for Voxel Grid Multi-task Learning)
Date: 2025 Revised
Description: 將點雲空間等分為 nxnxn 個格子，並生成每個格子是否含有澆口的二元標籤。
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
                 augment=False, grid_num=3):
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}
        self.normal_channel = normal_channel
        self.augment = augment 
        self.grid_num = grid_num # 預設 3x3x3 = 27 格

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
        # 這裡請確保路徑與您的 json 檔名一致
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

        # 3. 自動計算 seg_classes
        self.seg_classes = {}
        for cat_name in self.cat.keys():
            all_seg_labels = set()
            check_fns = self.meta[cat_name][:20]
            for fn in check_fns:
                try:
                    l_data = np.loadtxt(fn).astype(np.float32)
                    l_segs = l_data[:, -1].astype(np.int32)
                    all_seg_labels.update(np.unique(l_segs).tolist())
                except: continue
            self.seg_classes[cat_name] = sorted(list(all_seg_labels)) if all_seg_labels else [0, 1]
        
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
            seg = data[:, -1].astype(np.int32)
            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls, seg)
        
        # 1. 點雲歸一化
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

        # 2. 數據增強 (Z軸 360度全旋轉)
        if self.augment:
            angle = np.random.uniform(-180, 180) * (np.pi / 180.0)
            cos_val, sin_val = np.cos(angle), np.sin(angle)
            rot_matrix = np.array([[cos_val, -sin_val, 0],
                                   [sin_val,  cos_val, 0],
                                   [0,        0,       1]])
            point_set[:, 0:3] = np.dot(point_set[:, 0:3], rot_matrix)
            if self.normal_channel:
                point_set[:, 3:6] = np.dot(point_set[:, 3:6], rot_matrix)

        # 3. --- 神奇辦法：計算格子標籤 ---
        # 獲取 XYZ 並平移到 [0, 1] 區間以進行分箱
        xyz = point_set[:, 0:3]
        min_bound = np.min(xyz, axis=0)
        max_bound = np.max(xyz, axis=0)
        # 避免除以零
        denom = (max_bound - min_bound) + 1e-8
        norm_xyz = (xyz - min_bound) / denom
        
        # 映射到 grid 座標 (0, 1, 2)
        grid_coords = (norm_xyz * (self.grid_num - 1e-5)).astype(np.int32)
        
        # 將三維座標轉為一維索引 (0 到 26)
        # 索引公式: x*n^2 + y*n + z
        grid_idx = grid_coords[:, 0] * (self.grid_num**2) + \
                   grid_coords[:, 1] * (self.grid_num) + \
                   grid_coords[:, 2]
        
        # 生成 Grid Binary Label: 只要格子內有標籤 1 (Gate)，該格即為 1
        grid_labels = np.zeros(self.grid_num**3, dtype=np.float32)
        gate_points_mask = (seg == 1)
        if np.any(gate_points_mask):
            # 找出所有含有澆口點的格子索引
            active_grids = np.unique(grid_idx[gate_points_mask])
            grid_labels[active_grids] = 1.0

        # 4. 採樣到固定點數 (如 4096)
        choice = np.random.choice(len(seg), self.npoints, replace=True)
        point_set = point_set[choice, :]
        seg = seg[choice]
        # 如果需要點對應的格子 ID 也可以一併返回
        # point_grid_idx = grid_idx[choice]

        return point_set, cls, seg, grid_labels

    def __len__(self):
        return len(self.datapath)