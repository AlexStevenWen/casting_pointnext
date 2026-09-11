# *_*coding:utf-8 *_*
import os
import json
import warnings
import numpy as np
from torch.utils.data import Dataset
warnings.filterwarnings('ignore')

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc

class PartNormalDataset(Dataset):
    def __init__(self, root='./data/shapenetcore_partanno_segmentation_benchmark_v0_normal', 
                 npoints=2500, split='train', class_choice=None, normal_channel=False,
                 augment=False): # 新增 augment 參數
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}
        self.normal_channel = normal_channel
        self.augment = augment # 是否開啟數據增強

        # 1. 讀取類別對應表
        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        
        self.classes_original = dict(zip(self.cat, range(len(self.cat))))

        if not class_choice is None:
            self.cat = {k: v for k, v in self.cat.items() if k in class_choice}

        # 2. 讀取訓練/測試文件列表
        self.meta = {}
        with open(os.path.join(self.root, 'train_test_split', 'shufflenet_train_FileList.json'), 'r') as f:
            train_ids = set(json.load(f))
        val_ids = set() 
        with open(os.path.join(self.root, 'train_test_split', 'shufflenet_test_FileList.json'), 'r') as f:
            test_ids = set(json.load(f))
            
        for item in self.cat:
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item])
            if not os.path.exists(dir_point):
                continue
            fns = sorted(os.listdir(dir_point))
            
            if split == 'trainval':
                fns = [fn for fn in fns if ((item + '/' + fn[0:-4]) in train_ids) or ((item + '/' + fn[0:-4]) in val_ids)]
            elif split == 'train':
                fns = [fn for fn in fns if (item + '/' + fn[0:-4]) in train_ids]
            elif split == 'val':
                fns = [fn for fn in fns if (item + '/' + fn[0:-4]) in val_ids]
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
        
        print(f"Loaded {len(self.datapath)} clouds. Augment: {self.augment}. Seg Classes: {self.seg_classes}")

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
        
        # 1. 常規歸一化
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

        # 2. --- 隨機旋轉增強 (Z軸 ±10度) ---
        if self.augment:
            # 隨機產生 -10 到 +10 度的弧度
            angle = np.random.uniform(-180, 180) * (np.pi / 180.0)
            cos_val = np.cos(angle)
            sin_val = np.sin(angle)
            
            # Z軸旋轉矩陣
            rot_matrix = np.array([[cos_val, -sin_val, 0],
                                   [sin_val,  cos_val, 0],
                                   [0,        0,       1]])
            
            # 僅旋轉座標 XYZ (point_set 的前 3 列)
            point_set[:, 0:3] = np.dot(point_set[:, 0:3], rot_matrix)
            
            # 如果包含法向量 (Normal)，法向量也需要隨之旋轉
            if self.normal_channel:
                point_set[:, 3:6] = np.dot(point_set[:, 3:6], rot_matrix)

        # 3. 採樣
        choice = np.random.choice(len(seg), self.npoints, replace=True)
        point_set = point_set[choice, :]
        seg = seg[choice]

        return point_set, cls, seg

    def __len__(self):
        return len(self.datapath)