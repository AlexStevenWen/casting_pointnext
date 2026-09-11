'''
@author: Xu Yan
@file: ModelNet.py
@time: 2021/3/19 15:51
Modified for Keypoint/Segmentation (4th dimension)
'''
import os
import numpy as np
import warnings
import pickle
from tqdm import tqdm
from torch.utils.data import Dataset

warnings.filterwarnings('ignore')

def pc_normalize(pc):
    """ 只針對空間坐標進行歸一化 """
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D] (D 可能包含 XYZ + 特徵)
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:, :3] # 採樣邏輯固定使用幾何坐標
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    # 返回完整 D 維度的採樣點
    point = point[centroids.astype(np.int32)]
    return point

class ModelNetDataLoader(Dataset):
    def __init__(self, root, args, split='train', process_data=False):
        self.root = root
        self.npoints = args.num_point
        self.process_data = process_data
        self.uniform = args.use_uniform_sample
        self.use_normals = args.use_normals
        self.num_category = args.num_category

        # 類別檔案處理邏輯
        if self.num_category == 10:
            self.catfile = os.path.join(self.root, 'modelnet10_shape_names.txt')
        elif self.num_category == 40:
            self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')
        else:
            self.catfile = os.path.join(self.root, 'shape_names.txt')
        
        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))

        shape_ids = {}
        if self.num_category == 10:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_test.txt'))]
        elif self.num_category == 40:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test.txt'))]
        else:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'train_files.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'test_files.txt'))]
        
        assert (split == 'train' or split == 'test')
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt') for i
                         in range(len(shape_ids[split]))]
        print('The size of %s data is %d' % (split, len(self.datapath)))

        # 緩存路徑，建議在檔名加上 '4ch' 以區分原始數據
        fps_str = '_fps' if self.uniform else ''
        self.save_path = os.path.join(root, f'modelnet{self.num_category}_{split}_{self.npoints}pts{fps_str}_4ch.dat')

        if self.process_data:
            if not os.path.exists(self.save_path):
                print('Processing data %s (only running in the first time)...' % self.save_path)
                self.list_of_points = [None] * len(self.datapath)
                self.list_of_labels = [None] * len(self.datapath)

                for index in tqdm(range(len(self.datapath)), total=len(self.datapath)):
                    fn = self.datapath[index]
                    cls = self.classes[self.datapath[index][0]]
                    cls = np.array([cls]).astype(np.int32)
                    # 讀取數據：假設原始 txt 每行是 x,y,z,nx,ny,nz,seg_id (共 7 列)
                    point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)

                    if self.uniform:
                        point_set = farthest_point_sample(point_set, self.npoints)
                    else:
                        point_set = point_set[0:self.npoints, :]

                    self.list_of_points[index] = point_set
                    self.list_of_labels[index] = cls

                with open(self.save_path, 'wb') as f:
                    pickle.dump([self.list_of_points, self.list_of_labels], f)
            else:
                print('Load processed data from %s...' % self.save_path)
                with open(self.save_path, 'rb') as f:
                    self.list_of_points, self.list_of_labels = pickle.load(f)

    def __len__(self):
        return len(self.datapath)

    def _get_item(self, index):
            if self.process_data:
                point_set, label = self.list_of_points[index], self.list_of_labels[index]
            else:
                fn = self.datapath[index]
                cls = self.classes[self.datapath[index][0]]
                label = np.array([cls]).astype(np.int32)
                # 讀取數據
                point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)

                if self.uniform:
                    point_set = farthest_point_sample(point_set, self.npoints)
                else:
                    point_set = point_set[0:self.npoints, :]
                    
            # 1. 歸一化前三維
            point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

            # 2. 這是關鍵修改：確保回傳 4 個通道
            if not self.use_normals:
                # 假設你的 txt 第 1~3 列是 XYZ，第 4 列是分割維度 (索引為 3)
                # 如果分割維度在第 7 列，請改成 point_set[:, 6:7]
                xyz = point_set[:, 0:3]
                seg_id = point_set[:, 3:4] # <--- 請根據你 txt 的實際列數調整這裡的索引
                point_set = np.concatenate([xyz, seg_id], axis=1) # 合併成 [N, 4]
            else:
                # 如果使用 Normals，則是 XYZ (3) + Normals (3) + Seg (1) = 7 通道
                # 確保這裡提取的範圍包含分割維度
                point_set = point_set[:, 0:7] 

            return point_set, label[0]

    def __getitem__(self, index):
        return self._get_item(index)