import os
import numpy as np
import warnings
import h5py
import torch
from torch.utils.data import Dataset

warnings.filterwarnings('ignore')

# (這是您 augment_pvn_data.py 中的 FPS 函數，我們需要它)
def farthest_point_sample(point, npoint):
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,), dtype=np.int32)
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids]
    return point

def _read_point_cloud_from_h5(file_path):
    with h5py.File(file_path, 'r') as f:
        if 'point_cloud' in f: 
            return f['point_cloud'][:]
        elif 'data' in f: 
            return f['data'][:]
        else: 
            raise ValueError(f"H5 {file_path} 中找不到 'point_cloud' 或 'data'")

class PVN3DDataLoader(Dataset):
    def __init__(self, root, args, split='train'):
        self.root = root
        self.npoints = args.num_point

        self.pc_path = os.path.join(self.root, 'point_clouds')
        self.kp_path = os.path.join(self.root, 'keypoint_labels')

        # 讀取 train_files.txt 或 test_files.txt
        filelist_path = os.path.join(self.root, f'{split}_files.txt')
        try:
            self.file_list = [line.rstrip() for line in open(filelist_path)]
        except FileNotFoundError:
            print(f"錯誤: 找不到 {filelist_path}")
            print("請先執行 split_pvn_data.py 來生成檔案列表。")
            raise

        print(f'The size of {split} data is {len(self.file_list)}')

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        # 1. 獲取檔名
        base_filename = self.file_list[index]

        # 2. 定義路徑
        pc_file_path = os.path.join(self.pc_path, f"{base_filename}.h5")
        kp_file_path = os.path.join(self.kp_path, f"{base_filename}.txt")

        # 3. 讀取資料
        try:
            point_set_raw = _read_point_cloud_from_h5(pc_file_path)
            keypoints_raw = np.loadtxt(kp_file_path, dtype=np.float32)
        except Exception as e:
            print(f"錯誤: 載入資料 {base_filename} 失敗: {e}")
            # 返回虛擬資料以防訓練中斷
            point_set_raw = np.zeros((self.npoints, 3), dtype=np.float32)
            keypoints_raw = np.zeros((8, 3), dtype=np.float32) # PVN3D 需要 8 個點

        # 4. 採樣點雲
        if point_set_raw.shape[0] > self.npoints:
            point_set = farthest_point_sample(point_set_raw, self.npoints)
        elif point_set_raw.shape[0] < self.npoints:
            # 點不夠，重複選
            choice = np.random.choice(point_set_raw.shape[0], self.npoints, replace=True)
            point_set = point_set_raw[choice, :]
        else:
            point_set = point_set_raw

        # 確保只有 XYZ
        if point_set.shape[1] > 3:
            point_set = point_set[:, 0:3]

        # 5. 返回 Numpy 陣列
        # 訓練腳本會將它們轉換為 Tensor
        return point_set.astype(np.float32), keypoints_raw.astype(np.float32)