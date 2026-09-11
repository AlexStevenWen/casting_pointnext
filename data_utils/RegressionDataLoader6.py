"""
用於 PointNet 姿態回歸任務的資料載入器。
修改自 RegressionDataLoader.py.py。
"""
import os
import numpy as np
import warnings
import pickle
import h5py
import pandas as pd
import torch

from tqdm import tqdm
from torch.utils.data import Dataset

warnings.filterwarnings('ignore')


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,), dtype=np.int32) # <--- 修改: 確保 dtype 為 int
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids] # <--- 修改: 簡化索引
    return point


def _read_point_cloud_from_h5(file_path):
    """ (新) 從 H5 檔案讀取點雲 """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到 H5 檔案: {file_path}")
    with h5py.File(file_path, 'r') as f:
        if 'data' in f:
            return f['data'][:]
        elif 'point_cloud' in f:
            return f['point_cloud'][:]
        else:
            raise ValueError(f"在 {file_path} 中找不到 'data' 或 'point_cloud' 鍵")


class RegressionDataLoader(Dataset):
    def __init__(self, root, args, split='train', process_data=False):
        self.root = root
        self.npoints = args.num_point
        self.process_data = process_data
        self.uniform = args.use_uniform_sample
        # self.use_normals = args.use_normals # 在這個載入器中，法向量被假定已移除
        
        # 1. (新) 定義 H5 點雲和 CSV 標籤的路徑
        self.pc_path = os.path.join(self.root, 'point_clouds')
        
        # 假設您已將 'labels.csv' 分割為 'train_labels.csv' 和 'test_labels.csv'
        label_csv_path = os.path.join(self.root, f'{split}_labels.csv')
        
        if not os.path.exists(label_csv_path):
            if split == 'train':
                 # 如果找不到 train_labels.csv，嘗試載入 'labels.csv' 作為備案
                 label_csv_path = os.path.join(self.root, 'labels.csv')
                 print(f"警告: 找不到 'train_labels.csv'。將使用 'labels.csv'。")
            else:
                 print(f"錯誤: 找不到 '{split}_labels.csv'。請確保測試標籤檔案存在。")
                 raise FileNotFoundError(label_csv_path)

        # 2. (新) 使用 Pandas 讀取 CSV
        try:
            self.data_frame = pd.read_csv(label_csv_path)
        except Exception as e:
            print(f"讀取 {label_csv_path} 時發生錯誤: {e}")
            raise

        # 3. (新) 獲取標籤欄位 (CSV 中除了 'filename' 以外的所有欄位)
        self.label_columns = [col for col in self.data_frame.columns if col != 'filename']
        
        # (可選) 檢查 args.output_dim 是否與 CSV 匹配
        if hasattr(args, 'output_dim') and args.output_dim != len(self.label_columns):
            print(f"警告: 模型的 output_dim ({args.output_dim}) 與 CSV 中的標籤數 ({len(self.label_columns)}) 不符。")

        print(f'The size of {split} data is {len(self.data_frame)}. Labels: {self.label_columns}')

        # 4. (新) 適應 process_data (預處理緩存)
        if self.process_data:
            dat_filename = f'regression_{split}_{self.npoints}pts'
            if self.uniform:
                dat_filename += '_fps.dat'
            else:
                dat_filename += '.dat'
            self.save_path = os.path.join(root, dat_filename)

            if not os.path.exists(self.save_path):
                print('Processing data %s (only running in the first time)...' % self.save_path)
                self.list_of_points = [None] * len(self.data_frame)
                self.list_of_labels = [None] * len(self.data_frame)

                for index in tqdm(range(len(self.data_frame)), total=len(self.data_frame)):
                    # 呼叫 _get_item_from_disk 來載入和處理單個項目
                    point_set, label = self._get_item_from_disk(index)
                    self.list_of_points[index] = point_set
                    self.list_of_labels[index] = label

                with open(self.save_path, 'wb') as f:
                    pickle.dump([self.list_of_points, self.list_of_labels], f)
            else:
                print('Load processed data from %s...' % self.save_path)
                with open(self.save_path, 'rb') as f:
                    self.list_of_points, self.list_of_labels = pickle.load(f)

    def __len__(self):
        return len(self.data_frame)

    def _get_item_from_disk(self, index):
        """ (新) 從磁碟實際讀取 H5 和 CSV 數據 """
        # 1. 從 DataFrame 獲取檔名和標籤
        row = self.data_frame.iloc[index]
        filename = row['filename']
        
        # .values 將其轉換為 numpy 陣列, .astype 確保是 float32
        label = row[self.label_columns].values.astype(np.float32)
        # 檢查標籤欄位是否包含 'rx' (更安全的作法)
        if 'rx' in self.label_columns:
            # 找到 'rx', 'ry', 'rz' 在 label_columns 中的索引
            rx_idx = self.label_columns.index('rx')
            ry_idx = self.label_columns.index('ry')
            rz_idx = self.label_columns.index('rz')
            
            # 只對旋轉角度進行歸一化
            label[rx_idx] = label[rx_idx] / np.pi
            label[ry_idx] = label[ry_idx] / np.pi
            label[rz_idx] = label[rz_idx] / np.pi
        # 2. 讀取 H5 檔案
        pc_file_path = os.path.join(self.pc_path, filename)
        try:
            point_set = _read_point_cloud_from_h5(pc_file_path)
        except Exception as e:
            print(f"\n錯誤: 載入 {pc_file_path} (索引 {index}) 失敗: {e}")
            # 返回一個虛擬資料，以防訓練中斷
            point_set = np.zeros((self.npoints, 3), dtype=np.float32)
            label = np.zeros(len(self.label_columns), dtype=np.float32)
            return point_set, label

        # 3. 採樣點雲 (如果 H5 點數與 npoints 不符)
        if self.uniform:
            point_set = farthest_point_sample(point_set, self.npoints)
        else:
            # 如果點太多，隨機選
            if point_set.shape[0] > self.npoints:
                choice = np.random.choice(point_set.shape[0], self.npoints, replace=False)
                point_set = point_set[choice, :]
            # 如果點不夠，重複選
            elif point_set.shape[0] < self.npoints:
                choice = np.random.choice(point_set.shape[0], self.npoints, replace=True)
                point_set = point_set[choice, :]
        
        # 4. 確保只有 XYZ (如果原始資料有法向量)
        if point_set.shape[1] > 3:
            point_set = point_set[:, 0:3]

        # 5. !!! 關鍵：移除 pc_normalize(point_set) !!!
        # 資料在生成時已經標準化過了。
        
        return point_set, label

    def __getitem__(self, index):
        if self.process_data:
            point_set, label = self.list_of_points[index], self.list_of_labels[index]
        else:
            point_set, label = self._get_item_from_disk(index)
        
        # 返回 numpy 陣列，訓練腳本 (train_regression.py) 會將它們轉換為 Tensor
        return point_set, label


if __name__ == '__main__':
    import torch
    
    print("--- 回歸資料載入器測試 ---")
    
    # --- (新) 測試設定 ---
    # 1. 建立一個虛擬的 'args' 物件
    class DummyArgs:
        num_point = 1024
        use_uniform_sample = False
        use_normals = False
        output_dim = 6  # 假設我們在測試 6D 回歸
    
    args = DummyArgs()

    # 2. (重要) 您需要手動建立一個測試資料夾
    # 範例結構:
    # C:\temp_dataset\
    # ├── point_clouds\
    # │   ├── test_file_001.h5
    # │   └── test_file_002.h5
    # └── train_labels.csv
    
    # 3. (重要) 您需要手動建立一個 train_labels.csv
    # 檔案內容 (範例):
    # filename,tx,ty,tz,rx,ry,rz
    # test_file_001.h5,0.0,0.0,0.0,0.1,0.2,0.3
    # test_file_002.h5,0.0,0.0,0.0,-0.1,0.5,-0.2
    
    # 4. (重要) 您需要手動建立虛擬的 .h5 檔案
    # (您可以使用 `ModelNetConverter` 生成真實的檔案來測試)

    try:
        # 將 'D:/your_regression_dataset/' 替換為您生成的資料集路徑
        DATA_PATH = 'D:/your_regression_dataset/' 
        
        data = RegressionDataLoader(DATA_PATH, args=args, split='train')
        DataLoader = torch.utils.data.DataLoader(data, batch_size=4, shuffle=True)
        
        # 取一個 BATCH 的資料
        point, label = next(iter(DataLoader))
        
        print(f"\n成功載入一個 Batch:")
        print(f"  Point shape (Batch_Size, Num_Points, Channels): {point.shape}")
        print(f"  Label shape (Batch_Size, Output_Dim): {label.shape}")
        print(f"  Point dtype: {point.dtype}")
        print(f"  Label dtype: {label.dtype}")
        print(f"\n  範例標籤 (第一筆): {label[0]}")
        
    except FileNotFoundError:
        print(f"\n--- 測試失敗 ---")
        print(f"請確認 'DATA_PATH' 變數設定正確，")
        print(f"且該路徑包含 'point_clouds' 資料夾和 'train_labels.csv' 檔案。")
    except Exception as e:
        print(f"\n--- 測試失敗 ---")
        print(f"發生錯誤: {e}")
        print("請檢查您的 H5 檔案是否損壞，或 CSV 格式是否正確。")