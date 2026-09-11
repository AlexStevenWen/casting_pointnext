import torch
import torch.nn as nn
from einops import repeat, rearrange

def exists(val):
    return val is not None

# --- 純 Pytorch 算子實現 ---
def farthest_point_sample_pt(xyz, npoint):
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def ball_query_pt(radius, k, src, query):
    """
    完全修復維度不匹配問題的 Ball Query
    src: [B, N, 3], query: [B, M, 3]
    """
    device = src.device
    b, n, _ = src.size()
    m = query.size(1)
    
    # 1. 計算歐式距離
    # 使用 torch.cdist 得到 [B, M, N]
    dists = torch.cdist(query, src) 
    
    # 2. 初始化索引 [B, M, N]
    idx = torch.arange(n, device=device).view(1, 1, n).expand(b, m, n).contiguous()
    
    # 3. 超出半徑的標記為 n (非法索引)
    idx[dists > radius] = n
    
    # 4. 排序並取前 k 個 [B, M, K]
    # 注意：這裡的 k 是函數傳入的參數
    idx = idx.sort(dim=-1)[0][:, :, :k]
    
    # 5. 處理無效點：用該組的第一個點索引填充
    # 取得實際拿到的點數 (實質上就是 k，但從張量拿最安全)
    actual_k = idx.shape[-1]
    
    # 取出每組的第一個索引 [B, M, 1]
    group_first = idx[:, :, [0]]
    
    # 建立遮罩
    mask = (idx == n)
    
    # 關鍵修正：直接使用 group_first 進行自動廣播填充
    # 如果自動廣播失敗，手動擴張至 [B, M, actual_k]
    idx = torch.where(mask, group_first.expand(-1, -1, actual_k), idx)
    
    return idx

# --- 嘗試載入 C++ 擴展 ---
try:
    from pathlib import Path
    torch.ops.load_library(Path(__file__).parent / '_C.so')
    _C = torch.ops.my_ops
    def fps_op(xyz, n): return _C.furthest_point_sampling_wrapper(xyz.shape[0], xyz.shape[1], n, xyz.contiguous(), torch.full((xyz.shape[0], xyz.shape[1]), 1e10, device=xyz.device), torch.zeros(xyz.shape[0], n, dtype=torch.int32, device=xyz.device)).long()
    def ball_op(r, k, s, q): 
        idx = torch.zeros(q.shape[0], q.shape[1], k, dtype=torch.int32, device=s.device)
        _C.ball_query_wrapper(s.shape[0], s.shape[1], q.shape[1], r, k, q.contiguous(), s.contiguous(), idx)
        return idx.long()
except:
    fps_op = farthest_point_sample_pt
    ball_op = ball_query_pt

def gather(x, idx):
    b, d, n = x.shape
    m = idx.shape[1]
    k = idx.shape[2]
    idx = idx.view(b, 1, m * k).repeat(1, d, 1)
    return x.gather(2, idx).view(b, d, m, k)