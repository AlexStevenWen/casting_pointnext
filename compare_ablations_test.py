import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 設定 Matplotlib 風格
plt.style.use('seaborn-v0_8-whitegrid')

# ==========================================
# 1. 實驗設定區 (在此填入你的 CSV 路徑與自訂代號)
# ==========================================


EXPERIMENTS = {
    # --- Binary Classification (8192 points) ---
    "random_binary_8192": "./log/part_seg/exp_pointnext_part_seg_nocls_ssg_invertedresidualmlp_nostn_seg_random_binary_8192_2026-02-28_20-28-15/training_metrics_history.csv",
    "fps_binary_8192": "./log/part_seg/exp_pointnext_part_seg_nocls_ssg_invertedresidualmlp_nostn_seg_fps_binary_8192_2026-03-01_00-07-57/training_metrics_history.csv",

    # --- Gaussian Regression ---
    "random_global_gaussian_regress_8192": "./log/part_seg/exp_pointnext_part_seg_nocls_ssg_invertedresidualmlp_nostn_regress_random_global_gaussian_2026-03-05_19-26-07/training_metrics_history.csv",
    "fps_global_gaussian_regress_8192": "./log/part_seg/exp_pointnext_part_seg_nocls_ssg_invertedresidualmlp_nostn_regress_fps_global_gaussian_8192_2026-03-05_15-11-51/training_metrics_history.csv",
    
    # --- Flat Core Gaussian ---
    "fps_global_flat_core_gaussian": "./log/part_seg/exp_pointnext_part_seg_nocls_ssg_invertedresidualmlp_nostn_regress_fps_global_flat_core_gaussian_2026-03-07_17-05-12/training_metrics_history.csv",
    "random_global_flat_core_gaussian": "./log/part_seg/exp_pointnext_part_seg_nocls_ssg_invertedresidualmlp_nostn_regress_random_global_flat_core_gaussian_2026-03-07_18-01-56/training_metrics_history.csv",

    # --- Linear Regression ---
    "fps_global_linear": "./log/part_seg/exp_pointnext_part_seg_nocls_ssg_invertedresidualmlp_nostn_regress_fps_global_linear_2026-03-07_15-48-54/training_metrics_history.csv",
    "random_global_linear": "./log/part_seg/exp_pointnext_part_seg_nocls_ssg_invertedresidualmlp_nostn_regress_random_global_linear_2026-03-07_14-47-28/training_metrics_history.csv",

    # --- Sigmoid Regression ---
    "fps_global_sigmoid": "./log/part_seg/exp_pointnext_part_seg_nocls_ssg_invertedresidualmlp_nostn_regress_fps_global_sigmoid_2026-03-10_19-18-41/training_metrics_history.csv",
    "random_global_sigmoid": "./log/part_seg/exp_pointnext_part_seg_nocls_ssg_invertedresidualmlp_nostn_regress_random_global_sigmoid_2026-03-09_13-05-51/training_metrics_history.csv",
}
OUTPUT_DIR = "./ablation_test"
SMOOTH_WINDOW = 5  # 平滑化的視窗大小 (例如 5 個 Epoch 取平均)
LAST_K_EPOCHS = 20 # 計算末期穩定度的 Epoch 數量

# ==========================================
# 工具函數：繪製平滑對比圖
# ==========================================
def plot_comparison(data_dict, metric_key, title, ylabel, filename, use_smoothing=True):
    plt.figure(figsize=(10, 6))
    colors = plt.cm.tab10.colors
    
    for idx, (name, df) in enumerate(data_dict.items()):
        if metric_key not in df.columns:
            print(f"警告: {name} 中找不到指標 '{metric_key}'，已跳過。")
            continue
            
        epochs = df['Epoch'] if 'Epoch' in df.columns else df.index
        raw_values = df[metric_key]
        color = colors[idx % len(colors)]
        
        # 畫出原始數據 (半透明，顯示真實的震盪)
        plt.plot(epochs, raw_values, color=color, alpha=0.2, label=f"{name} (Raw)")
        
        # 畫出平滑曲線 (實線，顯示整體趨勢)
        if use_smoothing:
            smoothed = raw_values.rolling(window=SMOOTH_WINDOW, min_periods=1).mean()
            plt.plot(epochs, smoothed, color=color, linewidth=2, label=f"{name} (Smoothed)")
        else:
            plt.plot(epochs, raw_values, color=color, linewidth=2, label=name)

    plt.title(title, fontsize=14, pad=15)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=200)
    plt.close()

# ==========================================
# 主程式
# ==========================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    data_dict = {}
    stats_list = []
    
    # 讀取資料與計算統計量
    print("載入實驗數據...")
    for name, path in EXPERIMENTS.items():
        if not os.path.exists(path):
            print(f"找不到檔案，請確認路徑: {path}")
            continue
            
        df = pd.read_csv(path)
        data_dict[name] = df
        
        # 計算統計量 (處理 IoU 震盪的關鍵)
        stats = {'Model': name}
        if 'test_iou' in df.columns:
            stats['Best_mIoU(%)'] = df['test_iou'].max() * 100
            stats[f'Top5_Avg_mIoU(%)'] = df['test_iou'].nlargest(5).mean() * 100
            stats[f'Last{LAST_K_EPOCHS}_Avg_mIoU(%)'] = df['test_iou'].tail(LAST_K_EPOCHS).mean() * 100
            stats[f'Last{LAST_K_EPOCHS}_Std_mIoU(%)'] = df['test_iou'].tail(LAST_K_EPOCHS).std() * 100 # 看誰比較穩定
            
        if 'shift_err' in df.columns:
            stats['Best_ShiftErr'] = df['shift_err'].min()
            stats[f'Last{LAST_K_EPOCHS}_Avg_ShiftErr'] = df['shift_err'].tail(LAST_K_EPOCHS).mean()
            
        stats_list.append(stats)

    if not data_dict:
        print("沒有成功載入任何資料，請檢查 EXPERIMENTS 設定。")
        return

    # 1. 產出對比圖表
    print("正在繪製對比圖表...")
    plot_comparison(data_dict, 'test_iou', 'mIoU Comparison (Foreground Overlap)', 'mIoU', 'compare_mIoU.png')
    plot_comparison(data_dict, 'shift_err', 'Spatial Shift Error Comparison', 'Distance Error', 'compare_shift_err.png')
    plot_comparison(data_dict, 'success_rate', 'Success Rate Comparison', 'Rate', 'compare_success_rate.png', use_smoothing=False)
    plot_comparison(data_dict, 'test_loss', 'Validation Loss Comparison', 'Loss', 'compare_loss.png')

    # 2. 產出科學統計表格
    print("\n================ 消融實驗統計結果 ================")
    stats_df = pd.DataFrame(stats_list).round(4)
    print(stats_df.to_string(index=False))
    
    stats_csv_path = os.path.join(OUTPUT_DIR, 'ablation_summary_table.csv')
    stats_df.to_csv(stats_csv_path, index=False, encoding='utf-8-sig')
    print("==================================================")
    print(f"\n所有圖表與統計結果已儲存至資料夾: {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()