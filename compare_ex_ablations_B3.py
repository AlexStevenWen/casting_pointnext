import os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
# 設定 Matplotlib 風格
# 設定 Matplotlib 風格
def setup_fonts(times_path="/mnt/c/Windows/Fonts/times.ttf",
                kaiu_path="/mnt/c/Windows/Fonts/kaiu.ttf"):
    plt.style.use('seaborn-v0_8-whitegrid')

    names = []
    for p in (times_path, kaiu_path):
        if os.path.exists(p):
            fm.fontManager.addfont(p)
            names.append(fm.FontProperties(fname=p).get_name())
        else:
            print(f"[警告] 找不到字體檔: {p}")

    if not names:
        raise FileNotFoundError("兩個字體檔都不存在，請確認路徑。")

    mpl.rcParams['font.family'] = names + ['DejaVu Sans']
    mpl.rcParams['axes.unicode_minus'] = False

    ver = tuple(int(x) for x in matplotlib.__version__.split('.')[:2])
    if ver < (3, 6):
        print(f"[警告] matplotlib {matplotlib.__version__} 不支援逐字元字體備援，中文會變 □。")
        print("       建議: pip install -U 'matplotlib>=3.8' && rm -rf ~/.cache/matplotlib")
    else:
        print(f"字體掛載完成 (matplotlib {matplotlib.__version__}): {names}")

    mpl.rcParams['font.size'] = 24
    mpl.rcParams['axes.titlesize'] = 36
    mpl.rcParams['axes.labelsize'] = 30
    mpl.rcParams['xtick.labelsize'] = 28
    mpl.rcParams['ytick.labelsize'] = 28
    mpl.rcParams['legend.fontsize'] = 24


EXPERIMENTS = {
    # --- Binary Classification (8192 points) ---
    "msg_bottleneckmlp_nostn_seg_custom_fps_4096_binary":"./log/part_seg/exp_pointnext_part_seg_nocls_msg_bottleneckmlp_nostn_seg_custom_4096_2026-06-29_10-15-09/training_metrics_history.csv",
    "msg_bottleneckmlp_nostn_seg_custom_fps_4096_gaussian":"./log/part_seg/exp_pointnext_part_seg_nocls_msg_bottleneckmlp_nostn_regress_custom_gaussian_4096_2026-06-30_08-06-02/training_metrics_history.csv",
    "msg_bottleneckmlp_nostn_seg_custom_fps_4096_local_linear":"./log/part_seg/exp_pointnext_part_seg_nocls_msg_bottleneckmlp_nostn_regress_custom_local_linear_4096_2026-06-29_21-32-16/training_metrics_history.csv",
    "msg_bottleneckmlp_nostn_seg_custom_fps_4096_global_linear":"./log/part_seg/exp_pointnext_part_seg_nocls_msg_bottleneckmlp_nostn_regress_custom_global_linear_4096_2026-06-30_14-27-05/training_metrics_history.csv",
    "msg_bottleneckmlp_nostn_seg_custom_fps_4096_global_gaussian":"./log/part_seg/exp_pointnext_part_seg_nocls_msg_bottleneckmlp_nostn_regress_custom_global_gaussian_4096_2026-06-30_17-30-32/training_metrics_history.csv"
    
    
}
OUTPUT_DIR = "./ablation_test_B3"
SMOOTH_WINDOW = 5  # 平滑化的視窗大小 (例如 5 個 Epoch 取平均)
LAST_K_EPOCHS = 20 # 計算末期穩定度的 Epoch 數量
USE_HATCH_FILL = False
# ==========================================
# 工具函數：繪製平滑對比圖 (已修正圖例擠壓問題)
# ==========================================
def plot_comparison(data_dict, metric_key, title, ylabel, filename, use_smoothing=True):
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.tab10.colors
    
    valid_plot = False
    for idx, (name, df) in enumerate(data_dict.items()):
        if metric_key not in df.columns:
            continue
            
        valid_plot = True
        epochs = df['Epoch'] if 'Epoch' in df.columns else df.index
        raw_values = df[metric_key]
        color = colors[idx % len(colors)]
        
        if use_smoothing:
            ax.plot(epochs, raw_values, color=color, alpha=0.2)
            smoothed = raw_values.rolling(window=SMOOTH_WINDOW, min_periods=1).mean()
            ax.plot(epochs, smoothed, color=color, linewidth=2.5, label=name)
        else:
            ax.plot(epochs, raw_values, color=color, linewidth=2.5, label=name)

    if not valid_plot:
        plt.close()
        return

    ax.set_title(title, pad=20, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel(ylabel)
    
    # 【關鍵修改】將 ncol 改為 2 欄堆疊，並微調 fontsize 和 labelspacing
    ax.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', borderaxespad=0., 
              ncol=2, fontsize=14, labelspacing=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300, bbox_inches='tight')
    plt.close()

def plot_comparison_black(data_dict, metric_key, title, ylabel, filename, use_smoothing=True):
    # 【關鍵修改】等比例放大畫布，例如設為 (14, 11)，維持原本 10:8 左右的視覺比例
    fig, ax = plt.subplots(figsize=(14, 11))
    
    bw_styles = [
        {'color': 'black',   'ls': '-',  'marker': 'o', 'hatch': ''},      
        {'color': 'black',   'ls': '--', 'marker': 's', 'hatch': '//'},    
        {'color': '#444444', 'ls': ':',  'marker': '^', 'hatch': '\\\\'},  
        {'color': 'black',   'ls': '-.', 'marker': 'D', 'hatch': 'xx'},    
        {'color': '#666666', 'ls': '-',  'marker': 'x', 'hatch': '..'}     
    ]
    
    valid_plot = False
    for idx, (name, df) in enumerate(data_dict.items()):
        if metric_key not in df.columns:
            continue
            
        valid_plot = True
        epochs = df['Epoch'] if 'Epoch' in df.columns else df.index
        raw_values = df[metric_key]
        
        style = bw_styles[idx % len(bw_styles)]
        mark_every = max(1, len(epochs) // 10)
        
        if use_smoothing:
            ax.plot(epochs, raw_values, color='#DDDDDD', alpha=0.6, linestyle=style['ls'])
            smoothed = raw_values.rolling(window=SMOOTH_WINDOW, min_periods=1).mean()
            
            if USE_HATCH_FILL and style['hatch'] != '':
                ax.fill_between(epochs, smoothed.min(), smoothed, 
                                facecolor='none', edgecolor=style['color'], 
                                hatch=style['hatch'], alpha=0.3)
            
            ax.plot(epochs, smoothed, color=style['color'], linestyle=style['ls'], 
                    marker=style['marker'], markevery=mark_every, markersize=8,
                    linewidth=2.5, label=name)
        else:
            ax.plot(epochs, raw_values, color=style['color'], linestyle=style['ls'], 
                    marker=style['marker'], markevery=mark_every, markersize=8,
                    linewidth=2.5, label=name)

    if not valid_plot:
        plt.close()
        return

    ax.set_title(title, pad=20, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel(ylabel)
    
    # 圖例維持 2 欄，現在畫布夠寬了，文字不會擠壓到圖形邊界
    ax.legend(bbox_to_anchor=(0.5, -0.12), loc='upper center', borderaxespad=0., 
              ncol=2, fontsize=17, labelspacing=0.6)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300, bbox_inches='tight')
    plt.close()
# ==========================================
# 主程式
# ==========================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 【修改處】在此處呼叫設定字體的函數，套用全局大字體
    try:
        setup_fonts()
    except Exception as e:
        print(f"字體設定發生問題，將使用預設字體: {e}")
    
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
            stats[f'Last{LAST_K_EPOCHS}_Std_mIoU(%)'] = df['test_iou'].tail(LAST_K_EPOCHS).std() * 100 
            
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
    plot_comparison_black(data_dict, 'test_iou', 'mIoU Comparison (Foreground Overlap)', 'mIoU', 'compare_mIoU_b.png')
    plot_comparison_black(data_dict, 'shift_err', 'Spatial Shift Error Comparison', 'Distance Error', 'compare_shift_err_b.png')
    plot_comparison_black(data_dict, 'success_rate', 'Success Rate Comparison', 'Rate', 'compare_success_rate_b.png', use_smoothing=False)
    plot_comparison_black(data_dict, 'test_loss', 'Validation Loss Comparison', 'Loss', 'compare_loss_b.png')
    
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