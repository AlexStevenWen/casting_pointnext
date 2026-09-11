"""
replot_from_csv.py  (修正版)
----------------------------
從已經訓練完成的 exp_dir 讀取 CSV，重新繪製所有圖表。
不需要重新訓練，也不需要載入 .pth 權重。

用法:
    python replot_from_csv.py --exp_dir ./log/part_seg/exp_xxx_2026-06-30_14-27-05
    python replot_from_csv.py --exp_dir <上面那個路徑> --out_dir ./replot
    python replot_from_csv.py --exp_dir <...> --epoch_select last   # 復刻原始腳本行為

本次修正:
  1. threshold_vs_miou.png 預設改為「實際存檔的 checkpoint 對應的 epoch」，
     並完整複製訓練腳本的 is_best 判準 (success_rate 為主, test_mse 破平手)。
     舊版只用 idxmax，平手時會挑到錯誤的 epoch。
  2. error_dist.png 加上防呆：若 final_inference_report.csv 的 Shift_Error
     退化成整欄 1.0 (訓練腳本 reg_threshold 殘值 bug 造成)，改為略過並警告，
     不再產生誤導性的圖。
  3. metrics_iou.png 同時畫 batch 加權與逐樣本兩種口徑。
  4. 結束時印出論文可直接引用的口徑摘要。
"""

import os
import argparse
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


# =====================================================================
# 字體設定：Times New Roman (英文/數字) + 標楷體 (中文)
# =====================================================================
def setup_fonts(times_path="/mnt/c/Windows/Fonts/times.ttf",
                kaiu_path="/mnt/c/Windows/Fonts/kaiu.ttf"):
    # style 一定要先設，否則會把後面的字體設定洗掉
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

    # 直接把實體字體名稱放進 font.family，不要透過 sans-serif 別名繞一層。
    # matplotlib >= 3.6 才支援逐字元備援 (per-glyph fallback)。
    mpl.rcParams['font.family'] = names + ['DejaVu Sans']
    mpl.rcParams['axes.unicode_minus'] = False

    ver = tuple(int(x) for x in matplotlib.__version__.split('.')[:2])
    if ver < (3, 6):
        print(f"[警告] matplotlib {matplotlib.__version__} 不支援逐字元字體備援。")
        print("       純英文圖表不受影響；若圖中有中文會變成 □。")
        print("       建議: pip install -U 'matplotlib>=3.8' && rm -rf ~/.cache/matplotlib")
    else:
        print(f"字體掛載完成 (matplotlib {matplotlib.__version__}): {names}")

    mpl.rcParams['font.size'] = 14
    mpl.rcParams['axes.titlesize'] = 18
    mpl.rcParams['axes.labelsize'] = 15
    mpl.rcParams['xtick.labelsize'] = 13
    mpl.rcParams['ytick.labelsize'] = 13
    mpl.rcParams['legend.fontsize'] = 13
    mpl.rcParams['savefig.dpi'] = 300
    mpl.rcParams['figure.dpi'] = 150


# =====================================================================
# 工具：找出實際被存進 best_regression_model.pth 的 epoch
# =====================================================================
def find_checkpoint_epoch(df):
    """
    完整複製訓練腳本的 is_best 判準：
        success_rate 嚴格變大            -> best
        success_rate 相等 且 test_mse 明顯變小 -> best
    回傳 (epoch, 說明字串)。
    """
    if 'success_rate' not in df.columns:
        return int(df['Epoch'].iloc[-1]), '無 success_rate 欄，退回最後一個 epoch'

    best_sr, best_mse = -1.0, float('inf')
    best_epoch = int(df['Epoch'].iloc[0])
    has_mse = 'test_mse' in df.columns

    for _, row in df.iterrows():
        sr = float(row['success_rate'])
        mse = float(row['test_mse']) if has_mse else 0.0

        if sr > best_sr:
            best_sr, best_mse = sr, mse
            best_epoch = int(row['Epoch'])
        elif sr == best_sr and mse < (best_mse - 1e-5):
            best_mse = mse
            best_epoch = int(row['Epoch'])

    n_tied = int((df['success_rate'] == df['success_rate'].max()).sum())
    note = f'success_rate={best_sr:.4f}, 最大值處共 {n_tied} 個 epoch 平手'
    return best_epoch, note


def th_cols_of(df):
    cols = [c for c in df.columns if c.startswith('mIoU_')]
    return sorted(cols, key=lambda c: float(c.split('_')[1]))


def select_row(df, mode):
    """依 mode 選出要拿來畫閾值曲線的那一列。"""
    if mode == 'last':
        row = df.iloc[-1]
        return row, int(row['Epoch']), '最後一個 epoch (復刻原始腳本行為)'

    if mode == 'best_iou':
        key = 'mIoU_0.5' if 'mIoU_0.5' in df.columns else 'test_iou'
        row = df.loc[df[key].idxmax()]
        return row, int(row['Epoch']), f'{key} 最高的 epoch'

    ep, note = find_checkpoint_epoch(df)
    row = df[df['Epoch'] == ep].iloc[0]
    return row, ep, f'實際存檔的 checkpoint ({note})'


# =====================================================================
# 各張圖
# =====================================================================
def plot_performance(out_dir, df):
    epochs = df['Epoch'].values

    # 1. Loss 曲線
    if {'train_loss', 'test_loss'}.issubset(df.columns):
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, df['train_loss'], 'r-', label='Train Loss')
        plt.plot(epochs, df['test_loss'], 'b--', label='Test Loss')
        plt.title('Loss Convergence')
        plt.xlabel('Epoch'); plt.ylabel('Loss')
        plt.legend(); plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'loss.png'), bbox_inches='tight')
        plt.close()

    # 2. mIoU 曲線（同時畫兩種口徑，讓落差一目了然）
    if 'test_iou' in df.columns:
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, df['test_iou'], 'g-', label='Test mIoU (batch-weighted)')
        if 'mIoU_0.5' in df.columns:
            plt.plot(epochs, df['mIoU_0.5'], color='darkorange', linestyle='--',
                     label='Test mIoU (per-sample, th=0.5)')
        plt.title('Segmentation Overlap (mIoU)')
        plt.xlabel('Epoch'); plt.ylabel('mIoU')
        plt.legend(); plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'metrics_iou.png'), bbox_inches='tight')
        plt.close()

    # 3. 空間誤差 + 成功率 (雙 Y 軸)
    if {'shift_err', 'success_rate'}.issubset(df.columns):
        fig, ax1 = plt.subplots(figsize=(10, 5))
        c1 = 'tab:red'
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Shift Error (Distance)', color=c1)
        ax1.plot(epochs, df['shift_err'], color=c1, label='Avg Shift Error')
        ax1.tick_params(axis='y', labelcolor=c1)

        ax2 = ax1.twinx()
        c2 = 'tab:blue'
        ax2.set_ylabel('Success Rate', color=c2)
        ax2.plot(epochs, df['success_rate'], color=c2, linestyle='--', label='Success Rate')
        ax2.tick_params(axis='y', labelcolor=c2)
        ax2.grid(False)

        plt.title('Spatial Metrics', pad=15)
        ax1.grid(True, alpha=0.3)
        fig.tight_layout()
        plt.savefig(os.path.join(out_dir, 'metrics_spatial.png'), bbox_inches='tight')
        plt.close(fig)

    # 4. Accuracy (分割模型才有)
    if 'test_acc' in df.columns:
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, df['test_acc'], 'k-', label='Test Accuracy')
        plt.title('Pixel/Point Accuracy')
        plt.xlabel('Epoch'); plt.ylabel('Accuracy')
        plt.legend(); plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'metrics_accuracy.png'), bbox_inches='tight')
        plt.close()

    # 5. MAE / MSE (回歸模型)
    if {'test_mae', 'test_mse'}.issubset(df.columns):
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, df['test_mae'], 'b-', label='Test MAE')
        plt.plot(epochs, df['test_mse'], 'g--', label='Test MSE')
        plt.title('Regression Error')
        plt.xlabel('Epoch'); plt.ylabel('Error Value')
        plt.legend(); plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'regression_error.png'), bbox_inches='tight')
        plt.close()


def plot_threshold_curve(out_dir, df, mode='checkpoint'):
    """從 mIoU_0.1 ~ mIoU_0.9 欄位重畫閾值曲線。"""
    tcols = th_cols_of(df)
    if not tcols:
        print("[略過] threshold_vs_miou.png：找不到 mIoU_* 欄位")
        return None

    row, epoch, why = select_row(df, mode)
    print(f'  threshold_vs_miou.png 取 epoch {epoch} —— {why}')

    thresholds = [float(c.split('_')[1]) for c in tcols]
    mious = [float(row[c]) * 100 for c in tcols]

    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, mious, marker='o', linestyle='-', color='indigo', linewidth=2)
    plt.title(f'Epoch {epoch} - mIoU vs. Decision Threshold')
    plt.xlabel('Decision Threshold')
    plt.ylabel('mIoU (%)')
    plt.grid(True, alpha=0.4)
    plt.xticks(thresholds)

    max_iou = max(mious)
    best_th = thresholds[mious.index(max_iou)]
    plt.axvline(x=best_th, color='red', linestyle='--',
                label=f'Best Th: {best_th:.1f} (mIoU: {max_iou:.2f}%)')

    # 固定門檻 0.5 也標出來，避免只看到 oracle threshold
    if 0.5 in thresholds:
        v05 = mious[thresholds.index(0.5)]
        plt.axvline(x=0.5, color='gray', linestyle=':', linewidth=1.5,
                    label=f'Fixed Th: 0.5 (mIoU: {v05:.2f}%)')

    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'threshold_vs_miou.png'), bbox_inches='tight')
    plt.close()
    return epoch


def plot_error_distribution(out_dir, report_csv, tolerance=0.1):
    """
    用最終推論報表的 Shift_Error 欄重畫誤差分佈直方圖。

    防呆：訓練腳本 save_dataset_inference 有 reg_threshold 殘值 bug，
    會讓 Shift_Error 整欄退化成懲罰值 1.0。這種資料畫出來的圖是錯的，直接略過。
    """
    if not os.path.exists(report_csv):
        print(f"[略過] error_dist.png：找不到 {report_csv}")
        return

    df = pd.read_csv(report_csv)
    if 'Shift_Error' not in df.columns:
        print("[略過] error_dist.png：報表缺少 Shift_Error 欄位")
        return

    if 'Split' in df.columns and (df['Split'] == 'test').any():
        errors = df.loc[df['Split'] == 'test', 'Shift_Error'].astype(float).values
        tag = 'Test Set'
    else:
        errors = df['Shift_Error'].astype(float).values
        tag = 'All Data'

    if len(errors) == 0:
        print("[略過] error_dist.png：沒有可用的誤差資料")
        return

    # --- 退化偵測 ---
    frac_penalty = float(np.mean(np.isclose(errors, 1.0)))
    if frac_penalty > 0.9:
        print()
        print("  " + "!" * 68)
        print(f"  [略過] error_dist.png：Shift_Error 有 {frac_penalty*100:.1f}% 是懲罰值 1.0。")
        print("  這是 save_dataset_inference 的 reg_threshold 殘值 bug —— shift error")
        print("  被用 th=0.9 算出來，pred_mask 幾乎全空所以一律記 1.0、Success 一律 False。")
        print("  請先修好訓練腳本並重跑最終推論，這張圖才有意義。")
        print("  (逐 epoch 的 shift_err 曲線不受影響，見 metrics_spatial.png)")
        print("  " + "!" * 68)
        print()
        return

    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(x=tolerance, color='r', linestyle='--', label=f'Tolerance ({tolerance})')
    plt.title(f'Shift Error Distribution ({tag})')
    plt.xlabel('Spatial Shift Error'); plt.ylabel('Count')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'error_dist.png'), bbox_inches='tight')
    plt.close()


# =====================================================================
def print_summary(df, ck_epoch):
    """印出論文可直接引用的口徑，避免再拿錯數字。"""
    print()
    print('-' * 72)
    print('論文引用口徑摘要')
    print('-' * 72)

    row_df = df[df['Epoch'] == ck_epoch]
    if len(row_df):
        row = row_df.iloc[0]
        print(f'交付模型 = epoch {ck_epoch} 的 checkpoint')
        if 'mIoU_0.5' in df.columns:
            print(f'  mIoU @ th=0.5 (逐樣本, 建議報這個) : {float(row["mIoU_0.5"])*100:7.2f} %')
        if 'test_iou' in df.columns:
            print(f'  test_iou      (batch 加權, 不建議) : {float(row["test_iou"])*100:7.2f} %')
        if 'success_rate' in df.columns:
            print(f'  success rate                       : {float(row["success_rate"])*100:7.2f} %')
        if 'shift_err' in df.columns:
            print(f'  avg shift error                    : {float(row["shift_err"]):7.4f}')

    print('\n對照組 (這些 epoch 的權重已被丟棄, 勿當主數字):')
    if 'test_iou' in df.columns:
        i = df['test_iou'].idxmax()
        print(f'  test_iou 最高       : {df.loc[i, "test_iou"]*100:7.2f} % @ epoch {int(df.loc[i, "Epoch"])}')
    if 'mIoU_0.5' in df.columns:
        j = df['mIoU_0.5'].idxmax()
        print(f'  mIoU_0.5 最高       : {df.loc[j, "mIoU_0.5"]*100:7.2f} % @ epoch {int(df.loc[j, "Epoch"])}')
    if 'best_scan_iou' in df.columns:
        k = df['best_scan_iou'].idxmax()
        print(f'  跨門檻最佳 (oracle) : {df.loc[k, "best_scan_iou"]*100:7.2f} % @ epoch {int(df.loc[k, "Epoch"])}'
              f'  th={df.loc[k, "best_scan_th"]:.1f}')
        print('    ^ 門檻是在 test set 上挑的，等於用測試集調參。')
        print('      要報的話請標註 oracle threshold，或改用 validation set 決定門檻。')
    print('-' * 72)


def main():
    ap = argparse.ArgumentParser('Replot training figures from saved CSVs')
    ap.add_argument('--exp_dir', type=str, required=True,
                    help='訓練產出的資料夾')
    ap.add_argument('--out_dir', type=str, default=None,
                    help='圖片輸出位置，預設直接覆蓋 exp_dir 內的舊圖')
    ap.add_argument('--epoch_select', type=str, default='checkpoint',
                    choices=['checkpoint', 'last', 'best_iou'],
                    help='threshold_vs_miou.png 要用哪個 epoch。'
                         'checkpoint=實際交付的模型(預設); last=復刻原始腳本; best_iou=最佳 mIoU')
    ap.add_argument('--tolerance', type=float, default=0.1)
    ap.add_argument('--times_path', type=str, default='/mnt/c/Windows/Fonts/times.ttf')
    ap.add_argument('--kaiu_path', type=str, default='/mnt/c/Windows/Fonts/kaiu.ttf')
    args = ap.parse_args()

    setup_fonts(args.times_path, args.kaiu_path)

    exp_dir = args.exp_dir
    out_dir = args.out_dir or exp_dir
    os.makedirs(out_dir, exist_ok=True)

    hist_csv = os.path.join(exp_dir, 'training_metrics_history.csv')
    if not os.path.exists(hist_csv):
        raise FileNotFoundError(f'找不到 {hist_csv}')

    df = pd.read_csv(hist_csv)
    if 'Epoch' not in df.columns:
        df.insert(0, 'Epoch', np.arange(1, len(df) + 1))

    print(f'讀取 {len(df)} 個 epoch 的紀錄，輸出到: {out_dir}')

    plot_performance(out_dir, df)
    plot_threshold_curve(out_dir, df, mode=args.epoch_select)
    plot_error_distribution(
        out_dir,
        os.path.join(exp_dir, 'logs', 'final_inference_report.csv'),
        tolerance=args.tolerance
    )

    ck_epoch, _ = find_checkpoint_epoch(df)
    print_summary(df, ck_epoch)
    print('\n完成。')


if __name__ == '__main__':
    main()