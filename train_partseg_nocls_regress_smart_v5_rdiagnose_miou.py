"""
diagnose_miou.py
----------------
把一個 (或多個) 實驗資料夾裡所有「best mIoU」的候選定義同時印出來，
用來確認你手上兩個對不起來的數字分別是哪一個。

用法:
    python diagnose_miou.py ./log/part_seg/exp_A  ./log/part_seg/exp_B
"""

import os
import sys
import numpy as np
import pandas as pd


def reproduce_checkpoint_epoch(df):
    """
    完整複製訓練腳本的 is_best 判準 (success_rate 為主, test_mse 破平手),
    找出實際被存進 best_regression_model.pth 的那個 epoch。
    """
    if 'success_rate' not in df.columns:
        return None, 0

    best_sr, best_mse = -1.0, float('inf')
    best_epoch = None
    n_tied_at_max = 0

    for _, row in df.iterrows():
        sr = float(row['success_rate'])
        mse = float(row['test_mse']) if 'test_mse' in df.columns else 0.0

        is_best = False
        if sr > best_sr:
            is_best = True
        elif sr == best_sr and mse < (best_mse - 1e-5):
            is_best = True

        if is_best:
            best_sr, best_mse = sr, mse
            best_epoch = int(row['Epoch'])

    n_tied_at_max = int((df['success_rate'] == df['success_rate'].max()).sum())
    return best_epoch, n_tied_at_max


def th_cols_of(df):
    cols = [c for c in df.columns if c.startswith('mIoU_')]
    return sorted(cols, key=lambda c: float(c.split('_')[1]))


def diagnose(exp_dir):
    print('=' * 78)
    print(f'實驗資料夾: {exp_dir}')
    print('=' * 78)

    hist_csv = os.path.join(exp_dir, 'training_metrics_history.csv')
    if not os.path.exists(hist_csv):
        print(f'  [跳過] 找不到 {hist_csv}')
        return

    df = pd.read_csv(hist_csv)
    if 'Epoch' not in df.columns:
        df.insert(0, 'Epoch', np.arange(1, len(df) + 1))

    n = len(df)
    last_epoch = int(df['Epoch'].iloc[-1])
    print(f'總 epoch 數: {n}   (最後一個 epoch = {last_epoch})\n')

    # ---------------------------------------------------------------
    # A. test_iou：固定門檻 0.5，batch 平均的平均
    # ---------------------------------------------------------------
    if 'test_iou' in df.columns:
        i = df['test_iou'].idxmax()
        print('[A] history["test_iou"]  (固定 th=reg_threshold, batch 平均的平均)')
        print(f'      max        = {df.loc[i, "test_iou"]*100:8.4f} %   @ epoch {int(df.loc[i, "Epoch"])}')
        print(f'      Top5 平均  = {df["test_iou"].nlargest(5).mean()*100:8.4f} %')
        print(f'      末 20 平均 = {df["test_iou"].tail(20).mean()*100:8.4f} %')
        print(f'      <-- 消融腳本的 Best_mIoU(%) 就是這個 max\n')

    # ---------------------------------------------------------------
    # B. mIoU_0.5：固定門檻 0.5，逐樣本平均
    # ---------------------------------------------------------------
    if 'mIoU_0.5' in df.columns:
        j = df['mIoU_0.5'].idxmax()
        print('[B] history["mIoU_0.5"]  (固定 th=0.5, 逐樣本平均)')
        print(f'      max        = {df.loc[j, "mIoU_0.5"]*100:8.4f} %   @ epoch {int(df.loc[j, "Epoch"])}')
        print('      <-- 論文該報的是這個口徑\n')

    # ---------------------------------------------------------------
    # A vs B：同門檻不同平均方式造成的落差
    # ---------------------------------------------------------------
    if {'test_iou', 'mIoU_0.5'}.issubset(df.columns):
        delta = (df['test_iou'] - df['mIoU_0.5']) * 100
        print('[A vs B] 兩者在「同一個 epoch」的差距 (平均的平均 vs 逐樣本平均)')
        print(f'      平均差 = {delta.mean():+8.4f} pp    最大差 = {delta.abs().max():8.4f} pp')
        if delta.abs().max() < 1e-6:
            print('      -> 幾乎沒有差距，代表你的 test set 剛好整除 batch_size。')
        else:
            print('      -> 有差距。testDataLoader 沒有 drop_last，最後一個小 batch')
            print('         在 test_iou 裡被過度加權了。')
        print()

    # ---------------------------------------------------------------
    # C. 各 epoch 的跨門檻最佳值
    # ---------------------------------------------------------------
    if 'best_scan_iou' in df.columns:
        k = df['best_scan_iou'].idxmax()
        print('[C] history["best_scan_iou"]  (每個 epoch 在 0.1~0.9 中最好的門檻)')
        print(f'      max        = {df.loc[k, "best_scan_iou"]*100:8.4f} %   @ epoch {int(df.loc[k, "Epoch"])}'
              f'   (th={df.loc[k, "best_scan_th"]:.1f})')
        print('      <-- 這是「最佳 epoch × 最佳門檻」的雙重挑選，會系統性高估\n')

    # ---------------------------------------------------------------
    # D. threshold_vs_miou.png 的兩種取法
    # ---------------------------------------------------------------
    tcols = th_cols_of(df)
    if tcols:
        print('[D] threshold_vs_miou.png 這張圖的內容取決於「挑哪個 epoch」')

        # D1: 原始訓練腳本的行為 = 最後一個 epoch (每 epoch 覆蓋存檔)
        row = df.iloc[-1]
        vals = [float(row[c]) * 100 for c in tcols]
        bi = int(np.argmax(vals))
        print(f'      D1 最後 epoch ({last_epoch})           : best th={float(tcols[bi].split("_")[1]):.1f}'
              f'  mIoU={vals[bi]:.4f} %   <-- 原始腳本存下來的就是這張')

        # D2: replot 腳本的行為 = success_rate 最高 (平手取第一個)
        if 'success_rate' in df.columns:
            r2 = df.loc[df['success_rate'].idxmax()]
            v2 = [float(r2[c]) * 100 for c in tcols]
            b2 = int(np.argmax(v2))
            print(f'      D2 success_rate 最高 (epoch {int(r2["Epoch"]):>4}) : best th={float(tcols[b2].split("_")[1]):.1f}'
                  f'  mIoU={v2[b2]:.4f} %   <-- replot 腳本畫的是這張')

        # D3: 真正被存成 checkpoint 的 epoch
        ck_epoch, n_tied = reproduce_checkpoint_epoch(df)
        if ck_epoch is not None:
            r3 = df[df['Epoch'] == ck_epoch].iloc[0]
            v3 = [float(r3[c]) * 100 for c in tcols]
            b3 = int(np.argmax(v3))
            print(f'      D3 實際存檔的 checkpoint (epoch {ck_epoch:>4}) : best th={float(tcols[b3].split("_")[1]):.1f}'
                  f'  mIoU={v3[b3]:.4f} %   <-- 你交付的模型')
            print(f'         (success_rate 在最大值上共有 {n_tied} 個 epoch 平手，'
                  f'訓練時用 test_mse 破平手)')
            if 'success_rate' in df.columns and ck_epoch != int(df.loc[df['success_rate'].idxmax(), 'Epoch']):
                print('         !! D2 與 D3 不同 epoch —— replot 腳本缺了 MSE 破平手邏輯')
        print()

    # ---------------------------------------------------------------
    # E. 最終推論報表 (best checkpoint 實際跑出來的結果)
    # ---------------------------------------------------------------
    rep_csv = os.path.join(exp_dir, 'logs', 'final_inference_report.csv')
    if os.path.exists(rep_csv):
        rep = pd.read_csv(rep_csv)
        sub = rep[rep['Split'] == 'test'] if 'Split' in rep.columns else rep
        rcols = th_cols_of(sub)
        print(f'[E] final_inference_report.csv  (best checkpoint, test set, n={len(sub)})')
        if rcols:
            rvals = [sub[c].mean() * 100 for c in rcols]
            rb = int(np.argmax(rvals))
            print(f'      th=0.5 逐檔平均 = {sub["mIoU_0.5"].mean()*100:8.4f} %'
                  if 'mIoU_0.5' in sub.columns else '')
            print(f'      跨門檻最佳      = {rvals[rb]:8.4f} %  (th={float(rcols[rb].split("_")[1]):.1f})')
        if 'Success' in sub.columns:
            print(f'      success rate    = {sub["Success"].mean()*100:8.4f} %')
        print('      <-- 這才是「你交付的模型的真實表現」，')
        print('          它沒有理由等於 [A] 或 [C] 的 max\n')
    else:
        print(f'[E] 找不到 {rep_csv}（訓練可能被中斷，沒跑到最終推論階段）\n')


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    for d in sys.argv[1:]:
        diagnose(d)


if __name__ == '__main__':
    main()