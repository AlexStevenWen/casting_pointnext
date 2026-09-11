import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# ==========================================
# [字體與格式修改區] - 針對 Linux 環境優化
# ==========================================
plt.style.use('seaborn-v0_8-whitegrid')

# 設定英文字體優先為 Times New Roman
# 遇到中文退回使用 Linux 標楷體 (AR PL UKai TW)、Noto Sans 或是微軟/Mac標楷體作相容
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = [
    'Times New Roman', 
    'AR PL UKai TW',       # Linux 開源標楷體 (繁體)
    'AR PL UKai HK',       # Linux 開源標楷體 (香港)
    'Noto Sans CJK TC',    # Linux 常用開源中文字體保底
    'DFKai-SB', 'BiauKai', 'KaiTi' # Windows/Mac 保底
]
plt.rcParams['axes.unicode_minus'] = False 

# 全域預設字體大小 (雖然下面有手動指定，但設定全域可確保細節如 Colorbar 也能放大)
plt.rcParams['font.size'] = 16          
plt.rcParams['axes.titlesize'] = 24     
plt.rcParams['axes.labelsize'] = 20     
plt.rcParams['xtick.labelsize'] = 16    
plt.rcParams['ytick.labelsize'] = 16    

# ==========================================
# 1. 設定你的實驗資料夾路徑 (請替換成實際的路徑)
# ==========================================
EXP_DIR = './log/classification/test_pointnext_cls_keypoint_2026-07-01_01-58' 

def replot_evaluation(exp_dir):
    print(f"正在讀取目錄: {exp_dir}")
    
    # ==========================================
    # 2. 重新繪製困難排行 (加大畫布與字體)
    # ==========================================
    summary_path = os.path.join(exp_dir, 'class_error_summary.csv')
    if os.path.exists(summary_path):
        df_summary = pd.read_csv(summary_path)
        df_plot = df_summary.sort_values(by='Error_Rate', ascending=False)
        
        # 加大畫布高度 (因應40個類別)
        plt.figure(figsize=(14, 16))
        sns.barplot(data=df_plot, x='Error_Rate', y='Class', hue='Class', palette='Reds_r', legend=False)
        
        # 放大所有字體
        plt.title('Error Rate by Class (Hardest Classes on Top)', fontsize=24, fontweight='bold', pad=24)
        plt.xlabel('Error Rate (0.0 - 1.0)', fontsize=24, fontweight='bold', labelpad=15)
        plt.ylabel('Category', fontsize=24, fontweight='bold', labelpad=15)
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24) # 放大 Y 軸類別字體
        plt.xlim(0, 1.0) 
        plt.grid(axis='x', linestyle='--', alpha=0.5)
        plt.tight_layout()
        
        save_path = os.path.join(exp_dir, 'class_error_ranking_large.png')
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"✅ 已儲存放大的困難排行: {save_path}")

    # ==========================================
    # 3. 重新繪製混淆矩陣 (加大畫布與內部數字)
    # ==========================================
    report_path = os.path.join(exp_dir, 'inference_full_report.csv')
    if os.path.exists(report_path):
        df_results = pd.read_csv(report_path)
        
        # 從大表還原類別名稱清單與對應 ID
        class_names_df = df_results[['True_ID', 'True_Name']].drop_duplicates().sort_values('True_ID')
        class_names = class_names_df['True_Name'].tolist()
        
        for split_label in ['Train', 'Test', 'All']:
            if split_label == 'All':
                df_split = df_results
            else:
                df_split = df_results[df_results['Split'] == split_label]
            
            if df_split.empty:
                continue
                
            y_true = df_split['True_ID'].tolist()
            y_pred = df_split['Pred_ID'].tolist()
            
            cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
            row_sums = cm.sum(axis=1)[:, np.newaxis]
            cm_norm = np.divide(cm.astype('float'), row_sums, 
                                out=np.zeros_like(cm, dtype='float'), 
                                where=row_sums!=0)

            # 2. 針對 5 類左右的最佳化設定
            plt.figure(figsize=(10, 8)) # 【關鍵】將畫布縮小到適當比例

            # 3. 畫熱力圖，將內部數字 (annot_kws) 設為 18
            sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap='Blues', 
                        annot_kws={"size": 18}, square=True,  # 內部數字大小
                        cbar_kws={"shrink": 0.8},
                        linewidths=0.5, linecolor='lightgray', 
                        xticklabels=class_names, yticklabels=class_names)

            # 4. 放大標題與軸標籤
            plt.title(f'Normalized Confusion Matrix ({split_label} Set)', fontsize=22, fontweight='bold', pad=20)
            plt.ylabel('True Label', fontsize=18, fontweight='bold', labelpad=15)
            plt.xlabel('Predicted Label', fontsize=18, fontweight='bold', labelpad=15)

            # 5. 放大刻度字體 (X軸/Y軸的類別名稱)
            plt.xticks(rotation=45, ha='right', fontsize=16)
            plt.yticks(rotation=0, fontsize=16)

            plt.tight_layout()

            save_name = f'confusion_matrix_heatmap_{split_label.lower()}.png'
            cm_save_path = os.path.join(exp_dir, save_name)
            plt.savefig(cm_save_path, dpi=300)
            plt.close()
            # 修正了原本這裡印錯變數的問題
            print(f"✅ 已儲存放大的混淆矩陣 ({split_label}): {cm_save_path}")

if __name__ == '__main__':
    replot_evaluation(EXP_DIR)