import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import argparse

def main(args):
    # 1. 讀取資料 (這部分不變)
    try:
        data = np.load(args.file)
    except FileNotFoundError:
        print(f"Error: File not found at {args.file}")
        return

    points = data['points']
    attributions = data['attributions']
    print(f"Loaded points: {points.shape}")
    print(f"Loaded attributions: {attributions.shape}")
    if 'target_label' in data:
        print(f"Target label: {data['target_label']}, Predicted label: {data['pred_label']}")
    
    # 2. 建立 Open3D 點雲物件 (這部分不變)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3]) 
    
    # 3. 建立顏色 (這部分不變)
    norm_attr = (attributions - attributions.min()) / (attributions.max() - attributions.min() + 1e-6)
    cmap = plt.get_cmap('jet') 
    colors = cmap(norm_attr)
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

    # 4. 【修改點】使用無頭模式 (Headless) 渲染並存檔
    print("Rendering headless and saving to 'snapshot.png'...")
    try:
        # 建立一個不可見的渲染器
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False) # 設為 False
        vis.add_geometry(pcd)
        
        # (可選) 調整視角
        # 你可能需要調整這些數字來獲得好的視角
        ctr = vis.get_view_control()
        ctr.set_front([0, -1, 0.5]) # 視線方向
        ctr.set_lookat([0, 0, 0])  # 看向的中心點
        ctr.set_up([0, 0, 1])      # 哪個方向是「上」
        ctr.set_zoom(0.5)          # 縮放
        
        # 擷取畫面並儲存
        vis.capture_screen_image("snapshot.png", do_render=True)
        vis.destroy_window()
        print("Successfully saved to 'snapshot.png'")
        
    except Exception as e:
        print(f"[Open3D WARNING] Headless rendering failed: {e}")
        print("This might still indicate a graphics driver issue even in headless mode.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Captum attributions from an .npz file")
    parser.add_argument('--file', type=str, required=True, help='Path to the captum_sample_0.npz file')
    args = parser.parse_args()
    main(args)