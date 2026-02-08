import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from collections import Counter

# --- 数据集配置 ---
CLASSES = ('car', 'bus', 'truck', 'van', 'freight_car')
DATASET_PATH = "/mnt/data/xiekaikai/DroneVehicle"
# 结果保存路径
SAVE_PATH = "/mnt/data/liurunxiang/workplace/point2rbox-v2-loss-new/dronevehicle_analysis.png"

def poly_to_obb(poly):
    """将8参数格式 [x1, y1, ..., x4, y4] 转换为 (cx, cy, w, h)"""
    points = np.array(poly).reshape((4, 2)).astype(np.float32)
    # 使用OpenCV计算最小外接矩形
    (cx, cy), (w, h), angle = cv2.minAreaRect(points)
    # 统一约定：width为长边，height为短边，消除旋转导致的长宽定义颠倒
    width = max(w, h)
    height = min(w, h)
    return cx, cy, width, height

def visualize_analysis(cat_data, target_cat_name, save_path):
    """可视化指定类别的尺寸随坐标分布情况"""
    if target_cat_name not in cat_data:
        print(f"找不到类别: {target_cat_name}")
        return

    # 找一个该类别实例最多的图片进行展示
    img_counts = Counter([x['img_id'] for x in cat_data[target_cat_name]])
    if not img_counts: return
    
    best_img_id = img_counts.most_common(1)[0][0]
    group = [x for x in cat_data[target_cat_name] if x['img_id'] == best_img_id]
    
    ys = np.array([obj['cy'] for obj in group])
    ws = np.array([obj['w'] for obj in group])
    
    plt.figure(figsize=(10, 6))
    plt.scatter(ys, ws, color='forestgreen', alpha=0.6, label=f'Instances (N={len(group)})')
    
    if len(ys) > 1:
        z = np.polyfit(ys, ws, 1)
        p = np.poly1d(z)
        plt.plot(ys, p(ys), "r--", linewidth=2, label='Linear Trend (Perspective)')

    plt.title(f"DroneVehicle Analysis: {target_cat_name} in {best_img_id}")
    plt.xlabel("Y-coordinate (Pixels)")
    plt.ylabel("Object Long-side Width (Pixels)")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.savefig(save_path)
    print(f"\n[可视化] 分析图已保存至: {save_path}")
    plt.close()

def verify_dronevehicle(base_path):
    # 建议验证 train 或 val 文件夹
    ann_dir = os.path.join(base_path, 'train', 'annfiles')
    if not os.path.exists(ann_dir):
        print(f"错误: 找不到标注目录 {ann_dir}")
        return

    cat_data = {name: [] for name in CLASSES}
    print(f"开始解析 DroneVehicle 标注: {ann_dir}")
    
    files = [f for f in os.listdir(ann_dir) if f.endswith('.txt')]
    for filename in files:
        img_id = filename.split('.')[0]
        with open(os.path.join(ann_dir, filename), 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) < 9: continue
                
                # DroneVehicle 格式: x1 y1 x2 y2 x3 y3 x4 y4 class_idx
                poly = [float(x) for x in parts[:8]]
                class_idx = int(parts[8])
                
                if class_idx >= len(CLASSES): continue
                category = CLASSES[class_idx]
                
                cx, cy, w, h = poly_to_obb(poly)
                cat_data[category].append({'img_id': img_id, 'cx': cx, 'cy': cy, 'w': w, 'h': h})

    print(f"\n" + "="*60)
    print(f"{'Category':<15} | {'Count':<8} | {'Avg CV (Size)':<12} | {'Avg R2 (Linear)':<8}")
    print("-" * 60)
    
    for cat in CLASSES:
        items = cat_data[cat]
        if not items: continue
        
        # 按图片分组
        img_groups = {}
        for item in items:
            iid = item['img_id']
            if iid not in img_groups: img_groups[iid] = []
            img_groups[iid].append(item)
        
        cv_list = []
        r2_list = []
        
        for iid, group in img_groups.items():
            if len(group) < 5: continue
            
            ws = np.array([obj['w'] for obj in group])
            ys = np.array([obj['cy'] for obj in group]).reshape(-1, 1)
            
            # 1. 计算变异系数 (CV)
            cv = np.std(ws) / (np.mean(ws) + 1e-6)
            cv_list.append(cv)
            
            # 2. 计算拟合优度 (R2)
            if len(group) >= 10:
                model = LinearRegression()
                model.fit(ys, ws)
                r2 = model.score(ys, ws)
                r2_list.append(r2)

        if cv_list:
            avg_cv = np.mean(cv_list)
            avg_r2 = np.mean(r2_list) if r2_list else 0.0
            print(f"{cat:<15} | {len(items):<8} | {avg_cv:.4f} | {avg_r2:.4f}")

    # 可视化样本最多的类（通常是 car）
    visualize_analysis(cat_data, 'car', SAVE_PATH)

if __name__ == "__main__":
    verify_dronevehicle(DATASET_PATH)
