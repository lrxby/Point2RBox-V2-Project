import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from collections import Counter

# --- 数据集配置 ---
CODRONE_CLASSES = (
    'car', 'truck', 'bus', 'traffic-light',
    'traffic-sign', 'bridge', 'people', 'bicycle',
    'motor', 'tricycle', 'boat', 'ship'
)
DATASET_PATH = "/mnt/data/xiekaikai/split_ss_codrone"
# 结果保存路径
SAVE_PATH = "/mnt/data/liurunxiang/workplace/point2rbox-v2-loss-new/codrone_analysis.png"

def poly_to_obb(poly):
    """将8参数格式 [x1, y1, ..., x4, y4] 转换为 (cx, cy, w, h)"""
    points = np.array(poly).reshape((4, 2)).astype(np.float32)
    # 使用OpenCV计算最小外接矩形
    (cx, cy), (w, h), angle = cv2.minAreaRect(points)
    # 统一约定：width为长边，height为短边
    width = max(w, h)
    height = min(w, h)
    return cx, cy, width, height

def visualize_codrone(cat_data, target_cat, save_path):
    """可视化分析：尺寸分布与线性趋势线"""
    if target_cat not in cat_data or not cat_data[target_cat]:
        print(f"找不到类别数据: {target_cat}")
        return

    # 找一个物体最密集的图片进行展示（CODrone中通常是boat或people）
    img_counts = Counter([x['img_id'] for x in cat_data[target_cat]])
    best_img_id = img_counts.most_common(1)[0][0]
    group = [x for x in cat_data[target_cat] if x['img_id'] == best_img_id]
    
    ys = np.array([obj['cy'] for obj in group])
    ws = np.array([obj['w'] for obj in group])
    
    plt.figure(figsize=(10, 6))
    plt.scatter(ys, ws, color='darkorange', alpha=0.6, label=f'Instances (N={len(group)})')
    
    if len(ys) > 1:
        z = np.polyfit(ys, ws, 1)
        p = np.poly1d(z)
        plt.plot(ys, p(ys), "r--", label='Perspective Trend')

    plt.title(f"CODrone Analysis: {target_cat} in Image {best_img_id}")
    plt.xlabel("Y-coordinate (Pixels)")
    plt.ylabel("Object Width (Pixels)")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.savefig(save_path)
    print(f"\n[可视化] 统计图已保存至: {save_path}")
    plt.close()

def verify_codrone(base_path):
    # 遍历 trainval 文件夹下的 annfiles
    ann_dir = os.path.join(base_path, 'trainval', 'annfiles')
    if not os.path.exists(ann_dir):
        print(f"错误: 找不到路径 {ann_dir}")
        return

    cat_data = {name: [] for name in CODRONE_CLASSES}
    print(f"正在解析 CODrone 标注文件: {ann_dir}")
    
    files = [f for f in os.listdir(ann_dir) if f.endswith('.txt')]
    for filename in files:
        img_id = filename.split('.')[0]
        with open(os.path.join(ann_dir, filename), 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) < 9: continue
                
                try:
                    # 前8位是坐标，第9位是类别名称
                    poly = [float(x) for x in parts[:8]]
                    category = parts[8]
                    
                    if category not in CODRONE_CLASSES: continue
                    
                    cx, cy, w, h = poly_to_obb(poly)
                    cat_data[category].append({'img_id': img_id, 'cx': cx, 'cy': cy, 'w': w, 'h': h})
                except ValueError:
                    continue

    print(f"\n" + "="*65)
    print(f"{'Category':<18} | {'Count':<8} | {'Avg CV (Size)':<12} | {'Avg R2 (Linear)':<8}")
    print("-" * 65)
    
    for cat in CODRONE_CLASSES:
        items = cat_data[cat]
        if not items: continue
        
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
            
            # 计算变异系数 (CV)
            cv = np.std(ws) / (np.mean(ws) + 1e-6)
            cv_list.append(cv)
            
            # 计算线性拟合优度 (R2)
            if len(group) >= 10:
                model = LinearRegression()
                model.fit(ys, ws)
                r2 = model.score(ys, ws)
                r2_list.append(r2)

        if cv_list:
            avg_cv = np.mean(cv_list)
            avg_r2 = np.mean(r2_list) if r2_list else 0.0
            print(f"{cat:<18} | {len(items):<8} | {avg_cv:.4f} | {avg_r2:.4f}")

    # 针对CODrone，boat通常是验证一致性最好的类别
    visualize_codrone(cat_data, 'boat', SAVE_PATH)

if __name__ == "__main__":
    verify_codrone(DATASET_PATH)
