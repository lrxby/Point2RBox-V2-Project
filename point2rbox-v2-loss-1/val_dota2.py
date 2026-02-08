import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import math

# --- 数据集配置 ---
CLASSES = ('plane', 'baseball-diamond', 'bridge', 'ground-track-field',
         'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
         'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout',
         'harbor', 'swimming-pool', 'helicopter')
DATASET_PATH = "/mnt/data/xiekaikai/split_ss_dota"
SAVE_DIR = "/mnt/data/liurunxiang/workplace/point2rbox-v2-loss-new/global_report_dota_area"
os.makedirs(SAVE_DIR, exist_ok=True)

def poly_to_obb_area(poly):
    """将8参数格式转换为 (cx, cy, area)"""
    points = np.array(poly).reshape((4, 2)).astype(np.float32)
    (cx, cy), (w, h), angle = cv2.minAreaRect(points)
    area = w * h
    return cx, cy, area

def run_full_analysis(base_path):
    ann_dir = os.path.join(base_path, 'trainval', 'annfiles')
    if not os.path.exists(ann_dir): ann_dir = os.path.join(base_path, 'train', 'annfiles')
    
    cat_global_areas = {name: [] for name in CLASSES}
    cat_image_stats = {name: [] for name in CLASSES} 

    files = [f for f in os.listdir(ann_dir) if f.endswith('.txt')]
    for filename in files:
        img_objs = {name: [] for name in CLASSES}
        with open(os.path.join(ann_dir, filename), 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) < 9: continue
                category = parts[8] if parts[8] in CLASSES else None
                if not category: continue
                poly = [float(x) for x in parts[:8]]
                cx, cy, area = poly_to_obb_area(poly)
                img_objs[category].append({'y': cy, 'area': area})
                cat_global_areas[category].append(area)

        for cat, objs in img_objs.items():
            if len(objs) >= 5:
                areas = np.array([o['area'] for o in objs])
                cv = np.std(areas) / (np.mean(areas) + 1e-6)
                slope, r2 = 0.0, 0.0
                if len(objs) >= 10:
                    ys = np.array([o['y'] for o in objs]).reshape(-1, 1)
                    model = LinearRegression().fit(ys, areas)
                    slope, r2 = model.coef_[0], model.score(ys, areas)
                cat_image_stats[cat].append({'cv': cv, 'slope': slope, 'r2': r2, 'count': len(objs)})

    print(f"\n{'Category':<15} | {'Global CV':<10} | {'Avg Local CV':<12} | {'Avg R2 Area':<8} | {'Pos Slope %'}")
    print("-" * 80)
    for cat in CLASSES:
        g_areas = cat_global_areas[cat]
        if not g_areas: continue
        g_cv = np.std(g_areas) / (np.mean(g_areas) + 1e-6)
        l_stats = cat_image_stats[cat]
        if l_stats:
            avg_l_cv = np.mean([s['cv'] for s in l_stats])
            valid_r2 = [s for s in l_stats if s['count'] >= 10]
            avg_r2 = np.mean([s['r2'] for s in valid_r2]) if valid_r2 else 0.0
            pos_slope = np.mean([1 if s['slope'] > 0 else 0 for s in valid_r2]) * 100 if valid_r2 else 0.0
            print(f"{cat:<15} | {g_cv:.4f}    | {avg_l_cv:.4f}     | {avg_r2:.4f}    | {pos_slope:.1f}%")

    plot_grid_summary(cat_global_areas, "Global Area Dist", "Area (px^2)", "area_dist_all.png")
    plot_grid_summary({cat: [s['r2'] for s in stats if s['count'] >= 10] for cat, stats in cat_image_stats.items()}, "Local R2 (Area)", "R2 Value", "r2_area_dist_all.png")

def plot_grid_summary(data_dict, title, xlabel, filename):
    valid_cats = [cat for cat in CLASSES if len(data_dict[cat]) > 0]
    cols = 4
    rows = math.ceil(len(valid_cats) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(20, rows * 4))
    axes = axes.flatten()
    for i, cat in enumerate(valid_cats):
        axes[i].hist(data_dict[cat], bins=40, color='lightcoral', edgecolor='black', alpha=0.7)
        axes[i].set_title(f"{cat} (N={len(data_dict[cat])})")
        axes[i].set_xlabel(xlabel)
    for j in range(i + 1, len(axes)): axes[j].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, filename))
    plt.close()

if __name__ == "__main__":
    run_full_analysis(DATASET_PATH)