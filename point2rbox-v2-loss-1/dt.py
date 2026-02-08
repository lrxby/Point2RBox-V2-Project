import os
import math
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm

# ================= 配置区域 (Strictly Aligned) =================
LABEL_DIR = '/mnt/data/xiekaikai/split_ss_dota/trainval/annfiles'
OUTPUT_DIR = 'work_dirs/loss_statistics_report_dota'

# [Strict] 必须与 configs/point2rbox_v2/point2rbox_v2-1x-dota.py 一致
K_RADIUS = 2.0
ALPHA = 1.0

# 统计所有类别
TARGET_CLASSES = [
    'plane', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 
    'ship', 'tennis-court', 'basketball-court', 'soccer-ball-field', 
    'roundabout', 'baseball-diamond', 'storage-tank',
    'harbor', 'swimming-pool', 'helicopter'
]

# [Strict] Square Classes (无方向物体, 强制归零)
SQUARE_CLASSES = ['baseball-diamond', 'storage-tank', 'roundabout']
# ==============================================================

def parse_dota_poly(poly, class_name):
    """[Strict] 解析坐标，模拟训练时的 Head 处理"""
    pts = np.array(poly).reshape(4, 2)
    rect = cv2.minAreaRect(pts.astype(np.float32))
    (cx, cy), (w, h), angle_deg = rect
    
    if w < h:
        w, h = h, w
        angle_deg += 90
    
    theta = np.deg2rad(angle_deg)
    
    if class_name in SQUARE_CLASSES:
        theta = 0.0

    scale = np.sqrt(w * h)
    scale = np.clip(scale, 16.0, 800.0)

    return {'cx': cx, 'cy': cy, 'theta': theta, 'scale': scale}

def calculate_naoa_loss_vectorized(objects):
    """[Strict] 计算一张图中所有目标的 Loss 列表"""
    N = len(objects)
    if N < 2:
        return np.array([]) 

    centers = np.array([[obj['cx'], obj['cy']] for obj in objects]) 
    scales = np.array([obj['scale'] for obj in objects])            
    thetas = np.array([obj['theta'] for obj in objects])            

    vec_preds = np.stack([np.cos(2 * thetas), np.sin(2 * thetas)], axis=1)

    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=-1)

    sigmas = scales * K_RADIUS
    sigma_mat = sigmas[:, np.newaxis] 
    
    weight_geo = np.exp(-dist_sq / (2 * sigma_mat ** 2))

    mask_diag = 1.0 - np.eye(N)
    W = weight_geo * mask_diag

    target_vecs = np.dot(W, vec_preds)

    consistency_weights = np.linalg.norm(target_vecs, axis=1) + 1e-6
    
    target_dirs = target_vecs / consistency_weights[:, np.newaxis]
    cos_sim = np.sum(vec_preds * target_dirs, axis=1)

    loss_per_item = consistency_weights * (1.0 - cos_sim)
    
    return loss_per_item

def plot_histogram_advanced(loss_data, class_name, save_dir, stats):
    """绘制高级分布直方图"""
    if len(loss_data) == 0: return

    plt.figure(figsize=(20, 10))
    bins = np.arange(0, 2.05, 0.05)
    n, bins, patches = plt.hist(loss_data, bins=bins, color='#66bb6a', edgecolor='black', alpha=0.9)
    
    max_height = 0
    for count, rect in zip(n, patches):
        height = rect.get_height()
        max_height = max(max_height, height)
        if height > 0:
            plt.text(rect.get_x() + rect.get_width()/2, height + (max_height * 0.01), 
                     str(int(height)), 
                     ha='center', va='bottom', fontsize=9, rotation=0, fontweight='bold')

    plt.axvline(stats['mean'], color='red', linestyle='dashed', linewidth=2, label=f"Mean: {stats['mean']:.4f}")
    plt.axvline(stats['p50'], color='blue', linestyle='dashed', linewidth=2, label=f"Median: {stats['p50']:.4f}")
    
    # 详细数据展板
    stats_text = (
        f"Class: {class_name}\n"
        f"Count: {stats['count']}\n"
        f"--------------------\n"
        f"Min: {stats['min']:.4f}\n"
        f"Max: {stats['max']:.4f}\n"
        f"Mean: {stats['mean']:.4f}\n"
        f"--------------------\n"
        f"P10: {stats['p10']:.4f}\n"
        f"P20: {stats['p20']:.4f}\n"
        f"P30: {stats['p30']:.4f}\n"
        f"P40: {stats['p40']:.4f}\n"
        f"(Median)P50: {stats['p50']:.4f}\n"
        f"P60: {stats['p60']:.4f}\n"
        f"P70: {stats['p70']:.4f}\n"
        f"P80: {stats['p80']:.4f}\n"
        f"P90: {stats['p90']:.4f}"
    )
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
    plt.text(0.98, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=11,
             verticalalignment='top', horizontalalignment='right', bbox=props, fontfamily='monospace')

    plt.ylim(0, max_height * 1.15) # 稍微拉高一点Y轴
    
    plt.title(f'GT Loss Distribution: {class_name}', fontsize=20)
    plt.xlabel('NAOA Loss Value (Bin Width = 0.05)', fontsize=15)
    plt.ylabel('Frequency (Count)', fontsize=15)
    
    # [关键修改] 图例位置改为 "upper center" 并微调坐标，彻底避开左侧第一根柱子
    plt.legend(fontsize=12, loc='upper center', bbox_to_anchor=(0.5, 1.0))
    
    plt.grid(axis='y', alpha=0.3)
    
    save_path = os.path.join(save_dir, f'{class_name}_dist_v5.png')
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    label_files = glob.glob(os.path.join(LABEL_DIR, '*.txt'))
    print(f"找到 {len(label_files)} 个标注文件。开始全量统计...")
    
    loss_stats = {cls: [] for cls in TARGET_CLASSES}
    
    for label_file in tqdm(label_files):
        with open(label_file, 'r') as f:
            lines = f.readlines()
            
        objects_by_cls = {cls: [] for cls in TARGET_CLASSES}
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 9: continue
            try:
                poly = [float(x) for x in parts[:8]]
                cls = parts[8]
                if cls in TARGET_CLASSES:
                    obj = parse_dota_poly(poly, cls)
                    objects_by_cls[cls].append(obj)
            except:
                continue
        
        for cls in TARGET_CLASSES:
            objs = objects_by_cls[cls]
            if len(objs) >= 2:
                losses = calculate_naoa_loss_vectorized(objs)
                loss_stats[cls].extend(losses)

    header_template = "{:<20} | {:<8} | {:<8} | {:<8} | {:<8} | {:<8} | {:<8} | {:<8} | {:<8} | {:<8} | {:<8} | {:<8} | {:<8} | {:<8}"
    row_template = "{:<20} | {:<8} | {:<8.4f} | {:<8.4f} | {:<8.4f} | {:<8.4f} | {:<8.4f} | {:<8.4f} | {:<8.4f} | {:<8.4f} | {:<8.4f} | {:<8.4f} | {:<8.4f} | {:<8.4f}"
    
    header_str = header_template.format(
        "Class", "Count", "Mean", "Min", "Max", 
        "P10", "P20", "P30", "P40", "P50", "P60", "P70", "P80", "P90"
    )
    
    print("\n" + "="*160)
    print(header_str)
    print("-" * 160)
    
    report_path = os.path.join(OUTPUT_DIR, 'statistics_report_full.txt')
    
    with open(report_path, 'w') as f_report:
        f_report.write(header_str + "\n" + "-"*160 + "\n")
        
        for cls in TARGET_CLASSES:
            data = np.array(loss_stats[cls])
            
            if len(data) == 0:
                continue
                
            mean_val = np.mean(data)
            min_val = np.min(data)
            max_val = np.max(data)
            percentiles = np.percentile(data, np.arange(10, 100, 10))
            
            stats = {
                'count': len(data),
                'mean': mean_val,
                'min': min_val,
                'max': max_val,
                'p10': percentiles[0],
                'p20': percentiles[1],
                'p30': percentiles[2],
                'p40': percentiles[3],
                'p50': percentiles[4], 
                'p60': percentiles[5],
                'p70': percentiles[6],
                'p80': percentiles[7],
                'p90': percentiles[8]
            }
            
            row_str = row_template.format(
                cls, len(data), mean_val, min_val, max_val,
                stats['p10'], stats['p20'], stats['p30'], stats['p40'], 
                stats['p50'], stats['p60'], stats['p70'], stats['p80'], stats['p90']
            )
            print(row_str)
            f_report.write(row_str + "\n")
            
            plot_histogram_advanced(data, cls, OUTPUT_DIR, stats)

    print("="*160)
    print(f"全量统计报告(含Min/Max)已生成: {report_path}")
    print(f"统计图已保存至: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == '__main__':
    main()