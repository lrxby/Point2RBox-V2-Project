import os
import numpy as np
import cv2
import math

# ================= 配置区域 =================
DATASET_PATH = "/mnt/data/xiekaikai/split_ss_dota"
# DOTA 类别定义
CLASSES = ('plane', 'baseball-diamond', 'bridge', 'ground-track-field',
           'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
           'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout',
           'harbor', 'swimming-pool', 'helicopter')

# 验证参数
MIN_INSTANCES = 3  # 单图单类最少实例数，少于此无法统计一致性

def poly_to_area(poly):
    """将8点坐标转换为面积"""
    points = np.array(poly).reshape((4, 2)).astype(np.float32)
    (cx, cy), (w, h), angle = cv2.minAreaRect(points)
    return w * h, cy

def calculate_layout_loss_static(areas):
    """
    计算静态一致性 Loss (Static Mode)
    基准：组内面积均值
    Loss：相对 L1 误差 (|A - Mean| / Mean)
    """
    if len(areas) < MIN_INSTANCES:
        return []
    
    areas_np = np.array(areas)
    mean_area = np.mean(areas_np)
    
    # 防止除零
    if mean_area == 0: return []
    
    # 计算相对误差：即每个框偏离均值百分之多少
    # 比如 area=110, mean=100 -> loss=0.1 (10%)
    losses = np.abs(areas_np - mean_area) / mean_area
    return losses.tolist()

def main():
    ann_dir = os.path.join(DATASET_PATH, 'trainval', 'annfiles')
    if not os.path.exists(ann_dir):
        ann_dir = os.path.join(DATASET_PATH, 'train', 'annfiles')
        
    print(f"[*] 正在分析 DOTA 数据集: {ann_dir}")
    print(f"[*] 模式: Static (Intra-Image Mean Consistency)")
    
    # 存储每个类别的所有 Loss 值
    cat_losses = {c: [] for c in CLASSES}
    
    files = [f for f in os.listdir(ann_dir) if f.endswith('.txt')]
    total_files = len(files)
    
    for i, filename in enumerate(files):
        if i % 1000 == 0: print(f"    处理进度: {i}/{total_files}...")
        
        # 1. 读取单张图片的标注
        img_instances = {c: [] for c in CLASSES} # {class: [area1, area2...]}
        
        with open(os.path.join(ann_dir, filename), 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 9: continue
                
                # 解析类别 (DOTA 格式: x1...y4 category difficulty)
                cat_name = parts[8]
                if cat_name not in CLASSES: continue
                
                try:
                    poly = [float(x) for x in parts[:8]]
                    area, cy = poly_to_area(poly)
                    if area > 1: # 过滤掉极小噪点
                        img_instances[cat_name].append(area)
                except:
                    continue
        
        # 2. 计算该图的 Loss 并存入总表
        for cat, areas in img_instances.items():
            losses = calculate_layout_loss_static(areas)
            cat_losses[cat].extend(losses)

    # 3. 输出统计报表
    print_statistics(cat_losses)

def print_statistics(cat_losses):
    print("\n" + "="*145)
    headers = ["Class", "Count", "Mean", "Min", "Max", "P10", "P20", "P30", "P40", "P50", "P60", "P70", "P80", "P90"]
    # 格式化打印头
    header_str = " | ".join([f"{h:<18}" if h == "Class" else f"{h:<8}" for h in headers])
    print(header_str)
    print("-" * 145)
    
    for cat in CLASSES:
        losses = cat_losses[cat]
        if len(losses) == 0:
            print(f"{cat:<18} | {'N/A':<8}")
            continue
            
        losses_np = np.array(losses)
        # 过滤掉极端异常值 (如 > 10.0 的 Loss，通常是标注错误) 避免拉偏均值
        # losses_np = losses_np[losses_np < 10.0] 
        
        count = len(losses_np)
        mean_val = np.mean(losses_np)
        min_val = np.min(losses_np)
        max_val = np.max(losses_np)
        percentiles = np.percentile(losses_np, [10, 20, 30, 40, 50, 60, 70, 80, 90])
        
        row_data = [cat, count, mean_val, min_val, max_val, *percentiles]
        
        # 格式化打印行
        row_str = f"{row_data[0]:<18} | {row_data[1]:<8} | {row_data[2]:.4f}   | {row_data[3]:.4f}   | {row_data[4]:.4f}   |"
        for p in row_data[5:]:
            row_str += f" {p:.4f}   |"
        print(row_str)
    print("="*145 + "\n")

if __name__ == "__main__":
    main()
