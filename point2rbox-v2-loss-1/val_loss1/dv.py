import os
import numpy as np
import cv2
from sklearn.linear_model import LinearRegression

# ================= 配置区域 =================
DATASET_PATH = "/mnt/data/xiekaikai/DroneVehicle"
CLASSES = ('car', 'bus', 'truck', 'van', 'freight_car')
MIN_INSTANCES = 5  # 线性回归至少需要一定数量的点

def poly_to_data(poly):
    """提取面积和Y坐标"""
    points = np.array(poly).reshape((4, 2)).astype(np.float32)
    (cx, cy), (w, h), angle = cv2.minAreaRect(points)
    return w * h, cy

def calculate_layout_loss_dynamic(data_list):
    """
    计算动态布局 Loss (Dynamic Mode)
    基准：线性回归 f(y) = ay + b
    Loss：相对 L1 误差
    """
    if len(data_list) < MIN_INSTANCES:
        return []
    
    areas = np.array([d['area'] for d in data_list])
    ys = np.array([d['y'] for d in data_list]).reshape(-1, 1)
    
    # 尝试线性拟合
    try:
        reg = LinearRegression().fit(ys, areas)
        targets = reg.predict(ys)
        
        # 物理约束：目标面积不能为负。如果回归出负数（极端情况），回退到均值
        if np.any(targets <= 0):
            target_val = np.mean(areas)
            targets = np.full_like(areas, target_val)
    except:
        target_val = np.mean(areas)
        targets = np.full_like(areas, target_val)
        
    # 计算 Loss
    # 避免 target 接近 0 导致除零
    targets = np.maximum(targets, 1.0) 
    losses = np.abs(areas - targets) / targets
    return losses.tolist()

def main():
    ann_dir = os.path.join(DATASET_PATH, 'train', 'annfiles')
    if not os.path.exists(ann_dir):
        print(f"错误: 路径不存在 {ann_dir}")
        return
        
    print(f"[*] 正在分析 DroneVehicle 数据集: {ann_dir}")
    print(f"[*] 模式: Dynamic (Perspective Linear Regression)")
    
    cat_losses = {c: [] for c in CLASSES}
    files = [f for f in os.listdir(ann_dir) if f.endswith('.txt')]
    
    for i, filename in enumerate(files):
        if i % 1000 == 0: print(f"    处理进度: {i}/{len(files)}...")
        
        img_data = {c: [] for c in CLASSES}
        
        with open(os.path.join(ann_dir, filename), 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) < 9: continue
                
                # DV 格式: x1...y4 class_idx
                try:
                    idx = int(parts[8])
                    if idx >= len(CLASSES): continue
                    cat_name = CLASSES[idx]
                    
                    poly = [float(x) for x in parts[:8]]
                    area, cy = poly_to_data(poly)
                    if area > 1:
                        img_data[cat_name].append({'area': area, 'y': cy})
                except:
                    continue
        
        for cat, data in img_data.items():
            losses = calculate_layout_loss_dynamic(data)
            cat_losses[cat].extend(losses)

    print_statistics(cat_losses)

def print_statistics(cat_losses):
    print("\n" + "="*145)
    headers = ["Class", "Count", "Mean", "Min", "Max", "P10", "P20", "P30", "P40", "P50", "P60", "P70", "P80", "P90"]
    print(" | ".join([f"{h:<18}" if h == "Class" else f"{h:<8}" for h in headers]))
    print("-" * 145)
    
    for cat in CLASSES:
        losses = np.array(cat_losses[cat])
        if len(losses) == 0: continue
        
        row = [cat, len(losses), np.mean(losses), np.min(losses), np.max(losses)]
        row.extend(np.percentile(losses, range(10, 100, 10)))
        
        print(f"{row[0]:<18} | {row[1]:<8} | {row[2]:.4f}   | {row[3]:.4f}   | {row[4]:.4f}   | " + "   | ".join([f"{x:.4f}" for x in row[5:]]) + "   |")
    print("="*145 + "\n")

if __name__ == "__main__":
    main()
