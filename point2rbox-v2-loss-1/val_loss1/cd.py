import os
import numpy as np
import cv2
from sklearn.linear_model import LinearRegression

# ================= 配置区域 =================
DATASET_PATH = "/mnt/data/xiekaikai/split_ss_codrone"
CLASSES = ('car', 'truck', 'bus', 'traffic-light', 'traffic-sign', 'bridge', 
           'people', 'bicycle', 'motor', 'tricycle', 'boat', 'ship')
MIN_INSTANCES = 5

def poly_to_data(poly):
    points = np.array(poly).reshape((4, 2)).astype(np.float32)
    (cx, cy), (w, h), angle = cv2.minAreaRect(points)
    return w * h, cy

def calculate_layout_loss_dynamic(data_list):
    if len(data_list) < MIN_INSTANCES: return []
    
    areas = np.array([d['area'] for d in data_list])
    ys = np.array([d['y'] for d in data_list]).reshape(-1, 1)
    
    # 线性回归拟合透视关系
    try:
        reg = LinearRegression().fit(ys, areas)
        targets = reg.predict(ys)
        # 负值修正
        if np.any(targets <= 0):
            targets = np.full_like(areas, np.mean(areas))
    except:
        targets = np.full_like(areas, np.mean(areas))
        
    targets = np.maximum(targets, 1.0)
    losses = np.abs(areas - targets) / targets
    return losses.tolist()

def main():
    ann_dir = os.path.join(DATASET_PATH, 'trainval', 'annfiles')
    if not os.path.exists(ann_dir):
        ann_dir = os.path.join(DATASET_PATH, 'train', 'annfiles')
        
    print(f"[*] 正在分析 CODrone 数据集: {ann_dir}")
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
                
                # CODrone 格式: x1...y4 category
                cat_name = parts[8]
                if cat_name not in CLASSES: continue
                
                try:
                    poly = [float(x) for x in parts[:8]]
                    area, cy = poly_to_data(poly)
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
        
        # 这里的 Mean Loss 直接反映了GT框与我们假设的先验分布的偏差
        # Mean 越小，说明 GT 越符合先验，我们的 Loss 设计就越合理
        row = [cat, len(losses), np.mean(losses), np.min(losses), np.max(losses)]
        row.extend(np.percentile(losses, range(10, 100, 10)))
        
        print(f"{row[0]:<18} | {row[1]:<8} | {row[2]:.4f}   | {row[3]:.4f}   | {row[4]:.4f}   | " + "   | ".join([f"{x:.4f}" for x in row[5:]]) + "   |")
    print("="*145 + "\n")

if __name__ == "__main__":
    main()
