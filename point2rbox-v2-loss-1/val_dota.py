import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def poly_to_obb(poly):
    """将DOTA的8参数格式 [x1, y1, ..., x4, y4] 转换为 (cx, cy, w, h)"""
    points = np.array(poly).reshape((4, 2)).astype(np.float32)
    # 使用OpenCV计算最小外接矩形
    (cx, cy), (w, h), angle = cv2.minAreaRect(points)
    # 统一约定：width为长边，height为短边，消除旋转导致的长宽定义颠倒
    width = max(w, h)
    height = min(w, h)
    return cx, cy, width, height

def visualize_example(cat_data, target_cat, save_path):
    """
    生成并保存指定类别的尺寸随坐标分布的散点图
    """
    if target_cat not in cat_data: 
        print(f"找不到类别: {target_cat}")
        return
    
    # 找一个该类别实例数量最多的图片进行展示
    from collections import Counter
    img_counts = Counter([x['img_id'] for x in cat_data[target_cat]])
    if not img_counts: return
    
    best_img_id = img_counts.most_common(1)[0][0]
    group = [x for x in cat_data[target_cat] if x['img_id'] == best_img_id]
    
    ys = np.array([obj['cy'] for obj in group])
    ws = np.array([obj['w'] for obj in group])
    
    plt.figure(figsize=(10, 6))
    plt.scatter(ys, ws, color='steelblue', alpha=0.7, label=f'Instances (N={len(group)})')
    
    # 绘制线性回归趋势线
    if len(ys) > 1:
        # 简单的一元线性拟合
        z = np.polyfit(ys, ws, 1)
        p = np.poly1d(z)
        plt.plot(ys, p(ys), "r--", linewidth=2, label='Linear Trend Line')

    plt.title(f"Size vs Y-coord: {target_cat} in Image {best_img_id}")
    plt.xlabel("Y-coordinate (Pixels) - Position in Image")
    plt.ylabel("Object Long-side Width (Pixels)")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # 保存图片到指定路径
    plt.savefig(save_path)
    print(f"\n[可视化] 示例图分析已保存至: {save_path}")
    plt.close()

def verify_dota_priors(dataset_path, save_path):
    """
    核心统计逻辑：计算CV值和R2拟合度
    """
    ann_dir = os.path.join(dataset_path, 'trainval', 'annfiles')
    if not os.path.exists(ann_dir):
        print(f"错误: 路径不存在 {ann_dir}")
        return

    cat_data = {} # 存储解析后的数据
    print(f"开始解析标注文件: {ann_dir}")
    
    files = [f for f in os.listdir(ann_dir) if f.endswith('.txt')]
    for filename in files:
        img_id = filename.split('.')[0]
        with open(os.path.join(ann_dir, filename), 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) < 10: continue
                
                # 解析坐标和类别
                poly = [float(x) for x in parts[:8]]
                category = parts[8]
                cx, cy, w, h = poly_to_obb(poly)
                
                if category not in cat_data:
                    cat_data[category] = []
                cat_data[category].append({'img_id': img_id, 'cx': cx, 'cy': cy, 'w': w, 'h': h})

    print(f"\n" + "="*30)
    print(f"{'Category':<20} | {'Count':<8} | {'Avg CV':<8} | {'Avg R2':<8}")
    print("-" * 55)
    
    for cat, items in cat_data.items():
        # 按图片分组统计
        img_groups = {}
        for item in items:
            iid = item['img_id']
            if iid not in img_groups: img_groups[iid] = []
            img_groups[iid].append(item)
        
        cv_list = []
        r2_list = []
        
        for iid, group in img_groups.items():
            if len(group) < 5: continue # 物体太少不参与统计
            
            ws = np.array([obj['w'] for obj in group])
            ys = np.array([obj['cy'] for obj in group]).reshape(-1, 1)
            
            # 1. 计算变异系数 (CV = std/mean)
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
            print(f"{cat:<20} | {len(items):<8} | {avg_cv:.4f} | {avg_r2:.4f}")

    # 生成可视化分析图
    visualize_example(cat_data, 'small-vehicle', save_path)

if __name__ == "__main__":
    # --- 配置区域 ---
    DATASET_PATH = "/mnt/data/xiekaikai/split_ss_dota"
    # 指定图片保存的绝对路径
    SAVE_PATH = "/mnt/data/liurunxiang/workplace/point2rbox-v2-loss-new/size_analysis.png"
    # ----------------
    
    verify_dota_priors(DATASET_PATH, SAVE_PATH)