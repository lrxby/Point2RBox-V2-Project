import os
import shutil
import numpy as np
import cv2
import glob
from tqdm import tqdm
from collections import defaultdict, Counter
import math
import random

# ================= 配置区域 (Configuration) =================
# 1. 源数据路径 (根据你的实际路径修改)
# 标注文件路径
SOURCE_LABEL_DIR = '/mnt/data/xiekaikai/DroneVehicle/train/annfiles'
# 图片文件路径
SOURCE_IMAGE_DIR = '/mnt/data/xiekaikai/DroneVehicle/train/images' 
IMAGE_EXT = '.jpg' # DroneVehicle 通常是 .jpg

# 2. 输出路径 (脚本会自动创建)
TARGET_DIR = './2dataset/dv'

# 3. 筛选配额 (总计 100 张)
QUOTA = {
    'high_chaos': 20,   # 最乱 (路口/拥堵)
    'low_chaos': 20,    # 最齐 (停车场/高速)
    'dense': 15,        # 最密
    'sparse': 10,       # 最疏 (乡村道路)
    'class_cover': 25,  # 类别覆盖 (5类 * 5张)
    'random': 10        # 随机补充
}

# 4. 类别映射 (根据数据集定义)
CLASS_MAP = {
    0: 'car',
    1: 'bus',
    2: 'truck',
    3: 'van',
    4: 'freight_car'
}
# ==========================================================

def parse_drone_label(label_path):
    """
    解析 DroneVehicle 标注格式: 
    x1 y1 x2 y2 x3 y3 x4 y4 category
    """
    objects = []
    try:
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 9: continue
                
                # 提取坐标
                poly = np.array([float(x) for x in parts[:8]]).reshape(4, 2)
                # 提取类别 ID (最后一位)
                class_id = int(parts[8])
                class_name = CLASS_MAP.get(class_id, 'unknown')
                
                # 计算旋转矩形属性
                rect = cv2.minAreaRect(poly.astype(np.float32))
                (cx, cy), (w, h), angle = rect
                scale = math.sqrt(w * h)
                
                # 统一角度逻辑 (长边定义)
                if w < h: angle += 90
                theta_rad = np.deg2rad(angle)
                
                objects.append({
                    'cls': class_name,
                    'cls_id': class_id,
                    'center': np.array([cx, cy]),
                    'theta': theta_rad,
                    'scale': scale
                })
    except Exception as e:
        print(f"Error parsing {label_path}: {e}")
        return []
        
    return objects

def calculate_chaos(objects, k_radius=2.0):
    """
    计算混乱度 (复用 4*theta 逻辑)
    """
    if len(objects) < 5: return 0.0 # 目标太少，不算混乱度
    
    centers = np.array([o['center'] for o in objects])
    thetas = np.array([o['theta'] for o in objects])
    scales = np.array([o['scale'] for o in objects])
    
    # 使用 4*theta (让水平和垂直等价，适合车辆)
    vecs = np.stack([np.cos(4*thetas), np.sin(4*thetas)], axis=1)
    
    # 距离矩阵
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=-1)
    
    # 邻域权重
    sigmas = np.clip(scales * k_radius, 16.0, 800.0)
    sigma_mat = sigmas[:, np.newaxis]
    W = np.exp(-dist_sq / (2 * sigma_mat**2))
    
    # 归一化 (去除自环以获得更客观的群体指标)
    np.fill_diagonal(W, 0)
    W_sum = np.sum(W, axis=1, keepdims=True) + 1e-6
    W_norm = W / W_sum
    
    # 聚合
    R = np.dot(W_norm, vecs)
    R_norm = np.linalg.norm(R, axis=1)
    
    # 平均混乱度
    avg_chaos = np.mean(1.0 - R_norm)
    return avg_chaos

def main():
    # 1. 初始化
    if os.path.exists(TARGET_DIR):
        print(f"Warning: {TARGET_DIR} 已存在，清空中...")
        shutil.rmtree(TARGET_DIR)
    
    # [修正点] 创建 images 文件夹
    os.makedirs(os.path.join(TARGET_DIR, 'images'))      # 图片目录
    os.makedirs(os.path.join(TARGET_DIR, 'annfiles'))    # 标注目录
    
    # 2. 扫描文件
    label_files = glob.glob(os.path.join(SOURCE_LABEL_DIR, '*.txt'))
    print(f"Found {len(label_files)} annotation files. Scanning metadata...")
    
    metadata = []
    for lp in tqdm(label_files):
        objs = parse_drone_label(lp)
        # 过滤空文件或极少目标的文件
        if len(objs) < 3: continue
        
        chaos = calculate_chaos(objs)
        
        # 统计主要类别
        cls_counts = Counter([o['cls'] for o in objs])
        primary_cls = cls_counts.most_common(1)[0][0]
        
        metadata.append({
            'path': lp,
            'filename': os.path.basename(lp),
            'num_objs': len(objs),
            'chaos_score': chaos,
            'primary_cls': primary_cls
        })
        
    print(f"Valid files for selection: {len(metadata)}")
    
    # 3. 智能筛选
    selected_files = set()
    final_selection = []
    
    def add_item(item, reason):
        if item['path'] not in selected_files:
            selected_files.add(item['path'])
            final_selection.append((item, reason))
            return True
        return False

    # --- 策略 A: High Chaos (最乱) ---
    # 必须有一定数量的目标，乱才有意义
    valid_chaos = [m for m in metadata if m['num_objs'] > 10]
    valid_chaos.sort(key=lambda x: x['chaos_score'], reverse=True)
    
    c = 0
    for m in valid_chaos:
        if c >= QUOTA['high_chaos']: break
        if add_item(m, f"High Chaos ({m['chaos_score']:.2f})"): c += 1

    # --- 策略 B: Low Chaos (最齐) ---
    valid_chaos.sort(key=lambda x: x['chaos_score']) # 升序
    c = 0
    for m in valid_chaos:
        if c >= QUOTA['low_chaos']: break
        # 排除 0 分是因为没邻居的情况
        if m['chaos_score'] < 0.001 and m['num_objs'] < 5: continue
        if add_item(m, f"Low Chaos ({m['chaos_score']:.2f})"): c += 1

    # --- 策略 C: High Density (最密) ---
    metadata.sort(key=lambda x: x['num_objs'], reverse=True)
    c = 0
    for m in metadata:
        if c >= QUOTA['dense']: break
        if add_item(m, f"High Density ({m['num_objs']} objs)"): c += 1

    # --- 策略 D: Sparse (稀疏) ---
    # 找 5~15 个目标的图
    sparse = [m for m in metadata if 5 <= m['num_objs'] <= 15]
    random.shuffle(sparse)
    c = 0
    for m in sparse:
        if c >= QUOTA['sparse']: break
        if add_item(m, f"Sparse ({m['num_objs']} objs)"): c += 1

    # --- 策略 E: Class Balance (类别覆盖) ---
    # 确保每个类别（尤其是 freight_car, bus）都有代表作
    for cls_name in CLASS_MAP.values():
        candidates = [m for m in metadata if m['primary_cls'] == cls_name]
        candidates.sort(key=lambda x: x['num_objs'], reverse=True) # 选该类目标多的
        
        limit = 5 # 每类选5张
        c = 0
        for m in candidates:
            if c >= limit: break
            if add_item(m, f"Class: {cls_name}"): c += 1

    # --- 策略 F: Random Fill (随机补位) ---
    remain = 100 - len(final_selection)
    if remain > 0:
        others = [m for m in metadata if m['path'] not in selected_files]
        random.shuffle(others)
        for i in range(min(remain, len(others))):
            add_item(others[i], "Random Fill")

    # 4. 复制文件
    print(f"Selected {len(final_selection)} images. Copying...")
    
    log_path = os.path.join(TARGET_DIR, 'selection_log.txt')
    with open(log_path, 'w') as log:
        log.write(f"Selection Log - Total: {len(final_selection)}\n")
        log.write("="*60 + "\n")
        
        for item, reason in tqdm(final_selection):
            # 复制 txt
            shutil.copy(item['path'], os.path.join(TARGET_DIR, 'annfiles', item['filename']))
            
            # 复制 img
            img_name = item['filename'].replace('.txt', IMAGE_EXT)
            src_img = os.path.join(SOURCE_IMAGE_DIR, img_name)
            
            # 检查图片是否存在 (DroneVehicle 有时候文件名大小写敏感或后缀不同)
            # 如果配置的是 .jpg 但实际是 .png，这里尝试兼容
            if not os.path.exists(src_img):
                # 尝试找同名 png
                src_img_png = src_img.replace('.jpg', '.png')
                if os.path.exists(src_img_png):
                    src_img = src_img_png
                    img_name = img_name.replace('.jpg', '.png')
            
            if os.path.exists(src_img):
                # [修正点] 复制到 images 目录
                shutil.copy(src_img, os.path.join(TARGET_DIR, 'images', img_name))
            else:
                print(f"[Warn] Image not found: {src_img}")
                log.write(f"[MISSING IMG] ")
            
            log.write(f"{item['filename']:<15} | {reason}\n")
            
    print(f"\nDone! Saved to {TARGET_DIR}")

if __name__ == '__main__':
    main()