import os
import glob
import shutil
import numpy as np
import random
from tqdm import tqdm

# ================= 配置区域 =================
# DOTA 数据集标注路径
SOURCE_ANN_DIR = '/mnt/data/xiekaikai/split_ss_dota/trainval/labelTxt'

# [新增] 自动推断图片源路径 (假设在 ../images)
# DOTA 标准结构通常是 labelTxt 和 images 同级
SOURCE_IMG_DIR = os.path.join(os.path.dirname(SOURCE_ANN_DIR), 'images')

# 结果保存路径
TARGET_DIR = './1dataset/dt'
TARGET_ANN_DIR = os.path.join(TARGET_DIR, 'labelTxt') # 保持 DOTA 命名习惯
TARGET_IMG_DIR = os.path.join(TARGET_DIR, 'images')   # [新增] 图片保存路径

# 挑选数量
TOTAL_SAMPLES = 100

# DOTA 15 类别
CLASSES = ('plane', 'baseball-diamond', 'bridge', 'ground-track-field',
           'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
           'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout',
           'harbor', 'swimming-pool', 'helicopter')

# 【关键】用于计算透视分的“锚点类别”
ANCHOR_CLASSES = ['small-vehicle', 'large-vehicle', 'plane', 'storage-tank', 
                  'tennis-court', 'basketball-court', 'roundabout', 'helicopter']

# 图像尺寸
IMG_W, IMG_H = 1024, 1024

# 正则项
RIDGE_LAMBDA = 1e-4 
# ===========================================

def polygon_area(coords):
    """计算多边形面积"""
    x = coords[0::2]
    y = coords[1::2]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def fit_perspective_score_strict(objects):
    """
    计算透视强度 (仅基于锚点类别)
    """
    # 筛选锚点物体
    anchors = [o for o in objects if o['cls_name'] in ANCHOR_CLASSES]
    
    N = len(anchors)
    if N < 5: return 0.0 
    
    present_classes = sorted(list(set([o['cls_name'] for o in anchors])))
    K = len(present_classes)
    cls_to_idx = {name: i for i, name in enumerate(present_classes)}

    # 1. 准备数据
    Y = np.array([o['log_s'] for o in anchors])
    X_raw = np.array([o['x'] for o in anchors])
    Y_raw = np.array([o['y'] for o in anchors])
    
    # 归一化
    x_mean, x_std = X_raw.mean(), X_raw.std() + 1e-6
    y_mean, y_std = Y_raw.mean(), Y_raw.std() + 1e-6
    X_norm = (X_raw - x_mean) / x_std
    Y_norm = (Y_raw - y_mean) / y_std
    
    # 2. 构建矩阵
    A = np.zeros((N, 2 + K))
    A[:, 0] = X_norm
    A[:, 1] = Y_norm
    for i, obj in enumerate(anchors):
        col_idx = 2 + cls_to_idx[obj['cls_name']]
        A[i, col_idx] = 1

    # 3. 求解
    M = A.T @ A
    I_reg = np.eye(2 + K) * RIDGE_LAMBDA
    try:
        theta = np.linalg.inv(M + I_reg) @ (A.T @ Y)
        wx, wy = theta[0], theta[1]
        return np.sqrt(wx**2 + wy**2)
    except:
        return 0.0

def analyze_file(file_path):
    objects = []
    classes_in_img = set()
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 9: continue
            
            cls_name = parts[8]
            if cls_name not in CLASSES: continue
            
            classes_in_img.add(cls_name)
            
            try:
                coords = list(map(float, parts[:8]))
            except ValueError:
                continue
                
            area = polygon_area(coords)
            if area <= 1: continue
            
            s = np.sqrt(area)
            s = max(s, 1e-2)
            
            cx = sum(coords[0::2]) / 4.0
            cy = sum(coords[1::2]) / 4.0
            
            objects.append({
                'x': cx, 
                'y': cy, 
                'log_s': np.log(s), 
                'cls_name': cls_name
            })
            
    perspective_score = fit_perspective_score_strict(objects)
    
    return {
        'path': file_path,
        'filename': os.path.basename(file_path),
        'count': len(objects),
        'perspective_score': perspective_score,
        'classes': classes_in_img,
        'has_anchor': any(c in ANCHOR_CLASSES for c in classes_in_img)
    }

def copy_image_for_ann(ann_filename, source_img_dir, target_img_dir):
    """查找并复制对应的图片文件"""
    # DOTA 标注通常是 P0000.txt，图片是 P0000.png
    basename = os.path.splitext(ann_filename)[0]
    
    # DOTA 主要是 .png，但也可能是 .jpg
    possible_exts = ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']
    
    found = False
    for ext in possible_exts:
        img_name = basename + ext
        src_path = os.path.join(source_img_dir, img_name)
        
        if os.path.exists(src_path):
            dst_path = os.path.join(target_img_dir, img_name)
            shutil.copy(src_path, dst_path)
            found = True
            break
            
    if not found:
        # 调试用，找不到图片时打印警告
        # print(f"Warning: Image for {ann_filename} not found in {source_img_dir}")
        pass
    return found

def main():
    # 1. 初始化
    if os.path.exists(TARGET_DIR):
        print(f"清空旧目录: {TARGET_DIR}")
        shutil.rmtree(TARGET_DIR)
    
    # 创建标注和图片目录
    os.makedirs(TARGET_ANN_DIR)
    os.makedirs(TARGET_IMG_DIR)

    # 检查图片源目录
    if not os.path.exists(SOURCE_IMG_DIR):
        print(f"Error: 图片源目录不存在: {SOURCE_IMG_DIR}")
        print("请检查 SOURCE_IMG_DIR 路径设置是否正确 (默认在 labelTxt 同级的 images 目录)")
        return

    # 2. 扫描
    all_files = glob.glob(os.path.join(SOURCE_ANN_DIR, '*.txt'))
    print(f"[DOTA] 正在扫描 {len(all_files)} 个标注文件...")
    
    file_stats = []
    for f in tqdm(all_files):
        stat = analyze_file(f)
        file_stats.append(stat)

    selected_files = set()
    selected_stats = []

    def add_files(candidates, reason, limit):
        added = 0
        for item in candidates:
            if added >= limit: break
            if item['filename'] not in selected_files:
                # 1. 复制标注
                shutil.copy(item['path'], os.path.join(TARGET_ANN_DIR, item['filename']))
                
                # 2. [新增] 复制图片
                copy_image_for_ann(item['filename'], SOURCE_IMG_DIR, TARGET_IMG_DIR)
                
                selected_files.add(item['filename'])
                selected_stats.append(item)
                added += 1
        print(f"策略 [{reason}]: 选中 {added} 张")

    # 3. 分层挑选 (DOTA 特供版)

    # 3.1 极端透视组 (High Perspective) - Top 30
    valid_persp = [x for x in file_stats if x['has_anchor']]
    sorted_by_persp = sorted(valid_persp, key=lambda x: x['perspective_score'], reverse=True)
    add_files(sorted_by_persp, "极端透视 (Strong Perspective)", 30)

    # 3.2 垂直俯拍组 (Flat) - Top 20
    flat_candidates = [x for x in sorted_by_persp[::-1] if x['count'] > 10]
    add_files(flat_candidates, "垂直俯拍 (Flat)", 20)

    # 3.3 极度密集组 (High Density) - Top 20
    sorted_by_count = sorted(file_stats, key=lambda x: x['count'], reverse=True)
    add_files(sorted_by_count, "极度密集 (High Density)", 20)

    # 3.4 稀疏/长尾类别补齐 (Rare Classes) - Top 15
    rare_classes = ['helicopter', 'roundabout', 'soccer-ball-field', 'basketball-court']
    rare_candidates = [x for x in file_stats if any(c in rare_classes for c in x['classes'])]
    random.shuffle(rare_candidates)
    add_files(rare_candidates, "稀有类别补齐 (Rare Classes)", 15)

    # 3.5 随机补齐
    remaining_quota = TOTAL_SAMPLES - len(selected_files)
    if remaining_quota > 0:
        leftovers = [x for x in file_stats if x['filename'] not in selected_files]
        random.shuffle(leftovers)
        add_files(leftovers, "随机补齐", remaining_quota)

    # 4. 输出报告
    print("-" * 40)
    print(f"DOTA 黄金验证集生成完毕！")
    print(f"标注路径: {TARGET_ANN_DIR}")
    print(f"图片路径: {TARGET_IMG_DIR}")
    
    cls_counter = {}
    for s in selected_stats:
        for c in s['classes']:
            cls_counter[c] = cls_counter.get(c, 0) + 1
            
    print("\n类别覆盖统计:")
    for cls, cnt in sorted(cls_counter.items(), key=lambda x: x[1], reverse=True):
        print(f"  {cls}: {cnt} 张")

    with open(os.path.join(TARGET_DIR, 'val_list.txt'), 'w') as f:
        for item in selected_stats:
            f.write(item['filename'] + '\n')

if __name__ == '__main__':
    main()
