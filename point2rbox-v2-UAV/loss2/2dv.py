import os
import glob
import shutil
import numpy as np
import random
from tqdm import tqdm

# ================= 配置区域 =================
# DroneVehicle 数据集标注路径
SOURCE_ANN_DIR = '/mnt/data/xiekaikai/DroneVehicle/train/annfiles'

# [新增] 自动推断图片源路径 (假设在 ../images)
# 如果您的数据集结构特殊，请手动修改此路径
SOURCE_IMG_DIR = os.path.join(os.path.dirname(SOURCE_ANN_DIR), 'images')

# 结果保存路径
TARGET_DIR = './1dataset/dv'
TARGET_ANN_DIR = os.path.join(TARGET_DIR, 'annfiles')
TARGET_IMG_DIR = os.path.join(TARGET_DIR, 'images') # [新增] 图片保存路径

# 挑选数量
TOTAL_SAMPLES = 100

# DroneVehicle 类别定义
# 0:car, 1:bus, 2:truck, 3:van, 4:freight_car
CLASS_NAMES = ('car', 'bus', 'truck', 'van', 'freight_car')

# 严格对齐 Loss 的正则项
RIDGE_LAMBDA = 1e-4 
# ===========================================

def polygon_area(coords):
    """计算多边形面积"""
    x = coords[0::2]
    y = coords[1::2]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def fit_perspective_score_strict(objects):
    """
    【核心评分函数】
    使用与 PASCL v2.0 Loss 完全一致的矩阵逻辑计算 '透视强度'
    """
    N = len(objects)
    present_classes = sorted(list(set([o['cls_id'] for o in objects])))
    K = len(present_classes)
    
    # 样本太少评分为 0
    if N < K + 3:
        return 0.0

    cls_to_idx = {cid: i for i, cid in enumerate(present_classes)}

    # 1. 准备数据
    Y = np.array([o['log_s'] for o in objects])
    X_raw = np.array([o['x'] for o in objects])
    Y_raw = np.array([o['y'] for o in objects])
    
    # Z-Score 归一化 (自适应任意图像尺寸)
    x_mean, x_std = X_raw.mean(), X_raw.std() + 1e-6
    y_mean, y_std = Y_raw.mean(), Y_raw.std() + 1e-6
    
    X_norm = (X_raw - x_mean) / x_std
    Y_norm = (Y_raw - y_mean) / y_std
    
    # 2. 构建设计矩阵 A
    # A: [x_norm, y_norm, one_hot_intercepts...]
    A = np.zeros((N, 2 + K))
    A[:, 0] = X_norm
    A[:, 1] = Y_norm
    
    for i, obj in enumerate(objects):
        col_idx = 2 + cls_to_idx[obj['cls_id']]
        A[i, col_idx] = 1 

    # 3. 矩阵求解 (Ridge Regression)
    M = A.T @ A
    I_reg = np.eye(2 + K) * RIDGE_LAMBDA
    
    try:
        theta = np.linalg.inv(M + I_reg) @ (A.T @ Y)
        wx_norm = theta[0]
        wy_norm = theta[1]
        
        # 评分：斜率模长
        score = np.sqrt(wx_norm**2 + wy_norm**2)
        return score
        
    except np.linalg.LinAlgError:
        return 0.0

def analyze_file(file_path):
    """读取 DroneVehicle 格式标注文件"""
    objects = []
    classes_in_img = set()
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            # DroneVehicle 格式: x1 y1 ... x4 y4 class_id
            if len(parts) < 9: continue
            
            try:
                # 解析类别 ID
                cls_id = int(parts[8])
                if cls_id < 0 or cls_id >= len(CLASS_NAMES): continue
                
                cls_name = CLASS_NAMES[cls_id]
                classes_in_img.add(cls_name)
                
                # 解析坐标
                coords = list(map(float, parts[:8]))
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
                    'cls_id': cls_id, 
                    'cls_name': cls_name
                })
            except ValueError:
                continue
            
    # 计算透视分
    perspective_score = fit_perspective_score_strict(objects)
    
    return {
        'path': file_path,
        'filename': os.path.basename(file_path),
        'count': len(objects),
        'perspective_score': perspective_score,
        'classes': classes_in_img
    }

def copy_image_for_ann(ann_filename, source_img_dir, target_img_dir):
    """根据标注文件名查找并复制对应的图片"""
    # DroneVehicle 通常是 00001.txt 对应 00001.jpg
    basename = os.path.splitext(ann_filename)[0]
    
    # 常见后缀列表，优先匹配 jpg (DroneVehicle 常用)
    possible_exts = ['.jpg', '.png', '.jpeg', '.bmp', '.tif', '.tiff']
    
    found = False
    for ext in possible_exts:
        img_name = basename + ext
        src_path = os.path.join(source_img_dir, img_name)
        
        # 有时候 DroneVehicle 可能会有 _vis 或 _inf 后缀，这里假设切图后名字一致
        if os.path.exists(src_path):
            dst_path = os.path.join(target_img_dir, img_name)
            shutil.copy(src_path, dst_path)
            found = True
            break
            
    if not found:
        # 尝试调试信息
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
        print("请检查 SOURCE_IMG_DIR 设置是否正确")
        return

    # 2. 扫描
    all_files = glob.glob(os.path.join(SOURCE_ANN_DIR, '*.txt'))
    print(f"[DroneVehicle] 正在扫描 {len(all_files)} 个标注文件...")
    
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
                has_img = copy_image_for_ann(item['filename'], SOURCE_IMG_DIR, TARGET_IMG_DIR)
                
                selected_files.add(item['filename'])
                selected_stats.append(item)
                added += 1
        print(f"策略 [{reason}]: 选中 {added} 张")

    # 3. 分层挑选 (DroneVehicle 特供版)
    
    # 3.1 极端透视组 (High Perspective) - Top 30
    sorted_by_persp = sorted(file_stats, key=lambda x: x['perspective_score'], reverse=True)
    add_files(sorted_by_persp, "极端透视 (Strong Perspective)", 30)

    # 3.2 垂直俯拍组 (Flat/Orthogonal) - Top 20
    # 且要求车辆有一定数量，排除空图
    flat_candidates = [x for x in sorted_by_persp[::-1] if x['count'] > 5]
    add_files(flat_candidates, "垂直俯拍 (Flat)", 20)

    # 3.3 极度密集组 (High Density) - Top 20
    # DroneVehicle 有些图非常密集
    sorted_by_count = sorted(file_stats, key=lambda x: x['count'], reverse=True)
    add_files(sorted_by_count, "极度密集 (High Density)", 20)

    # 3.4 稀疏困难组 (Sparse) - Top 15
    # 只有 3-8 辆车的图
    sparse_candidates = [x for x in file_stats if 3 <= x['count'] <= 8]
    random.shuffle(sparse_candidates)
    add_files(sparse_candidates, "稀疏困难 (Sparse)", 15)

    # 3.5 补齐剩余
    remaining_quota = TOTAL_SAMPLES - len(selected_files)
    if remaining_quota > 0:
        leftovers = [x for x in file_stats if x['filename'] not in selected_files]
        random.shuffle(leftovers)
        add_files(leftovers, "随机补齐", remaining_quota)

    # 4. 输出报告
    print("-" * 40)
    print(f"DroneVehicle 黄金验证集生成完毕！")
    print(f"标注保存路径: {TARGET_ANN_DIR}")
    print(f"图片保存路径: {TARGET_IMG_DIR}")
    
    cls_counter = {}
    for s in selected_stats:
        for c in s['classes']:
            cls_counter[c] = cls_counter.get(c, 0) + 1
            
    print("\n类别覆盖统计:")
    for cls, cnt in sorted(cls_counter.items(), key=lambda x: x[1], reverse=True):
        print(f"  {cls}: {cnt} 张")

    # 生成列表文件
    with open(os.path.join(TARGET_DIR, 'val_list.txt'), 'w') as f:
        for item in selected_stats:
            f.write(item['filename'] + '\n')

if __name__ == '__main__':
    main()