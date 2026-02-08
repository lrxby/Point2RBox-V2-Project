import os
import shutil
import numpy as np
import cv2
import glob
from tqdm import tqdm
from collections import defaultdict, Counter
import math

# ================= 配置区域 =================
# 1. 源数据路径
SOURCE_LABEL_DIR = '/mnt/data/xiekaikai/split_ss_dota/trainval/labelTxt'
SOURCE_IMAGE_DIR = '/mnt/data/xiekaikai/split_ss_dota/trainval/images'
IMAGE_EXT = '.png' 

# 2. 输出路径 (会自动创建)
TARGET_DIR = './2dataset/dt'

# 3. 筛选数量配置
TOTAL_TARGET = 100
QUOTA = {
    'high_chaos': 15,   # 最乱
    'low_chaos': 15,    # 最齐
    'dense': 10,        # 最密
    'sparse': 10,       # 最疏
    'large_scale': 10,  # 巨大物体
    'class_cover': 30,  # 类别覆盖 (保底)
    'random': 10        # 随机补位
}

# 4. 类别列表
CLASSES = (
    'plane', 'baseball-diamond', 'bridge', 'ground-track-field',
    'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
    'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout',
    'harbor', 'swimming-pool', 'helicopter'
)
# ===========================================

def parse_label(label_path):
    """解析单个标签文件，提取元数据"""
    objects = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 9: continue
            
            # 解析坐标
            poly = np.array([float(x) for x in parts[:8]]).reshape(4, 2)
            cls = parts[8]
            
            # 计算旋转矩形属性
            rect = cv2.minAreaRect(poly.astype(np.float32))
            (cx, cy), (w, h), angle = rect
            scale = math.sqrt(w * h)
            
            # 统一角度到 0-180 或 4theta 空间
            if w < h: angle += 90
            theta_rad = np.deg2rad(angle)
            
            objects.append({
                'cls': cls,
                'center': np.array([cx, cy]),
                'theta': theta_rad,
                'scale': scale,
                'area': w * h
            })
    return objects

def calculate_chaos(objects, k_radius=2.0):
    """
    计算一张图片的平均混乱度 (Chaos Score)
    复用 NCL 中的逻辑: Variance = 1 - ||Mean Vector||
    """
    if len(objects) < 5: return 0.0 # 太少的不算混乱度
    
    centers = np.array([o['center'] for o in objects])
    thetas = np.array([o['theta'] for o in objects])
    scales = np.array([o['scale'] for o in objects])
    
    # 使用 4*theta (解决平行问题)
    vecs = np.stack([np.cos(4*thetas), np.sin(4*thetas)], axis=1)
    
    # 距离矩阵
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=-1)
    
    # 邻域权重
    sigmas = np.clip(scales * k_radius, 16.0, 800.0)
    sigma_mat = sigmas[:, np.newaxis]
    W = np.exp(-dist_sq / (2 * sigma_mat**2))
    
    # 归一化
    np.fill_diagonal(W, 0) # 计算全图混乱度时，去掉自环可能更客观，或者保留也行
    W_sum = np.sum(W, axis=1, keepdims=True) + 1e-6
    W_norm = W / W_sum
    
    # 聚合
    R = np.dot(W_norm, vecs)
    R_norm = np.linalg.norm(R, axis=1)
    
    # 单图平均混乱度
    avg_chaos = np.mean(1.0 - R_norm)
    return avg_chaos

def main():
    # 1. 初始化目录
    if os.path.exists(TARGET_DIR):
        print(f"Warning: {TARGET_DIR} 已存在，正在清空...")
        shutil.rmtree(TARGET_DIR)
    
    os.makedirs(os.path.join(TARGET_DIR, 'images'))
    os.makedirs(os.path.join(TARGET_DIR, 'labelTxt'))
    
    # 2. 扫描所有文件
    label_files = glob.glob(os.path.join(SOURCE_LABEL_DIR, '*.txt'))
    print(f"开始扫描 {len(label_files)} 个文件...")
    
    metadata = []
    
    for lp in tqdm(label_files):
        objs = parse_label(lp)
        if not objs: continue
        
        # 统计信息
        num_objs = len(objs)
        avg_scale = np.mean([o['scale'] for o in objs])
        chaos_score = calculate_chaos(objs)
        
        # 类别统计
        cls_counts = Counter([o['cls'] for o in objs])
        primary_cls = cls_counts.most_common(1)[0][0] # 数量最多的类别
        
        metadata.append({
            'path': lp,
            'filename': os.path.basename(lp),
            'num_objs': num_objs,
            'avg_scale': avg_scale,
            'chaos_score': chaos_score,
            'primary_cls': primary_cls,
            'cls_set': set(cls_counts.keys())
        })
        
    print("扫描完成，开始智能筛选...")
    
    selected_files = set() # 记录已选文件路径，防止重复
    final_selection = []   # 记录筛选结果及理由
    
    def add_selection(item, reason):
        if item['path'] not in selected_files:
            selected_files.add(item['path'])
            final_selection.append((item, reason))
            return True
        return False

    # --- 策略 1: 极端混乱 (High Chaos) ---
    # 过滤掉只有极少数目标的图，那些混乱度没意义
    valid_chaos = [m for m in metadata if m['num_objs'] > 10]
    valid_chaos.sort(key=lambda x: x['chaos_score'], reverse=True) # 降序
    
    count = 0
    for m in valid_chaos:
        if count >= QUOTA['high_chaos']: break
        if add_selection(m, f"High Chaos (Score: {m['chaos_score']:.2f})"):
            count += 1
            
    # --- 策略 2: 极端整齐 (Low Chaos) ---
    valid_chaos.sort(key=lambda x: x['chaos_score']) # 升序
    count = 0
    for m in valid_chaos:
        if count >= QUOTA['low_chaos']: break
        # 排除混乱度为0是因为没邻居的情况
        if m['chaos_score'] < 0.001 and m['num_objs'] < 5: continue 
        if add_selection(m, f"Low Chaos (Score: {m['chaos_score']:.2f})"):
            count += 1

    # --- 策略 3: 极端密集 (Dense) ---
    metadata.sort(key=lambda x: x['num_objs'], reverse=True)
    count = 0
    for m in metadata:
        if count >= QUOTA['dense']: break
        if add_selection(m, f"High Density ({m['num_objs']} objects)"):
            count += 1
            
    # --- 策略 4: 稀疏但非空 (Sparse) ---
    # 找 obj 在 [5, 20] 之间的
    sparse_candidates = [m for m in metadata if 5 <= m['num_objs'] <= 20]
    # 随机一点
    import random
    random.shuffle(sparse_candidates)
    count = 0
    for m in sparse_candidates:
        if count >= QUOTA['sparse']: break
        if add_selection(m, f"Sparse ({m['num_objs']} objects)"):
            count += 1

    # --- 策略 5: 类别覆盖 (Class Cover) ---
    # 确保每个类别至少有几张代表作
    for cls in CLASSES:
        # 找以该类别为主的图片
        candidates = [m for m in metadata if m['primary_cls'] == cls]
        # 按数量降序，优先选该类别目标多的图
        candidates.sort(key=lambda x: x['num_objs'], reverse=True)
        
        cls_quota = 2 # 每个类别至少选2张
        c = 0
        for m in candidates:
            if c >= cls_quota: break
            if add_selection(m, f"Class Representative: {cls}"):
                c += 1

    # --- 策略 6: 尺度极端 (Scale) ---
    metadata.sort(key=lambda x: x['avg_scale'], reverse=True) # 巨大
    count = 0
    for m in metadata:
        if count >= QUOTA['large_scale'] // 2: break
        if add_selection(m, f"Large Scale (Avg: {m['avg_scale']:.1f})"):
            count += 1
            
    metadata.sort(key=lambda x: x['avg_scale']) # 极小
    count = 0
    for m in metadata:
        if count >= QUOTA['large_scale'] // 2: break
        if add_selection(m, f"Small Scale (Avg: {m['avg_scale']:.1f})"):
            count += 1
            
    # --- 策略 7: 随机补位 (Fill Random) ---
    remaining_quota = TOTAL_TARGET - len(final_selection)
    if remaining_quota > 0:
        unselected = [m for m in metadata if m['path'] not in selected_files]
        random.shuffle(unselected)
        for i in range(remaining_quota):
            if i < len(unselected):
                add_selection(unselected[i], "Random Sample")

    # 3. 执行复制操作
    print(f"筛选完毕，共选出 {len(final_selection)} 张图片。正在复制文件...")
    
    log_path = os.path.join(TARGET_DIR, 'selection_log.txt')
    with open(log_path, 'w') as log_f:
        log_f.write(f"Selection Log - Total: {len(final_selection)}\n")
        log_f.write("="*50 + "\n")
        
        for item, reason in tqdm(final_selection):
            # 复制 txt
            src_txt = item['path']
            dst_txt = os.path.join(TARGET_DIR, 'labelTxt', item['filename'])
            shutil.copy(src_txt, dst_txt)
            
            # 复制 img
            img_name = item['filename'].replace('.txt', IMAGE_EXT)
            src_img = os.path.join(SOURCE_IMAGE_DIR, img_name)
            dst_img = os.path.join(TARGET_DIR, 'images', img_name)
            
            # 尝试复制图片 (兼容 png/jpg)
            if not os.path.exists(src_img):
                 src_img = src_img.replace('.png', '.jpg')
                 dst_img = dst_img.replace('.png', '.jpg')
            
            if os.path.exists(src_img):
                shutil.copy(src_img, dst_img)
            else:
                print(f"Warning: Image not found for {src_txt}")
            
            # 记录日志
            log_f.write(f"{item['filename']:<20} | {reason}\n")
            
    print(f"\n成功! 100 张代表性图片已保存在: {TARGET_DIR}")
    print(f"筛选日志已保存至: {log_path}")
    print("现在你可以使用这个文件夹进行快速迭代验证了！")

if __name__ == '__main__':
    main()
