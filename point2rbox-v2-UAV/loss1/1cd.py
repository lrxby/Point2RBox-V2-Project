import os
import shutil
import numpy as np
import cv2
import glob
from tqdm import tqdm
from collections import defaultdict
import math
import random

# ================= 配置区域 =================
# 1. 源数据路径
SOURCE_LABEL_DIR = '/mnt/data/xiekaikai/split_ss_codrone/trainval/annfiles'
SOURCE_IMAGE_DIR = '/mnt/data/xiekaikai/split_ss_codrone/trainval/images' 
IMAGE_EXT = '.png' 

# 2. 输出路径
TARGET_DIR = './2dataset/cd'

# 3. 筛选参数
SAMPLES_PER_CLASS = 10 # 每个类别选10张

# 4. 类别定义 (CODrone 12类)
CLASSES = [
    'car', 'truck', 'bus', 'traffic-light',
    'traffic-sign', 'bridge', 'people', 'bicycle',
    'motor', 'tricycle', 'boat', 'ship'
]

# 5. 刚体类别 (用于计算混乱度)
RIGID_CLASSES = {
    'car', 'truck', 'bus', 'bicycle', 'motor', 'tricycle', 'boat', 'ship'
}
# ===========================================

def parse_codrone_poly(line):
    parts = line.strip().split()
    if len(parts) < 9: return None
    poly = np.array([float(x) for x in parts[:8]]).reshape(4, 2)
    cls = parts[8]
    rect = cv2.minAreaRect(poly.astype(np.float32))
    (cx, cy), (w, h), angle = rect
    if w < h: angle += 90
    theta_rad = np.deg2rad(angle)
    scale = math.sqrt(w * h)
    return {
        'cls': cls, 'center': np.array([cx, cy]),
        'theta': theta_rad, 'scale': scale
    }

def calculate_class_specific_chaos(objects, target_cls):
    """
    计算特定类别的混乱度。
    如果该类别是非刚体(如people)，直接返回 -1 (不参与混乱排序)
    """
    if target_cls not in RIGID_CLASSES:
        return -1.0
        
    # 只提取该类别的物体计算
    valid_objs = [o for o in objects if o['cls'] == target_cls]
    if len(valid_objs) < 3: return -1.0 # 数量太少不算

    centers = np.array([o['center'] for o in valid_objs])
    thetas = np.array([o['theta'] for o in valid_objs])
    scales = np.array([o['scale'] for o in valid_objs])

    # 4*theta 
    vecs = np.stack([np.cos(4*thetas), np.sin(4*thetas)], axis=1)
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=-1)
    
    sigmas = np.clip(scales * 2.0, 16.0, 800.0)
    sigma_mat = sigmas[:, np.newaxis]
    W = np.exp(-dist_sq / (2 * sigma_mat**2))
    
    np.fill_diagonal(W, 0)
    W_sum = np.sum(W, axis=1, keepdims=True) + 1e-6
    W_norm = W / W_sum
    
    R = np.dot(W_norm, vecs)
    R_norm = np.linalg.norm(R, axis=1)
    return np.mean(1.0 - R_norm)

def main():
    # 1. 初始化目录
    if os.path.exists(TARGET_DIR):
        print(f"Cleaning {TARGET_DIR}...")
        shutil.rmtree(TARGET_DIR)
    
    # 创建扁平结构 (用于跑脚本)
    flat_img_dir = os.path.join(TARGET_DIR, 'images')
    flat_ann_dir = os.path.join(TARGET_DIR, 'annfiles')
    os.makedirs(flat_img_dir)
    os.makedirs(flat_ann_dir)
    
    # 创建分类结构 (用于人工看)
    bins_dir = os.path.join(TARGET_DIR, 'class_bins')
    for cls in CLASSES:
        os.makedirs(os.path.join(bins_dir, cls))

    # 2. 扫描元数据
    label_files = glob.glob(os.path.join(SOURCE_LABEL_DIR, '*.txt'))
    print(f"Scanning {len(label_files)} files...")
    
    # 建立倒排索引: class -> [metadata list]
    class_pool = defaultdict(list)
    
    for lp in tqdm(label_files):
        objs = []
        with open(lp, 'r') as f:
            for line in f:
                o = parse_codrone_poly(line)
                if o: objs.append(o)
        
        if not objs: continue
        
        filename = os.path.basename(lp)
        
        # 统计该文件中包含的类别
        # 注意: 一张图可能包含多个类别，它会同时进入多个类别的候选池
        present_classes = set([o['cls'] for o in objs])
        
        for cls in present_classes:
            if cls not in CLASSES: continue
            
            # 提取该类别的专属指标
            cls_objs = [o for o in objs if o['cls'] == cls]
            count = len(cls_objs)
            avg_scale = np.mean([o['scale'] for o in cls_objs])
            chaos = calculate_class_specific_chaos(objs, cls)
            
            class_pool[cls].append({
                'path': lp,
                'filename': filename,
                'count': count,
                'scale': avg_scale,
                'chaos': chaos
            })

    # 3. 核心筛选逻辑 (Diversity Selection)
    final_selection = set() # 记录 filename，防止重复复制到 flat 目录
    selection_log = [] # 记录详细信息

    print("\nExecuting Diversity Selection...")
    
    for cls in CLASSES:
        candidates = class_pool[cls]
        # 如果样本不足，全选
        if len(candidates) <= SAMPLES_PER_CLASS:
            selected = candidates
            reasons = ["Not enough samples"] * len(candidates)
        else:
            selected = []
            reasons = []
            
            # --- Slot 1-2: 极度密集 (High Density) ---
            # 按数量降序
            sorted_by_count = sorted(candidates, key=lambda x: x['count'], reverse=True)
            for i in range(2):
                if sorted_by_count:
                    item = sorted_by_count.pop(0)
                    selected.append(item)
                    reasons.append(f"Top Density ({item['count']} objs)")
                    # 从 candidates 中移除已选，防止重复逻辑干扰 (虽然 set 会去重，但逻辑上移除更好)
                    if item in candidates: candidates.remove(item)

            # --- Slot 3-4: 最混乱 (High Chaos) ---
            # 仅针对刚体，且 chaos > 0
            rigid_candidates = [x for x in candidates if x['chaos'] >= 0]
            if rigid_candidates:
                sorted_by_chaos = sorted(rigid_candidates, key=lambda x: x['chaos'], reverse=True)
                for i in range(2):
                    if sorted_by_chaos:
                        item = sorted_by_chaos.pop(0)
                        selected.append(item)
                        reasons.append(f"High Chaos ({item['chaos']:.2f})")
                        if item in candidates: candidates.remove(item)
            else:
                # 非刚体用随机补位
                pass 

            # --- Slot 5-6: 最整齐 (Low Chaos) ---
            if rigid_candidates:
                # 重新筛选，因为上面pop掉了
                rigid_candidates = [x for x in candidates if x['chaos'] >= 0] 
                sorted_by_chaos_asc = sorted(rigid_candidates, key=lambda x: x['chaos'])
                for i in range(2):
                    if sorted_by_chaos_asc:
                        item = sorted_by_chaos_asc.pop(0)
                        # 过滤掉数量太少的"伪整齐"
                        if item['count'] < 3 and len(sorted_by_chaos_asc) > 0: continue
                        selected.append(item)
                        reasons.append(f"Low Chaos ({item['chaos']:.2f})")
                        if item in candidates: candidates.remove(item)

            # --- Slot 7: 最大尺度 (Large Scale) ---
            sorted_by_scale = sorted(candidates, key=lambda x: x['scale'], reverse=True)
            if sorted_by_scale:
                item = sorted_by_scale[0]
                selected.append(item)
                reasons.append(f"Max Scale ({item['scale']:.1f})")
                if item in candidates: candidates.remove(item)

            # --- Slot 8: 最小尺度 (Tiny Scale) ---
            sorted_by_scale_asc = sorted(candidates, key=lambda x: x['scale'])
            if sorted_by_scale_asc:
                item = sorted_by_scale_asc[0]
                selected.append(item)
                reasons.append(f"Min Scale ({item['scale']:.1f})")
                if item in candidates: candidates.remove(item)

            # --- Slot 9-10 (Fill Remaining): 随机补位 ---
            # 补足到 10 张
            while len(selected) < SAMPLES_PER_CLASS and candidates:
                item = random.choice(candidates)
                selected.append(item)
                reasons.append("Random Diversity")
                candidates.remove(item)

        # 4. 执行复制
        print(f"  Class [{cls}]: Selected {len(selected)} images")
        
        for item, r in zip(selected, reasons):
            # A. 复制到分类文件夹 (Class Bins) - 方便人工查看
            # 这里允许重复：一张图如果既有Car又有Bus，会在两个文件夹都出现
            bin_ann_path = os.path.join(bins_dir, cls, item['filename'])
            shutil.copy(item['path'], bin_ann_path)
            
            # 找图片
            img_base = item['filename'].replace('.txt', '')
            found_img = False
            for ext in ['.png', '.jpg', '.bmp']:
                src_img = os.path.join(SOURCE_IMAGE_DIR, img_base + ext)
                if os.path.exists(src_img):
                    # 复制到分类文件夹
                    shutil.copy(src_img, os.path.join(bins_dir, cls, img_base + ext))
                    
                    # B. 复制到扁平文件夹 (Flat Dir) - 供脚本运行
                    # 只有第一次遇到这张图时才复制
                    if item['filename'] not in final_selection:
                        shutil.copy(item['path'], os.path.join(flat_ann_dir, item['filename']))
                        shutil.copy(src_img, os.path.join(flat_img_dir, img_base + ext))
                        final_selection.add(item['filename'])
                    
                    found_img = True
                    break
            
            if not found_img:
                print(f"[Warn] Image missing for {item['filename']}")

            selection_log.append(f"[{cls}] {item['filename']} -> {r}")

    # 5. 保存日志
    with open(os.path.join(TARGET_DIR, 'selection_log.txt'), 'w') as f:
        f.write('\n'.join(selection_log))

    print(f"\nDone! Total unique images: {len(final_selection)}")
    print(f"1. Visualization Script Input: {flat_img_dir}")
    print(f"2. Human Check Views: {bins_dir}")

if __name__ == '__main__':
    main()