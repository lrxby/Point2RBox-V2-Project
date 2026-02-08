import os
import glob
import shutil
import numpy as np
from tqdm import tqdm

# ================= 配置区域 =================
# 原始标注文件路径
SOURCE_ANN_DIR = '/mnt/data/xiekaikai/split_ss_codrone/trainval/annfiles'

# 图片文件路径 (自动推断)
SOURCE_IMG_DIR = os.path.join(os.path.dirname(SOURCE_ANN_DIR), 'images')

# 结果保存根路径
TARGET_ROOT = './dataset/cd'

# CODrone 全类别
CLASSES = ('car', 'truck', 'bus', 'traffic-light',
           'traffic-sign', 'bridge', 'people', 'bicycle',
           'motor', 'tricycle', 'boat', 'ship')

# 正则项
RIDGE_LAMBDA = 1e-4 
# ===========================================

def polygon_area(coords):
    x = coords[0::2]
    y = coords[1::2]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def fit_perspective_score_strict(objects):
    """计算透视分"""
    N = len(objects)
    present_classes = sorted(list(set([o['cls'] for o in objects])))
    K = len(present_classes)
    
    if N < K + 3: return 0.0

    cls_to_idx = {name: i for i, name in enumerate(present_classes)}
    Y = np.array([o['log_s'] for o in objects])
    X_raw = np.array([o['x'] for o in objects])
    Y_raw = np.array([o['y'] for o in objects])
    
    x_mean, x_std = X_raw.mean(), X_raw.std() + 1e-6
    y_mean, y_std = Y_raw.mean(), Y_raw.std() + 1e-6
    X_norm = (X_raw - x_mean) / x_std
    Y_norm = (Y_raw - y_mean) / y_std
    
    A = np.zeros((N, 2 + K))
    A[:, 0] = X_norm
    A[:, 1] = Y_norm
    for i, obj in enumerate(objects):
        A[i, 2 + cls_to_idx[obj['cls']]] = 1

    M = A.T @ A
    I_reg = np.eye(2 + K) * RIDGE_LAMBDA
    try:
        theta = np.linalg.inv(M + I_reg) @ (A.T @ Y)
        return np.sqrt(theta[0]**2 + theta[1]**2)
    except:
        return 0.0

def analyze_file(file_path):
    """分析单文件：统计数量、透视分、平均尺寸"""
    objects = []
    class_stats = {c: {'count': 0, 'mean_size': 0.0} for c in CLASSES}
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 9: continue
            
            cls_name = parts[8]
            if cls_name not in CLASSES: continue
            
            coords = list(map(float, parts[:8]))
            area = polygon_area(coords)
            if area <= 1: continue
            
            s = np.sqrt(area)
            s = max(s, 1e-2)
            cx = sum(coords[0::2]) / 4.0
            cy = sum(coords[1::2]) / 4.0
            
            objects.append({
                'x': cx, 'y': cy, 'log_s': np.log(s), 'cls': cls_name
            })
            
            # 更新统计
            class_stats[cls_name]['count'] += 1
            # 累加尺寸 (最后求平均)
            class_stats[cls_name]['mean_size'] += s

    # 计算平均尺寸
    for c in CLASSES:
        if class_stats[c]['count'] > 0:
            class_stats[c]['mean_size'] /= class_stats[c]['count']
            
    perspective_score = fit_perspective_score_strict(objects)
    
    return {
        'path': file_path,
        'filename': os.path.basename(file_path),
        'perspective_score': perspective_score,
        'class_stats': class_stats,
        'total_objects': len(objects)
    }

def copy_data(stat_item, target_folder):
    ann_src = stat_item['path']
    ann_dst = os.path.join(target_folder, stat_item['filename'])
    shutil.copy(ann_src, ann_dst)
    
    basename = os.path.splitext(stat_item['filename'])[0]
    for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tif']:
        src_img = os.path.join(SOURCE_IMG_DIR, basename + ext)
        if os.path.exists(src_img):
            shutil.copy(src_img, os.path.join(target_folder, basename + ext))
            break

def main():
    if os.path.exists(TARGET_ROOT):
        shutil.rmtree(TARGET_ROOT)
    os.makedirs(TARGET_ROOT)

    # 1. 全量扫描
    all_files = glob.glob(os.path.join(SOURCE_ANN_DIR, '*.txt'))
    print(f"正在全量扫描 {len(all_files)} 个标注文件...")
    
    all_stats = []
    for f in tqdm(all_files):
        all_stats.append(analyze_file(f))

    print("\n开始分层挑选 (Stratified Sampling)...")
    
    for target_class in CLASSES:
        cls_dir = os.path.join(TARGET_ROOT, target_class)
        os.makedirs(cls_dir)
        
        # 基础筛选：必须包含该类别至少3个物体 (太少没法单独看)
        candidates = [s for s in all_stats if s['class_stats'][target_class]['count'] >= 2]
        
        if len(candidates) < 10:
            # 样本极少，全部选取
            final_selection = candidates
        else:
            final_selection = []
            seen_files = set()

            def add_to_selection(items):
                for item in items:
                    if item['filename'] not in seen_files:
                        final_selection.append(item)
                        seen_files.add(item['filename'])

            # --- 策略A: 密集/强透视组 (Top 3) ---
            # 按数量 * 透视分 排序
            candidates.sort(key=lambda x: (x['class_stats'][target_class]['count'], x['perspective_score']), reverse=True)
            add_to_selection(candidates[:3])
            
            # --- 策略B: 稀疏/低透视组 (Bottom 3) ---
            # 按数量少、透视分低排序 (但要保证透视分不是0，排除完全无法拟合的)
            valid_low = [c for c in candidates if c['perspective_score'] > 0]
            valid_low.sort(key=lambda x: (x['class_stats'][target_class]['count'], x['perspective_score'])) # 升序
            add_to_selection(valid_low[:3])
            
            # --- 策略C: 小目标组 (Small 2) ---
            # 按该类别的平均尺寸升序
            candidates.sort(key=lambda x: x['class_stats'][target_class]['mean_size'])
            add_to_selection(candidates[:2])
            
            # --- 策略D: 大目标组 (Large 2) ---
            # 按该类别的平均尺寸降序
            candidates.sort(key=lambda x: x['class_stats'][target_class]['mean_size'], reverse=True)
            add_to_selection(candidates[:2])
            
            # --- 补齐 (如果去重后不足10张) ---
            if len(final_selection) < 10:
                candidates.sort(key=lambda x: x['total_objects'], reverse=True) # 选总物体多的补齐
                for c in candidates:
                    if len(final_selection) >= 10: break
                    if c['filename'] not in seen_files:
                        final_selection.append(c)
                        seen_files.add(c['filename'])

        print(f"类别 [{target_class:<15}]: 候选库 {len(candidates):<5} -> 选中 {len(final_selection)}")
        
        for item in final_selection:
            copy_data(item, cls_dir)

    print("-" * 40)
    print(f"挑选完成！保存在: {TARGET_ROOT}")
    print("每个类别文件夹包含 10 张图，覆盖：强透视、弱透视、大目标、小目标。")

if __name__ == '__main__':
    main()