import os
import math
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import shutil
from tqdm import tqdm
from collections import defaultdict, Counter

# ================= 配置区域 =================
# 1. 挑选后的数据集路径 (输入路径)
# 脚本会递归搜索该路径下的所有子文件夹
SELECTED_DATASET_DIR = '/mnt/data/liurunxiang/workplace/point2rbox-v2-UAV/loss1/dataset/cd'

# 2. 完整数据集路径配置 (用于统计报告 - 仍然是扁平结构)
FULL_DATASET_CONFIG = {
    'Trainval Set': '/mnt/data/xiekaikai/split_ss_codrone/trainval/annfiles',
    'Test Set':     '/mnt/data/xiekaikai/split_ss_codrone/test/annfiles'
}

# 3. 输出路径
OUTPUT_DIR = './visual/cd'

# 4. 核心参数 (与 NAOALoss 保持一致)
K_RADIUS = 2.0   
SCORE_ALPHA = 1.0 

# 5. 类别定义 & 输出顺序
CLASSES_ORDER = [
    'car', 'truck', 'bus', 'traffic-light',
    'traffic-sign', 'bridge', 'people', 'bicycle',
    'motor', 'tricycle', 'boat', 'ship'
]

# 6. 参与计算的类别
TARGET_CLASSES = set(CLASSES_ORDER)

# 7. 颜色映射
COLORMAP = plt.get_cmap('jet') 
# ===========================================

def parse_poly(line):
    """解析 DOTA 格式标注"""
    parts = line.strip().split()
    if len(parts) < 9: return None
    poly = np.array([float(x) for x in parts[:8]]).reshape(4, 2)
    cls = parts[8]
    
    rect = cv2.minAreaRect(poly.astype(np.float32))
    (cx, cy), (w, h), angle = rect
    scale = math.sqrt(w * h)
    
    # 统一角度定义 [-90, 0) -> 弧度
    if w < h: angle += 90
    theta_rad = np.deg2rad(angle)
    
    return {
        'cls': cls, 'cx': cx, 'cy': cy, 'theta': theta_rad,
        'scale': scale, 'poly': poly.astype(np.int32),
        'score': 1.0 # GT 数据默认置信度为 1.0
    }

def calculate_chaos_score(objects):
    """
    计算 NAOALoss 混乱度 (V4 第一性原理版)
    """
    valid_objs = [o for o in objects if o['cls'] in TARGET_CLASSES]
    N = len(valid_objs)
    
    final_scores = np.full(len(objects), -1.0)
    if N < 2: return final_scores 

    # 1. 数据准备
    centers = np.array([[o['cx'], o['cy']] for o in valid_objs])
    scales = np.array([o['scale'] for o in valid_objs])
    thetas = np.array([o['theta'] for o in valid_objs])
    scores = np.array([o['score'] for o in valid_objs]) 

    # 2. 矢量化 (强制 4-Theta)
    vecs = np.stack([np.cos(4*thetas), np.sin(4*thetas)], axis=1)

    # 3. 构建亲和矩阵
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=-1)
    
    scales = np.clip(scales, 16.0, 800.0)
    sigmas = scales * K_RADIUS
    sigma_mat = sigmas[:, np.newaxis]
    
    W_geo = np.exp(-dist_sq / (2 * sigma_mat**2))
    
    scores_pow = np.power(scores, SCORE_ALPHA)
    W_conf = scores_pow[np.newaxis, :] 
    
    classes = np.array([o['cls'] for o in valid_objs])
    mask_cls = (classes[:, np.newaxis] == classes[np.newaxis, :]).astype(float)
    
    W = W_geo * W_conf * mask_cls

    # 4. 归一化 (包含自环)
    W_sum = np.sum(W, axis=1, keepdims=True)
    W_norm = W / W_sum

    # 5. 能量/混乱度计算
    mean_vecs = np.dot(W_norm, vecs)
    R_norm = np.linalg.norm(mean_vecs, axis=1)
    scores_valid = 1.0 - R_norm
    scores_valid = np.clip(scores_valid, 0.0, 1.0)

    valid_idx = 0
    for i, obj in enumerate(objects):
        if obj['cls'] in TARGET_CLASSES:
            final_scores[i] = scores_valid[valid_idx]
            valid_idx += 1
            
    return final_scores

def render_gaussian_spot(image, cx, cy, scale, score):
    """渲染高斯光斑"""
    scale_clamped = max(16.0, min(800.0, scale))
    sigma = scale_clamped * K_RADIUS
    
    radius = int(3 * sigma)
    H, W, _ = image.shape
    x_min, x_max = max(0, int(cx - radius)), min(W, int(cx + radius + 1))
    y_min, y_max = max(0, int(cy - radius)), min(H, int(cy + radius + 1))
    if x_max <= x_min or y_max <= y_min: return

    y_grid, x_grid = np.ogrid[y_min:y_max, x_min:x_max]
    dist_sq = (x_grid - cx)**2 + (y_grid - cy)**2
    
    alpha_map = np.exp(-dist_sq / (2 * sigma**2)) * 0.6
    
    color_rgba = COLORMAP(score)
    color_bgr = np.array([color_rgba[2], color_rgba[1], color_rgba[0]]) * 255
    roi = image[y_min:y_max, x_min:x_max]
    color_layer = np.zeros_like(roi)
    color_layer[:] = color_bgr
    alpha = alpha_map[:, :, np.newaxis]
    image[y_min:y_max, x_min:x_max] = roi * (1 - alpha) + color_layer * alpha

def draw_info_box(image, obj, score, tag):
    """绘制信息框"""
    c_rgba = COLORMAP(score)
    c_bgr = (int(c_rgba[2]*255), int(c_rgba[1]*255), int(c_rgba[0]*255))
    cv2.polylines(image, [obj['poly']], True, c_bgr, 2, cv2.LINE_AA)
    text = f"{tag}: {score:.2f}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    tx, ty = int(obj['cx']), int(obj['cy'])
    sub = image[ty-th-4:ty+4, tx:tx+tw+4]
    if sub.shape[0]>0 and sub.shape[1]>0:
        white = np.zeros_like(sub)
        cv2.addWeighted(sub, 0.5, white, 0.5, 0, sub)
    col = (0,0,255) if score > 0.5 else (255,255,255)
    cv2.putText(image, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)

def analyze_directory(dataset_name, dir_path):
    """分析单个数据集目录"""
    if not os.path.exists(dir_path):
        print(f"[Skipping] Path not found: {dir_path}")
        return defaultdict(list)

    files = glob.glob(os.path.join(dir_path, '*.txt'))
    print(f"Loading {len(files)} files from {dataset_name}...")
    
    local_stats = defaultdict(list)
    
    for lp in tqdm(files, desc=f"Analyzing {dataset_name}", leave=False):
        objs = []
        try:
            with open(lp, 'r') as f:
                for line in f:
                    o = parse_poly(line)
                    if o: objs.append(o)
        except: continue
            
        if not objs: continue
        
        scores = calculate_chaos_score(objs)
        for i, score in enumerate(scores):
            if score >= 0:
                local_stats[objs[i]['cls']].append(score)
                
    return local_stats

def print_report(title, stats_dict):
    """打印统计报告"""
    print("\n" + "="*65)
    print(f"【 {title} 统计报告 】")
    print(f"{'Class Name':<15} | {'Avg Chaos':<10} | {'Count':<8} | {'Status'}")
    print("-" * 65)
    
    all_scores = []
    
    for cls in CLASSES_ORDER:
        val_list = stats_dict.get(cls, []) 
        
        if len(val_list) == 0:
            print(f"{cls:<15} | {'N/A':<10} | {0:<8} | -")
            continue
            
        avg_val = np.mean(val_list)
        count = len(val_list)
        all_scores.extend(val_list)
        
        status = ""
        if avg_val > 0.45: status = "[HIGH]"
        elif avg_val < 0.15: status = "[OK]"
        else: status = ""
            
        print(f"{cls:<15} | {avg_val:.4f}     | {count:<8} | {status}")
        
    global_avg = np.mean(all_scores) if all_scores else 0.0
    print("-" * 65)
    print(f"{'GLOBAL AVG':<15} | {global_avg:.4f}     | {len(all_scores):<8} |")
    print("="*65 + "\n")
    return all_scores 

def visualize_selected_samples():
    """可视化精选样本 (支持递归查找子文件夹)"""
    print(f"正在可视化精选样本: {SELECTED_DATASET_DIR} (递归搜索)...")
    
    # 清理并重建输出目录
    if os.path.exists(OUTPUT_DIR): shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)
    
    for cls in CLASSES_ORDER:
        os.makedirs(os.path.join(OUTPUT_DIR, cls), exist_ok=True)
    
    # 【核心修改】递归查找所有 .txt 文件
    # 这样无论是扁平结构还是分文件夹结构都能找到
    label_files = glob.glob(os.path.join(SELECTED_DATASET_DIR, '**', '*.txt'), recursive=True)
    
    # 过滤掉非标注文件 (如 selection_log.txt)
    label_files = [f for f in label_files if 'log.txt' not in f]
    
    print(f"找到 {len(label_files)} 个样本文件。")

    for lp in tqdm(label_files, desc="Visualizing"):
        objs = []
        with open(lp, 'r') as f:
            for line in f:
                o = parse_poly(line)
                if o: objs.append(o)
        if not objs: continue

        # 获取文件所在目录和文件名
        file_dir = os.path.dirname(lp)
        img_base = os.path.basename(lp).replace('.txt', '')
        
        img_path = None
        found_img = False
        
        # 1. 优先在同级目录查找图片 (适应 class_bins 结构)
        for ext in ['.png', '.jpg', '.bmp']:
            p = os.path.join(file_dir, img_base + ext)
            if os.path.exists(p):
                img_path = p
                found_img = True
                break
        
        # 2. 如果同级没找到，尝试去扁平的 'images' 目录找 (适应 flatten 结构)
        if not found_img:
            # 假设 dataset/cd/images 存在
            flat_img_dir = os.path.join(SELECTED_DATASET_DIR, 'images')
            if os.path.exists(flat_img_dir):
                for ext in ['.png', '.jpg', '.bmp']:
                    p = os.path.join(flat_img_dir, img_base + ext)
                    if os.path.exists(p):
                        img_path = p
                        found_img = True
                        break

        if not found_img: 
            # print(f"[Warn] Image not found for {img_base}")
            continue
            
        img_raw = cv2.imread(img_path)
        if img_raw is None: continue

        scores = calculate_chaos_score(objs)
        valid_indices = [i for i, s in enumerate(scores) if s >= 0]
        avg_img_chaos = np.mean(scores[valid_indices]) if valid_indices else 0.0

        # === 确定图片的归属类别 ===
        if valid_indices:
            valid_classes = [objs[i]['cls'] for i in valid_indices]
            primary_cls = Counter(valid_classes).most_common(1)[0][0]
        else:
            continue

        # 渲染图像
        img_vis = (img_raw.astype(np.float32) * 0.4).astype(np.uint8)
        if valid_indices:
            sub_scores = scores[valid_indices]
            idx_max = valid_indices[np.argmax(sub_scores)]
            idx_min = valid_indices[np.argmin(sub_scores)]
            
            img_float = img_vis.astype(np.float32)
            render_gaussian_spot(img_float, objs[idx_max]['cx'], objs[idx_max]['cy'], objs[idx_max]['scale'], scores[idx_max])
            render_gaussian_spot(img_float, objs[idx_min]['cx'], objs[idx_min]['cy'], objs[idx_min]['scale'], scores[idx_min])
            img_vis = np.clip(img_float, 0, 255).astype(np.uint8)
            
            draw_info_box(img_vis, objs[idx_max], scores[idx_max], "MAX")
            if idx_max != idx_min:
                draw_info_box(img_vis, objs[idx_min], scores[idx_min], "MIN")

        for i, obj in enumerate(objs):
            if valid_indices and (i == idx_max or i == idx_min): continue
            color = (255, 255, 0) if scores[i] >= 0 else (100, 100, 100)
            cv2.polylines(img_vis, [obj['poly']], True, color, 1, cv2.LINE_AA)

        # 保存到对应类别的文件夹
        save_path = os.path.join(OUTPUT_DIR, primary_cls, f"Chaos_{avg_img_chaos:.2f}_{img_base}.jpg")
        cv2.imwrite(save_path, img_vis)

def main():
    # 1. 统计 Trainval
    trainval_stats = analyze_directory('Trainval Set', FULL_DATASET_CONFIG['Trainval Set'])
    print_report('Trainval Set', trainval_stats)
    
    # 2. 统计 Test
    test_stats = analyze_directory('Test Set', FULL_DATASET_CONFIG['Test Set'])
    print_report('Test Set', test_stats)
    
    # 3. 合并统计 Global
    global_stats = defaultdict(list)
    for cls in CLASSES_ORDER:
        global_stats[cls].extend(trainval_stats[cls])
        global_stats[cls].extend(test_stats[cls])
    
    print_report('Global (Trainval + Test)', global_stats)
    
    # 4. 精选样本可视化 (按类别分类保存)
    visualize_selected_samples()
    print(f"\n可视化图片已按类别保存至: {OUTPUT_DIR}")

if __name__ == '__main__':
    main()