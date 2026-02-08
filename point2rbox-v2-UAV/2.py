import os
import math
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
import shutil

# ================= 配置区域 =================
# 1. 路径
LABEL_DIR = '/mnt/data/xiekaikai/split_ss_dota/trainval/labelTxt'
IMAGE_DIR = '/mnt/data/xiekaikai/split_ss_dota/trainval/images' 
IMAGE_EXT = '.png' 

OUTPUT_DIR = 'work_dirs/chaos_visualization_v8_4theta'

# 2. 筛选策略
TOP_K = 5               # 取最混乱的 5 张
BOTTOM_K = 5            # 取最整齐的 5 张
MIN_OBJECTS = 5         # 过滤少于5个目标的图

# 3. 参数 (必须与训练一致)
K_RADIUS = 2.0  

# 4. 类别
TARGET_CLASSES = (
    'plane', 'baseball-diamond', 'bridge', 'ground-track-field',
    'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
    'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout',
    'harbor', 'swimming-pool', 'helicopter'
)

# 5. 颜色映射
COLORMAP = plt.get_cmap('jet') 
# ===========================================

def parse_dota_poly(line):
    parts = line.strip().split()
    if len(parts) < 9: return None
    poly = np.array([float(x) for x in parts[:8]]).reshape(4, 2)
    class_name = parts[8]
    rect = cv2.minAreaRect(poly.astype(np.float32))
    (cx, cy), (w, h), angle_deg = rect
    scale = np.sqrt(w * h)
    
    # 自适应半径
    sigma = np.clip(scale * K_RADIUS, 16.0, 800.0)
    
    return {
        'cx': cx, 'cy': cy, 
        'sigma': sigma,
        'class': class_name, 
        # 基础弧度
        'theta': np.deg2rad(angle_deg + 90 if w < h else angle_deg),
        'scale': scale,
        'poly': poly.astype(np.int32)
    }

def calculate_chaos_score_4theta(objects):
    """
    [核心修改] 使用 4*theta 计算混乱度
    让 0度和90度等价 (Cross-symmetry)
    """
    N = len(objects)
    if N == 0: return [], []

    centers = np.array([[obj['cx'], obj['cy']] for obj in objects])
    scales = np.array([obj['scale'] for obj in objects])
    thetas = np.array([obj['theta'] for obj in objects])
    
    # --- 关键修改: 4 * theta ---
    # 这样 0度(cos0=1) 和 90度(cos360=1) 向量完全一致
    vecs = np.stack([np.cos(4 * thetas), np.sin(4 * thetas)], axis=1)

    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=-1)
    
    sigmas = np.clip(scales * K_RADIUS, 16.0, 800.0)
    sigma_mat = sigmas[:, np.newaxis] 
    
    W = np.exp(-dist_sq / (2 * sigma_mat ** 2))
    
    W_sum = np.sum(W, axis=1, keepdims=True) + 1e-8
    W_norm = W / W_sum

    R = np.dot(W_norm, vecs)
    R_norm = np.linalg.norm(R, axis=1)
    scores = 1.0 - R_norm
    scores = np.clip(scores, 0.0, 1.0)
    
    return scores

def render_gaussian_spot(image, cx, cy, sigma, score):
    """绘制高斯光斑 (只用于 Max/Min 高亮)"""
    H, W, _ = image.shape
    radius = int(3 * sigma)
    x_min = max(0, int(cx - radius))
    x_max = min(W, int(cx + radius + 1))
    y_min = max(0, int(cy - radius))
    y_max = min(H, int(cy + radius + 1))
    
    if x_max <= x_min or y_max <= y_min: return

    y_grid, x_grid = np.ogrid[y_min:y_max, x_min:x_max]
    dist_sq = (x_grid - cx)**2 + (y_grid - cy)**2
    
    # 透明度 0.6
    alpha_map = np.exp(-dist_sq / (2 * sigma**2)) * 0.6
    
    color_rgba = COLORMAP(score) 
    color_bgr = np.array([color_rgba[2], color_rgba[1], color_rgba[0]]) * 255
    
    roi = image[y_min:y_max, x_min:x_max]
    color_layer = np.zeros_like(roi)
    color_layer[:] = color_bgr
    
    alpha = alpha_map[:, :, np.newaxis]
    blended = roi * (1 - alpha) + color_layer * alpha
    image[y_min:y_max, x_min:x_max] = blended

def draw_gt_and_score(image, obj, score, is_highlight=False):
    """
    绘制 GT 框和分数
    is_highlight: 是否是 Max/Min，如果是，颜色加粗醒目
    """
    # 默认颜色: 青色框，白色字
    box_color = (255, 255, 0) 
    text_color = (255, 255, 255)
    thickness = 1
    font_scale = 0.4
    
    # 高亮颜色: 根据分数变色 (Max=红, Min=蓝)
    if is_highlight:
        color_rgba = COLORMAP(score)
        box_color = (int(color_rgba[2]*255), int(color_rgba[1]*255), int(color_rgba[0]*255))
        thickness = 2
        font_scale = 0.5

    # 1. 绘制 GT 框
    cv2.polylines(image, [obj['poly']], True, box_color, thickness, cv2.LINE_AA)
    
    # 2. 绘制分数 (带背景)
    text = f"{score:.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    (t_w, t_h), _ = cv2.getTextSize(text, font, font_scale, 1)
    
    cx, cy = int(obj['cx']), int(obj['cy'])
    
    # 稍微错开，不挡中心
    # 如果是高亮目标，文字位置更显眼一点
    tx, ty = cx - t_w//2, cy
    
    # 绘制文字背景框 (半透明黑)
    # 使用 OpenCV 绘制半透明矩形需要 overlay
    overlay = image.copy()
    cv2.rectangle(overlay, (tx-2, ty-t_h-2), (tx+t_w+2, ty+4), (0,0,0), -1)
    cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image) # 0.6 不透明度
    
    # 绘制文字
    # 如果分数很高(>0.5)，文字标红，警示
    final_txt_color = (0, 0, 255) if score > 0.5 else text_color
    cv2.putText(image, text, (tx, ty), font, font_scale, final_txt_color, 1, cv2.LINE_AA)

def scan_and_rank_dataset(label_files):
    print("Step 1: 全量扫描 (使用 4*theta 计算模式)...")
    ranking_db = defaultdict(list)
    
    for label_file in tqdm(label_files):
        objects_by_cls = defaultdict(list)
        with open(label_file, 'r') as f:
            for line in f:
                obj = parse_dota_poly(line)
                if obj and obj['class'] in TARGET_CLASSES:
                    objects_by_cls[obj['class']].append(obj)
        
        for cls_name, objs in objects_by_cls.items():
            if len(objs) < MIN_OBJECTS: continue
            
            # 使用 4*theta 计算
            scores = calculate_chaos_score_4theta(objs)
            avg_score = np.mean(scores)
            
            ranking_db[cls_name].append({
                'file': label_file,
                'avg_score': avg_score,
                'objects': objs,
                'scores': scores
            })
            
    print("Step 2: 建立排行榜...")
    final_picks = defaultdict(dict)
    for cls_name, items in ranking_db.items():
        sorted_items = sorted(items, key=lambda x: x['avg_score'])
        if len(sorted_items) < (TOP_K + BOTTOM_K): continue
        final_picks[cls_name]['lowest'] = sorted_items[:BOTTOM_K]
        final_picks[cls_name]['highest'] = sorted_items[-TOP_K:]
        
    return final_picks

def main():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)
    
    label_files = glob.glob(os.path.join(LABEL_DIR, '*.txt'))
    
    # 1. 扫描
    picks = scan_and_rank_dataset(label_files)
    
    print("Step 3: 渲染详细视图 (Full Context Mode)...")
    
    for cls_name, groups in picks.items():
        # 创建目录
        path_high = os.path.join(OUTPUT_DIR, cls_name, 'highest_5_chaos_4theta')
        path_low = os.path.join(OUTPUT_DIR, cls_name, 'lowest_5_chaos_4theta')
        os.makedirs(path_high, exist_ok=True)
        os.makedirs(path_low, exist_ok=True)

        tasks = []
        for i, item in enumerate(groups['lowest']):
            tasks.append((item, path_low, f"rank{i+1}_LOW"))
        for i, item in enumerate(groups['highest']):
            real_rank = TOP_K - i
            tasks.append((item, path_high, f"rank{real_rank}_HIGH"))
            
        for item, save_dir, tag in tasks:
            label_file = item['file']
            objects = item['objects']
            scores = item['scores']
            
            # 读图
            basename = os.path.basename(label_file).replace('.txt', '')
            image_path = os.path.join(IMAGE_DIR, basename + IMAGE_EXT)
            if not os.path.exists(image_path):
                image_path = os.path.join(IMAGE_DIR, basename + '.jpg')
                if not os.path.exists(image_path): continue
                    
            img_raw = cv2.imread(image_path)
            if img_raw is None: continue
            
            # 压暗
            img_vis = (img_raw.astype(np.float32) * 0.3).astype(np.uint8) # 0.3 更暗，突出文字
            
            idx_max = np.argmax(scores)
            idx_min = np.argmin(scores)
            
            # --- 层1: Max/Min 的高斯光斑 (在底层) ---
            img_float = img_vis.astype(np.float32)
            render_gaussian_spot(img_float, objects[idx_max]['cx'], objects[idx_max]['cy'], 
                               objects[idx_max]['sigma'], scores[idx_max])
            render_gaussian_spot(img_float, objects[idx_min]['cx'], objects[idx_min]['cy'], 
                               objects[idx_min]['sigma'], scores[idx_min])
            img_vis = np.clip(img_float, 0, 255).astype(np.uint8)
            
            # --- 层2: 所有物体的 GT 框和分数 ---
            for i, obj in enumerate(objects):
                is_extreme = (i == idx_max) or (i == idx_min)
                draw_gt_and_score(img_vis, obj, scores[i], is_highlight=is_extreme)
                
            # 保存
            save_name = f"{tag}_{basename}.png"
            cv2.imwrite(os.path.join(save_dir, save_name), img_vis)
            
    print(f"\n可视化完成! 结果保存在: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == '__main__':
    main()