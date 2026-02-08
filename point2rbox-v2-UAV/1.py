import os
import math
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import random
import colorsys
from tqdm import tqdm

# ================= 配置区域 =================
# 1. 路径
LABEL_DIR = '/mnt/data/xiekaikai/split_ss_dota/trainval/labelTxt'
IMAGE_DIR = '/mnt/data/xiekaikai/split_ss_dota/trainval/images' 
IMAGE_EXT = '.png' # 或 .jpg

OUTPUT_DIR = 'work_dirs/chaos_visualization_v5_final'

# 2. 采样
MAX_VIS_IMAGES = 50     
MIN_OBJECTS = 5         

# 3. 参数
K_RADIUS = 2.0  

# 4. 类别
TARGET_CLASSES = (
    'plane', 'baseball-diamond', 'bridge', 'ground-track-field',
    'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
    'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout',
    'harbor', 'swimming-pool', 'helicopter'
)
# ===========================================

def generate_distinct_colors(n):
    """生成 n 个互不相同的明亮颜色 (BGR格式)"""
    colors = []
    for i in range(n):
        # 使用 Golden Ratio 让色相分布最均匀
        hue = (i * 0.618033988749895) % 1.0 
        saturation = 0.85
        value = 0.95
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append((int(b*255), int(g*255), int(r*255)))
    return colors

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
        'theta': np.deg2rad(angle_deg + 90 if w < h else angle_deg),
        'scale': scale,
        'poly': poly.astype(np.int32)
    }

def calculate_chaos_score_vectorized(objects):
    """计算混乱度 (Physics)"""
    N = len(objects)
    if N == 0: return [], []

    centers = np.array([[obj['cx'], obj['cy']] for obj in objects])
    scales = np.array([obj['scale'] for obj in objects])
    thetas = np.array([obj['theta'] for obj in objects])
    
    vecs = np.stack([np.cos(2 * thetas), np.sin(2 * thetas)], axis=1)

    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=-1)
    
    sigmas = np.clip(scales * K_RADIUS, 16.0, 800.0)
    sigma_mat = sigmas[:, np.newaxis] 
    
    # 纯高斯权重 (同一类)
    W = np.exp(-dist_sq / (2 * sigma_mat ** 2))
    
    # 归一化
    W_sum = np.sum(W, axis=1, keepdims=True) + 1e-8
    W_norm = W / W_sum

    R = np.dot(W_norm, vecs)
    R_norm = np.linalg.norm(R, axis=1)
    scores = 1.0 - R_norm
    scores = np.clip(scores, 0.0, 1.0)
    
    return scores

def render_instance_gaussian(image, obj, color_bgr):
    """绘制高斯光束"""
    H, W, _ = image.shape
    cx, cy = obj['cx'], obj['cy']
    sigma = obj['sigma']
    
    radius = int(3 * sigma)
    x_min = max(0, int(cx - radius))
    x_max = min(W, int(cx + radius + 1))
    y_min = max(0, int(cy - radius))
    y_max = min(H, int(cy + radius + 1))
    
    if x_max <= x_min or y_max <= y_min: return

    y_grid, x_grid = np.ogrid[y_min:y_max, x_min:x_max]
    dist_sq = (x_grid - cx)**2 + (y_grid - cy)**2
    
    # 透明度：中心 0.6 -> 边缘 0
    alpha_map = np.exp(-dist_sq / (2 * sigma**2)) * 0.6
    
    roi = image[y_min:y_max, x_min:x_max]
    color_layer = np.zeros_like(roi)
    color_layer[:] = color_bgr
    
    alpha = alpha_map[:, :, np.newaxis]
    blended = roi * (1 - alpha) + color_layer * alpha
    image[y_min:y_max, x_min:x_max] = blended

def draw_text_with_bg(img, text, pos, text_color=(255, 255, 255), bg_color=(0, 0, 0)):
    """绘制带背景框的文字，确保看得清"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.4
    thickness = 1
    (t_w, t_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    
    x, y = pos
    # 画背景黑框
    cv2.rectangle(img, (x, y - t_h - 2), (x + t_w, y + 2), bg_color, -1)
    # 画文字
    cv2.putText(img, text, (x, y), font, scale, text_color, thickness, cv2.LINE_AA)

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    label_files = glob.glob(os.path.join(LABEL_DIR, '*.txt'))
    print(f"Found {len(label_files)} label files.")
    
    random.seed(42)
    random.shuffle(label_files)
    
    saved_count = 0
    pbar = tqdm(total=MAX_VIS_IMAGES)

    for label_file in label_files:
        if saved_count >= MAX_VIS_IMAGES:
            break

        # 1. 解析对象
        objects_by_cls = {cls: [] for cls in TARGET_CLASSES}
        with open(label_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                obj = parse_dota_poly(line)
                if obj and obj['class'] in TARGET_CLASSES:
                    objects_by_cls[obj['class']].append(obj)

        img_raw = None
        basename = os.path.basename(label_file).replace('.txt', '')

        # 2. 按类别绘图
        for cls_name, objects in objects_by_cls.items():
            if len(objects) < MIN_OBJECTS:
                continue
            
            if img_raw is None:
                image_path = os.path.join(IMAGE_DIR, basename + IMAGE_EXT)
                if not os.path.exists(image_path):
                    image_path = os.path.join(IMAGE_DIR, basename + '.jpg')
                    if not os.path.exists(image_path): break
                img_raw = cv2.imread(image_path)
                if img_raw is None: break

            # 复制底图并压暗
            img_vis = (img_raw.astype(np.float32) * 0.4).astype(np.uint8)
            
            # 计算混乱度分数
            scores = calculate_chaos_score_vectorized(objects)
            
            # 生成互不相同的颜色
            colors = generate_distinct_colors(len(objects))
            
            # 渲染循环
            for i, obj in enumerate(objects):
                score = scores[i]
                color = colors[i]
                
                # A. 渲染高斯光束 (实例颜色)
                img_float = img_vis.astype(np.float32)
                render_instance_gaussian(img_float, obj, color)
                img_vis = np.clip(img_float, 0, 255).astype(np.uint8)
                
                # B. 绘制 GT 框 (实例颜色，线宽2)
                cv2.polylines(img_vis, [obj['poly']], True, color, 2, cv2.LINE_AA)
                
                # C. 绘制分数文字 (关键优化)
                cx, cy = int(obj['cx']), int(obj['cy'])
                
                # 文本内容: 分数
                text = f"{score:.2f}"
                
                # 文本颜色逻辑: 
                # 如果混乱度 > 0.5，文字显示为亮黄色/红色引起注意
                # 否则显示白色
                text_color = (0, 255, 255) if score > 0.5 else (255, 255, 255)
                
                # 绘制带背景的文字，确保不被遮挡
                draw_text_with_bg(img_vis, text, (cx - 10, cy), text_color=text_color)

            # 保存
            save_name = f"{basename}_{cls_name}.png"
            cv2.imwrite(os.path.join(OUTPUT_DIR, save_name), img_vis)
            
            saved_count += 1
            pbar.update(1)
            if saved_count >= MAX_VIS_IMAGES: break
            
    pbar.close()
    print(f"Saved to {OUTPUT_DIR}")

if __name__ == '__main__':
    main()