import os
import math
import numpy as np
import cv2
import glob
from tqdm import tqdm

# ================= 1. 配置区域 (Strictly Aligned) =================
IMG_DIR = '/mnt/data/xiekaikai/split_ss_codrone/trainval/images'
LABEL_DIR = '/mnt/data/xiekaikai/split_ss_codrone/trainval/annfiles'
OUTPUT_DIR = 'work_dirs/vis_codrone'

# [Strict] 必须与 configs 一致
K_RADIUS = 2.0
ALPHA = 1.0

# 验证类别 (CODrone)
TARGET_CLASSES = [
    'car', 'truck', 'bus', 'traffic-light',
    'traffic-sign', 'bridge', 'people', 'bicycle',
    'motor', 'tricycle', 'boat', 'ship'
]
TOP_K = 10

# [Strict] Square Classes (无方向物体)
# CODrone 中 traffic-light/sign 可能无方向，但若未在 config 中明确指定，则设为空。
# 若需要忽略这些类的方向 Loss，可将其加入列表。
SQUARE_CLASSES = [] 

# [Visual] 类别颜色映射
CLASS_COLOR_MAP = {
    'car': (0, 255, 0),         
    'truck': (255, 0, 0),       
    'bus': (0, 255, 255),       
    'traffic-light': (0, 0, 255),
    'traffic-sign': (255, 0, 255),
    'bridge': (128, 128, 0),
    'people': (200, 200, 200),
    'bicycle': (255, 191, 0),
    'motor': (0, 165, 255),
    'tricycle': (0, 128, 128),
    'boat': (128, 0, 128),
    'ship': (0, 0, 128)
}
DEFAULT_COLOR = (200, 200, 200)
# ==============================================================

def parse_poly(poly, class_name):
    """
    [Strict] 解析坐标 -> (cx, cy, scale, theta)
    """
    pts = np.array(poly).reshape(4, 2)
    rect = cv2.minAreaRect(pts.astype(np.float32))
    (cx, cy), (w, h), angle_deg = rect
    
    # [Strict Step A] 模拟 le90 角度定义 (长边定义)
    if w < h:
        w, h = h, w
        angle_deg += 90
    
    # 转弧度
    theta = np.deg2rad(angle_deg)
    
    # [Strict Step B] 模拟 Head 层的 square_cls 处理
    if class_name in SQUARE_CLASSES:
        theta = 0.0

    # [Strict Step C] 计算尺度
    scale = np.sqrt(w * h)
    
    # [Strict Step D] 尺度截断 (CODrone 物体较小，但保持统一参数)
    scale = np.clip(scale, 16.0, 800.0)

    return {'cx': cx, 'cy': cy, 'theta': theta, 'scale': scale, 'poly': pts}

def calculate_naoa_loss_numpy_exact(objects):
    """
    [Strict] NAOALoss 的 Numpy 矩阵实现
    """
    N = len(objects)
    if N < 2: return 0.0

    centers = np.array([[obj['cx'], obj['cy']] for obj in objects]) 
    scales = np.array([obj['scale'] for obj in objects])            
    thetas = np.array([obj['theta'] for obj in objects])            

    vec_preds = np.stack([np.cos(2 * thetas), np.sin(2 * thetas)], axis=1)

    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=-1)

    sigmas = scales * K_RADIUS
    sigma_mat = sigmas[:, np.newaxis] 
    
    weight_geo = np.exp(-dist_sq / (2 * sigma_mat ** 2))

    mask_diag = 1.0 - np.eye(N)
    W = weight_geo * mask_diag

    target_vecs = np.dot(W, vec_preds)

    consistency_weights = np.linalg.norm(target_vecs, axis=1) + 1e-6
    target_dirs = target_vecs / consistency_weights[:, np.newaxis]
    cos_sim = np.sum(vec_preds * target_dirs, axis=1)
    loss_per_item = consistency_weights * (1.0 - cos_sim)
    
    return np.mean(loss_per_item)

def draw_results(img_path, objects, loss_val, save_name, class_name):
    # 处理图片扩展名
    if not os.path.exists(img_path):
        base = os.path.splitext(img_path)[0]
        for ext in ['.png', '.jpg', '.bmp', '.tif']:
            if os.path.exists(base + ext):
                img_path = base + ext
                break
    if not os.path.exists(img_path): return

    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None: return

    color = CLASS_COLOR_MAP.get(class_name, DEFAULT_COLOR)

    for obj in objects:
        pts = obj['poly'].astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(img, [pts], True, color, 2)

    h, w, _ = img.shape
    cv2.rectangle(img, (0, 0), (w, 60), (0, 0, 0), -1)
    
    # 物理意义判定
    if loss_val < 0.05: verdict, v_color = "ORDERLY", (0, 255, 0)
    elif loss_val > 0.25: verdict, v_color = "CHAOTIC", (0, 0, 255)
    else: verdict, v_color = "MEDIUM", (0, 255, 255)
        
    text_left = f"Class: {class_name} | N: {len(objects)}"
    text_right = f"Mean Loss: {loss_val:.4f} [{verdict}]"
    
    cv2.putText(img, text_left, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(img, text_right, (w - 650, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, v_color, 2)

    if not os.path.exists(os.path.dirname(save_name)):
        os.makedirs(os.path.dirname(save_name))
    cv2.imencode('.png', img)[1].tofile(save_name)

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    label_files = glob.glob(os.path.join(LABEL_DIR, '*.txt'))
    print(f"找到 {len(label_files)} 个标注文件。开始 CODrone 物理验证...")
    
    results = {cls: [] for cls in TARGET_CLASSES}
    
    for label_file in tqdm(label_files):
        filename = os.path.basename(label_file)
        img_id = os.path.splitext(filename)[0]
        
        with open(label_file, 'r') as f:
            lines = f.readlines()
            
        file_objects = {cls: [] for cls in TARGET_CLASSES}
        
        for line in lines:
            parts = line.strip().split()
            # 格式: x1 y1 ... x4 y4 class_name difficulty
            if len(parts) < 10: continue
            try:
                poly = [float(x) for x in parts[:8]]
                cls_name = parts[8]
                
                if cls_name in TARGET_CLASSES:
                    obj = parse_poly(poly, cls_name)
                    file_objects[cls_name].append(obj)
            except:
                continue
        
        # 计算 Loss
        for cls in TARGET_CLASSES:
            objs = file_objects[cls]
            if len(objs) >= 2: 
                loss = calculate_naoa_loss_numpy_exact(objs)
                results[cls].append({
                    'img_id': img_id,
                    'loss': loss,
                    'objects': objs
                })
                    
    # 结果排序与保存
    for cls in TARGET_CLASSES:
        data = results[cls]
        if not data: continue
        
        data.sort(key=lambda x: x['loss'])
        
        print(f"\n类别 [{cls}]: 样本数 {len(data)}")
        print(f"  Min Loss: {data[0]['loss']:.4f}")
        print(f"  Max Loss: {data[-1]['loss']:.4f}")
        
        # 保存 Low Loss
        save_dir_low = os.path.join(OUTPUT_DIR, cls, 'Low_Loss_Orderly')
        for i in range(min(TOP_K, len(data))):
            item = data[i]
            img_path = os.path.join(IMG_DIR, item['img_id'] + '.png')
            save_name = os.path.join(save_dir_low, f"rank{i+1}_loss{item['loss']:.4f}_{item['img_id']}.png")
            draw_results(img_path, item['objects'], item['loss'], save_name, cls)
            
        # 保存 High Loss
        save_dir_high = os.path.join(OUTPUT_DIR, cls, 'High_Loss_Chaotic')
        for i in range(min(TOP_K, len(data))):
            item = data[-(i+1)] 
            img_path = os.path.join(IMG_DIR, item['img_id'] + '.png')
            save_name = os.path.join(save_dir_high, f"rank{i+1}_loss{item['loss']:.4f}_{item['img_id']}.png")
            draw_results(img_path, item['objects'], item['loss'], save_name, cls)

    print(f"\n验证完成！结果保存在: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == '__main__':
    main()
