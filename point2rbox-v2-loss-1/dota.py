import os
import math
import numpy as np
import cv2
import glob
from tqdm import tqdm

# ================= 1. 配置区域 (Strictly Aligned) =================
IMG_DIR = '/mnt/data/xiekaikai/split_ss_dota/trainval/images'
LABEL_DIR = '/mnt/data/xiekaikai/split_ss_dota/trainval/annfiles'
OUTPUT_DIR = 'work_dirs/vis_dota'

# [Strict] 必须与 configs/point2rbox_v2/point2rbox_v2-1x-dota.py 一致
K_RADIUS = 2.0
ALPHA = 1.0

# 验证类别 (包含所有 DOTA 常见类别)
TARGET_CLASSES = [
    'plane', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 
    'ship', 'tennis-court', 'basketball-court', 'soccer-ball-field', 
    'roundabout', 'baseball-diamond', 'storage-tank',
    'harbor', 'swimming-pool', 'helicopter'
]
# TARGET_CLASSES = ['ground-track-field']
TOP_K = 10

# [Strict] Square Classes (无方向物体)
# 对应 Config 中的 square_cls = [1, 9, 11]
# 1: baseball-diamond, 9: storage-tank, 11: roundabout
# 在训练时，Head 层会将这些类别的预测角度强制设为 0。
# 验证脚本必须模拟这一步，否则算出来的 Loss 会虚高（不符合训练真实情况）。
SQUARE_CLASSES = ['baseball-diamond', 'storage-tank', 'roundabout']

# [Visual] 类别颜色映射 (BGR格式)
CLASS_COLOR_MAP = {
    'plane': (0, 255, 0),          # 绿色
    'bridge': (0, 255, 255),       # 黄色
    'ground-track-field': (255, 0, 0), # 蓝色
    'small-vehicle': (255, 0, 255),# 紫色
    'large-vehicle': (0, 165, 255),# 橙色
    'ship': (0, 0, 255),           # 红色
    'tennis-court': (128, 128, 0), # 深青色
    'basketball-court': (128, 0, 128), # 深紫色
    'soccer-ball-field': (0, 128, 128),# 深黄色
    'harbor': (255, 191, 0),       # 深天蓝
    'swimming-pool': (0, 191, 255),# 深橙色
    'helicopter': (50, 205, 50),   # 酸橙绿
    'roundabout': (128, 0, 0),     # 深蓝
    'baseball-diamond': (0, 128, 0), # 深绿
    'storage-tank': (0, 0, 128)    # 深红
}
DEFAULT_COLOR = (200, 200, 200) # 灰色
# ==============================================================

def parse_dota_poly(poly, class_name):
    """
    [Strict] 解析 DOTA 8点坐标 -> (cx, cy, scale, theta)
    逻辑说明: 模拟 AngleCoder 的解码过程和 Scale 计算
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
    # 如果是无方向物体，训练代码会强制将其角度设为 0
    if class_name in SQUARE_CLASSES:
        theta = 0.0

    # [Strict Step C] 计算尺度 (对应: scales = (wh[:, 0] * wh[:, 1]).sqrt())
    scale = np.sqrt(w * h)
    
    # [Strict Step D] 尺度截断 (对应: .clamp(min=16.0, max=800.0))
    scale = np.clip(scale, 16.0, 800.0)

    return {'cx': cx, 'cy': cy, 'theta': theta, 'scale': scale, 'poly': pts}

def calculate_naoa_loss_numpy_exact(objects):
    """
    [Strict] NAOALoss 的 Numpy 矩阵实现 (完全复刻 forward 函数)
    对应: mmrotate/models/losses/point2rbox_v2_loss.py -> NAOALoss.forward
    """
    N = len(objects)
    if N < 2:
        return 0.0

    # 1. 准备数据
    centers = np.array([[obj['cx'], obj['cy']] for obj in objects]) # [N, 2]
    scales = np.array([obj['scale'] for obj in objects])            # [N]
    thetas = np.array([obj['theta'] for obj in objects])            # [N]

    # 2. Angle Vectorization (对应: torch.stack([cos(2t), sin(2t)]))
    vec_preds = np.stack([np.cos(2 * thetas), np.sin(2 * thetas)], axis=1)

    # 3. Construct Affinity Matrix
    # 3.1 Spatial Distance (对应: torch.cdist(...).pow(2))
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=-1)

    # 3.2 Adaptive Sigma (对应: sigmas = scales * self.k_radius)
    sigmas = scales * K_RADIUS
    sigma_mat = sigmas[:, np.newaxis] # Broadcasting
    
    # Gaussian Weight (对应: torch.exp(-dist_sq / (2 * sigma_mat.pow(2))))
    # [Strict Update] 移除了 +1e-6，因为 PyTorch 代码里没有，且 scale 已 clip 保证安全
    weight_geo = np.exp(-dist_sq / (2 * sigma_mat ** 2))

    # 3.3 Masks (对应: mask_diag = 1.0 - torch.eye(N))
    mask_diag = 1.0 - np.eye(N)
    
    # Combine Weights
    # 假设 Score=1.0 (GT验证)，省略 pos_scores 乘法
    W = weight_geo * mask_diag

    # 4. Aggregation (对应: target_vecs = torch.mm(W, vec_preds))
    target_vecs = np.dot(W, vec_preds)

    # 5. Loss Calculation
    # Consistency Weight (对应: consistency_weights = target_vecs.norm(dim=1))
    # [Strict] 这里的 +1e-6 必须保留，对应训练代码第 500 行
    consistency_weights = np.linalg.norm(target_vecs, axis=1) + 1e-6
    
    # Normalized Target Direction
    target_dirs = target_vecs / consistency_weights[:, np.newaxis]

    # Cosine Similarity
    cos_sim = np.sum(vec_preds * target_dirs, axis=1)

    # Weighted Loss
    loss_per_item = consistency_weights * (1.0 - cos_sim)
    
    # [Strict Correction] 
    # 训练代码是对所有样本求均值 (return loss_per_item.mean())
    avg_loss = np.mean(loss_per_item)
    
    return avg_loss

def draw_results(img_path, objects, loss_val, save_name, class_name):
    if not os.path.exists(img_path):
        base = os.path.splitext(img_path)[0]
        for ext in ['.png', '.jpg', '.bmp', '.tif']:
            if os.path.exists(base + ext):
                img_path = base + ext
                break
    if not os.path.exists(img_path): return

    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None: return

    # 使用全局定义的颜色映射
    color = CLASS_COLOR_MAP.get(class_name, DEFAULT_COLOR)

    # 画框
    for obj in objects:
        pts = obj['poly'].astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(img, [pts], True, color, 2)

    # 标注信息
    h, w, _ = img.shape
    cv2.rectangle(img, (0, 0), (w, 60), (0, 0, 0), -1)
    
    # 物理意义判定 (阈值仅供参考)
    if loss_val < 0.05: verdict, v_color = "ORDERLY", (0, 255, 0)
    elif loss_val > 0.25: verdict, v_color = "CHAOTIC", (0, 0, 255)
    else: verdict, v_color = "MEDIUM", (0, 255, 255)
        
    text_left = f"Class: {class_name} | N: {len(objects)}"
    text_right = f"Mean Loss: {loss_val:.4f} [{verdict}]"
    
    cv2.putText(img, text_left, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(img, text_right, (w - 600, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, v_color, 2)

    if not os.path.exists(os.path.dirname(save_name)):
        os.makedirs(os.path.dirname(save_name))
    cv2.imencode('.png', img)[1].tofile(save_name)

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    label_files = glob.glob(os.path.join(LABEL_DIR, '*.txt'))
    print(f"找到 {len(label_files)} 个标注文件。开始严格物理验证...")
    
    results = {cls: [] for cls in TARGET_CLASSES}
    
    for label_file in tqdm(label_files):
        filename = os.path.basename(label_file)
        img_id = os.path.splitext(filename)[0]
        
        with open(label_file, 'r') as f:
            lines = f.readlines()
            
        file_objects = {cls: [] for cls in TARGET_CLASSES}
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 9: continue
            try:
                poly = [float(x) for x in parts[:8]]
                cls = parts[8]
                if cls in TARGET_CLASSES:
                    # [Strict] 传入 cls 以便内部判断是否为 square_cls
                    obj = parse_dota_poly(poly, cls)
                    file_objects[cls].append(obj)
            except:
                continue
        
        # 计算 Loss
        for cls in TARGET_CLASSES:
            objs = file_objects[cls]
            # 为了验证有意义，只统计数量 >= 5 的图片
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
        
        # 按 Loss 排序
        data.sort(key=lambda x: x['loss'])
        
        print(f"\n类别 [{cls}]: 样本数 {len(data)}")
        print(f"  Min Loss: {data[0]['loss']:.4f}")
        print(f"  Max Loss: {data[-1]['loss']:.4f}")
        
        # 保存 Low Loss (整齐)
        save_dir_low = os.path.join(OUTPUT_DIR, cls, 'Low_Loss_Orderly')
        for i in range(min(TOP_K, len(data))):
            item = data[i]
            img_path = os.path.join(IMG_DIR, item['img_id'] + '.png')
            save_name = os.path.join(save_dir_low, f"rank{i+1}_loss{item['loss']:.4f}_{item['img_id']}.png")
            draw_results(img_path, item['objects'], item['loss'], save_name, cls)
            
        # 保存 High Loss (杂乱)
        save_dir_high = os.path.join(OUTPUT_DIR, cls, 'High_Loss_Chaotic')
        for i in range(min(TOP_K, len(data))):
            item = data[-(i+1)] 
            img_path = os.path.join(IMG_DIR, item['img_id'] + '.png')
            save_name = os.path.join(save_dir_high, f"rank{i+1}_loss{item['loss']:.4f}_{item['img_id']}.png")
            draw_results(img_path, item['objects'], item['loss'], save_name, cls)

    print(f"\n验证完成！结果保存在: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == '__main__':
    main()