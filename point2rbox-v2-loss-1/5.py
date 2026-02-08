import os
import math
import numpy as np
import cv2
import glob
from tqdm import tqdm

# ================= 1. 配置区域 (Strictly Aligned) =================
IMG_DIR = '/mnt/data/xiekaikai/split_ss_dota/trainval/images'
LABEL_DIR = '/mnt/data/xiekaikai/split_ss_dota/trainval/annfiles'
OUTPUT_DIR = 'work_dirs/loss_local_heatmap' # 结果保存在这里

# [Strict] 必须与 configs/point2rbox_v2/point2rbox_v2-1x-dota.py 一致
K_RADIUS = 2.0
ALPHA = 1.0

# 验证类别
TARGET_CLASSES = [
    'small-vehicle', 'large-vehicle', 'ship', 'plane', 'harbor', 
    'bridge', 'ground-track-field', 'tennis-court', 'basketball-court', 
    'soccer-ball-field', 'roundabout', 'baseball-diamond', 'storage-tank',
    'swimming-pool', 'helicopter'
]
TOP_K = 10 # 每个类别保存多少张图

# [Strict] Square Classes (强制 Loss=0 的类别)
SQUARE_CLASSES = ['baseball-diamond', 'storage-tank', 'roundabout']
# ==============================================================

def parse_dota_poly(poly, class_name):
    """[Strict] 解析坐标，模拟训练时的 Head 处理"""
    pts = np.array(poly).reshape(4, 2)
    rect = cv2.minAreaRect(pts.astype(np.float32))
    (cx, cy), (w, h), angle_deg = rect
    
    if w < h:
        w, h = h, w
        angle_deg += 90
    
    theta = np.deg2rad(angle_deg)
    
    # [Strict] 模拟 Head 层对无方向物体的强制归零
    if class_name in SQUARE_CLASSES:
        theta = 0.0

    scale = np.sqrt(w * h)
    scale = np.clip(scale, 16.0, 800.0)

    return {'cx': cx, 'cy': cy, 'theta': theta, 'scale': scale, 'poly': pts}

def calculate_naoa_loss_per_item(objects):
    """
    [Strict] 计算每一个目标的 Loss，而不是返回平均值
    """
    N = len(objects)
    # 如果少于2个，无法计算邻域，返回全0
    if N < 2:
        return np.zeros(N)

    # 1. 准备数据
    centers = np.array([[obj['cx'], obj['cy']] for obj in objects]) 
    scales = np.array([obj['scale'] for obj in objects])            
    thetas = np.array([obj['theta'] for obj in objects])            

    # 2. Angle Vectorization
    vec_preds = np.stack([np.cos(2 * thetas), np.sin(2 * thetas)], axis=1)

    # 3. Construct Affinity Matrix
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=-1)

    sigmas = scales * K_RADIUS
    sigma_mat = sigmas[:, np.newaxis] 
    
    # [Strict] 移除 1e-6 (与 PyTorch 一致)
    weight_geo = np.exp(-dist_sq / (2 * sigma_mat ** 2))

    mask_diag = 1.0 - np.eye(N)
    W = weight_geo * mask_diag

    # 4. Aggregation
    target_vecs = np.dot(W, vec_preds)

    # 5. Loss Calculation
    # [Strict] 保留 1e-6
    consistency_weights = np.linalg.norm(target_vecs, axis=1) + 1e-6
    
    target_dirs = target_vecs / consistency_weights[:, np.newaxis]
    cos_sim = np.sum(vec_preds * target_dirs, axis=1)

    # [Result] 得到每个物体单独的 Loss
    loss_per_item = consistency_weights * (1.0 - cos_sim)
    
    return loss_per_item

def get_color_by_loss(loss):
    """
    根据 Loss 大小返回颜色 (BGR)
    Low Loss (0.0) -> Green (0, 255, 0)
    High Loss (>0.5) -> Red (0, 0, 255)
    """
    # 归一化：假设 0.5 以上就是非常乱了
    norm = np.clip(loss / 0.5, 0, 1) 
    
    # 线性插值颜色：绿 -> 黄 -> 红
    # Green: (0, 255, 0), Red: (0, 0, 255)
    b = 0
    g = int(255 * (1 - norm))
    r = int(255 * norm)
    return (b, g, r)

def draw_heatmap_results(img_path, objects, losses, save_name, class_name):
    if not os.path.exists(img_path):
        base = os.path.splitext(img_path)[0]
        for ext in ['.png', '.jpg', '.bmp', '.tif']:
            if os.path.exists(base + ext):
                img_path = base + ext
                break
    if not os.path.exists(img_path): return

    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None: return

    # 画每一个框
    for i, obj in enumerate(objects):
        loss_val = losses[i]
        
        # 1. 根据 Loss 决定颜色
        color = get_color_by_loss(loss_val)
        
        # 2. 画多边形框
        pts = obj['poly'].astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(img, [pts], True, color, 2)
        
        # 3. 在框中心写上 Loss 数值 (可选，太小可能看不清)
        # 如果 Loss 非常小(接近0)，只画绿框不写字，避免遮挡
        # 如果 Loss 较大，写出来提示
        if loss_val > 0.05:
            cx, cy = int(obj['cx']), int(obj['cy'])
            label_text = f"{loss_val:.2f}"
            cv2.putText(img, label_text, (cx-10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # 标注全局信息
    mean_loss = np.mean(losses)
    h, w, _ = img.shape
    cv2.rectangle(img, (0, 0), (w, 40), (0, 0, 0), -1)
    
    text = f"Class: {class_name} | N: {len(objects)} | Mean Loss: {mean_loss:.4f}"
    cv2.putText(img, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    if not os.path.exists(os.path.dirname(save_name)):
        os.makedirs(os.path.dirname(save_name))
    cv2.imencode('.png', img)[1].tofile(save_name)

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    label_files = glob.glob(os.path.join(LABEL_DIR, '*.txt'))
    print(f"找到 {len(label_files)} 个标注文件。开始局部物理验证...")
    
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
                    obj = parse_dota_poly(poly, cls)
                    file_objects[cls].append(obj)
            except:
                continue
        
        # 计算 Loss
        for cls in TARGET_CLASSES:
            objs = file_objects[cls]
            if len(objs) >= 5: 
                # [关键] 获取每个目标的单独 Loss
                losses = calculate_naoa_loss_per_item(objs)
                
                mean_loss = np.mean(losses)
                results[cls].append({
                    'img_id': img_id,
                    'mean_loss': mean_loss, # 用于排序筛选图片
                    'losses': losses,       # 用于画图
                    'objects': objs
                })
                    
    # 结果保存
    for cls in TARGET_CLASSES:
        data = results[cls]
        if not data: continue
        
        # 1. 依然按 Mean Loss 排序，找出整体最整齐和最乱的图
        data.sort(key=lambda x: x['mean_loss'])
        
        print(f"\n类别 [{cls}]: 样本数 {len(data)}")
        
        # 保存 Low Loss (整体整齐)
        save_dir_low = os.path.join(OUTPUT_DIR, cls, 'Overall_Orderly')
        for i in range(min(TOP_K, len(data))):
            item = data[i]
            img_path = os.path.join(IMG_DIR, item['img_id'] + '.png')
            save_name = os.path.join(save_dir_low, f"rank{i+1}_mean{item['mean_loss']:.4f}.png")
            draw_heatmap_results(img_path, item['objects'], item['losses'], save_name, cls)
            
        # 保存 High Loss (整体杂乱)
        # 这里的图最有意思：可能会看到局部整齐(绿)和局部杂乱(红)的混合
        save_dir_high = os.path.join(OUTPUT_DIR, cls, 'Overall_Chaotic')
        for i in range(min(TOP_K, len(data))):
            item = data[-(i+1)] 
            img_path = os.path.join(IMG_DIR, item['img_id'] + '.png')
            save_name = os.path.join(save_dir_high, f"rank{i+1}_mean{item['mean_loss']:.4f}.png")
            draw_heatmap_results(img_path, item['objects'], item['losses'], save_name, cls)

    print(f"\n验证完成！请查看: {os.path.abspath(OUTPUT_DIR)}")
    print("提示：绿色框代表该物体与邻居对齐良好(Loss≈0)，红色框代表该物体与邻居冲突严重(Loss大)。")

if __name__ == '__main__':
    main()
