import os
import math
import numpy as np
import cv2
import glob

# ================= 配置 =================
# 填入你想要分析的具体的 txt 路径（把你上面贴出的内容保存为 txt）
# 或者直接指定 label 文件夹
LABEL_DIR = '/mnt/data/xiekaikai/split_ss_dota/trainval/annfiles'
IMG_DIR = '/mnt/data/xiekaikai/split_ss_dota/trainval/images'
OUTPUT_DIR = 'work_dirs/loss_debug_visual'

# [关键修改] 指定这 6 张图的 ID
# 注意：这里只写文件名（不带扩展名），假设你的文件名格式是 Pxxxx__1024__xxx___xxx
TARGET_IDS = [
    'P1397__1024__824___0',      # 图1
    'P1397__1024__824___824',    # 图2
    'P2280__1024__0___3296',     # 图3
    'P0071__1024__0___0',        # 图4
    'P2182__1024__0___0',        # 图5
    'P0113__1024__824___2472'    # 图6
]

K_RADIUS = 2.0
TARGET_CLASSES = ['small-vehicle', 'large-vehicle', 'ship', 'plane']
# ========================================

def parse_rect(poly):
    pts = np.array(poly).reshape(4, 2)
    rect = cv2.minAreaRect(pts.astype(np.float32))
    (cx, cy), (w, h), angle_deg = rect
    
    # le90 物理定义：长边为方向
    if w < h:
        w, h = h, w
        angle_deg += 90
    
    theta = np.deg2rad(angle_deg)
    scale = np.sqrt(w * h)
    scale = np.clip(scale, 16.0, 800.0)
    return cx, cy, w, h, theta, scale, pts

def visualize_connections(img_path, label_path, save_path):
    # 路径检查
    if not os.path.exists(img_path):
        # 尝试其他后缀
        base = os.path.splitext(img_path)[0]
        found = False
        for ext in ['.png', '.jpg', '.bmp', '.tif']:
            if os.path.exists(base + ext):
                img_path = base + ext
                found = True
                break
        if not found:
            print(f"[Skip] Image not found: {img_path}")
            return

    if not os.path.exists(label_path):
        print(f"[Skip] Label not found: {label_path}")
        return

    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    
    with open(label_path, 'r') as f:
        lines = f.readlines()

    objects = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 9: continue
        try:
            poly = [float(x) for x in parts[:8]]
            cls = parts[8]
            if cls in TARGET_CLASSES:
                cx, cy, w, h, theta, scale, pts = parse_rect(poly)
                objects.append({
                    'cx': cx, 'cy': cy, 'theta': theta, 'scale': scale, 
                    'pts': pts.astype(np.int32), 'cls': cls
                })
        except:
            continue

    if len(objects) < 2: 
        print(f"[Skip] Not enough objects in {os.path.basename(img_path)}")
        return

    # 准备数据
    centers = np.array([[obj['cx'], obj['cy']] for obj in objects])
    scales = np.array([obj['scale'] for obj in objects])
    
    # 距离矩阵
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=-1)
    
    # 计算权重矩阵
    sigmas = scales * K_RADIUS
    sigma_mat = sigmas[:, np.newaxis]
    # 高斯权重计算
    weight_geo = np.exp(-dist_sq / (2 * sigma_mat ** 2 + 1e-6))
    
    # 绘制
    # 1. 先把图变暗，方便看线
    overlay = img.copy()
    cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)

    # 2. 画连接线 (权重 > 0.1 的邻居)
    neighbor_count = 0
    for i in range(len(objects)):
        for j in range(len(objects)):
            if i == j: continue
            # 如果 j 是 i 的强邻居
            if weight_geo[i, j] > 1e-4: 
                pt1 = (int(objects[i]['cx']), int(objects[i]['cy']))
                pt2 = (int(objects[j]['cx']), int(objects[j]['cy']))
                # 线越粗代表权重越大
                thickness = max(1, int(weight_geo[i, j] * 3))
                cv2.line(img, pt1, pt2, (0, 0, 255), thickness) # 红色连线
                neighbor_count += 1

    # 3. 画框和方向箭头
    for obj in objects:
        # 画框
        cv2.polylines(img, [obj['pts']], True, (0, 255, 0), 1)
        
        # 画长边方向 (Arrow)
        # 长度设为 scale 的一半，稍微夸张一点
        length = obj['scale'] * 0.8
        dx = length * math.cos(obj['theta'])
        dy = length * math.sin(obj['theta'])
        
        pt1 = (int(obj['cx']), int(obj['cy']))
        pt2 = (int(obj['cx'] + dx), int(obj['cy'] + dy))
        
        # 蓝色箭头代表算法认为的"角度"
        # 箭头末端加个圆点表示起点
        cv2.circle(img, pt1, 3, (255, 0, 0), -1) 
        cv2.arrowedLine(img, pt1, pt2, (255, 0, 0), 2, tipLength=0.3)

    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_name))
    cv2.imwrite(save_path, img)
    print(f"Generated: {save_path} (Neighbors: {neighbor_count})")

# 执行逻辑
if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

print(f"开始处理指定的 {len(TARGET_IDS)} 张图片...")

for target_id in TARGET_IDS:
    # 构造文件名 (假设标注文件是 .txt)
    txt_name = target_id + '.txt'
    txt_path = os.path.join(LABEL_DIR, txt_name)
    
    # 图片路径需要根据 ID 去找，这里假设直接在 IMG_DIR 下
    # 如果找不到，脚本里的 check 会尝试补全后缀
    img_path = os.path.join(IMG_DIR, target_id) 
    
    save_name = os.path.join(OUTPUT_DIR, f"Debug_{target_id}.png")
    
    visualize_connections(img_path, txt_path, save_name)

print(f"\n全部完成！结果保存在: {os.path.abspath(OUTPUT_DIR)}")
