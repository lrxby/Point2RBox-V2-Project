import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

# ================= 配置区域 =================
# 数据集路径
ANN_DIR = '/mnt/data/xiekaikai/split_ss_codrone/trainval/annfiles'
# 结果保存路径
SAVE_DIR = './perspective_analysis_results'

# 类别定义 (CODrone)
CLASSES = ('car', 'truck', 'bus', 'traffic-light',
           'traffic-sign', 'bridge', 'people', 'bicycle',
           'motor', 'tricycle', 'boat', 'ship')

# 【关键】指定参与拟合的刚性类别
# 建议先只用最稳的类别来确定透视平面
# 这里排除了 people(非刚性), bridge(差异大), boat/ship(可能长尾)
TARGET_CLASSES = ['car', 'truck', 'bus', 'traffic-light', 'traffic-sign']
# TARGET_CLASSES = list(CLASSES) # 如果想测试全类别，取消注释此行

# 图像尺寸 (用于归一化和绘图背景，DOTA切图通常是1024)
IMG_W, IMG_H = 1024, 1024
# ===========================================

def polygon_area(coords):
    """计算多边形面积 (Shoelace Formula)"""
    x = coords[0::2]
    y = coords[1::2]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def parse_txt(txt_path):
    """解析 DOTA 格式 txt"""
    objects = []
    with open(txt_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 9: continue
            
            # 提取坐标 (x1, y1, ..., x4, y4)
            coords = list(map(float, parts[:8]))
            cls_name = parts[8]
            
            # 筛选刚性类别
            if cls_name not in TARGET_CLASSES:
                continue
                
            # 计算中心点和尺寸
            poly_area = polygon_area(coords)
            if poly_area <= 1: continue # 过滤极小噪点
            
            s = np.sqrt(poly_area) # 尺寸 = sqrt(area)
            cx = sum(coords[0::2]) / 4.0
            cy = sum(coords[1::2]) / 4.0
            
            objects.append({
                'cls': cls_name,
                'x': cx,
                'y': cy,
                's': s,
                'log_s': np.log(s),
                'coords': coords
            })
    return objects

def fit_perspective_plane(objects):
    """
    核心数学逻辑：PASCL v2.0 混合效应模型
    求解: log(s) = wx * x + wy * y + b_class
    """
    if len(objects) < 5: return None, None, None # 样本太少不拟合

    # 1. 准备数据
    N = len(objects)
    unique_classes = sorted(list(set([o['cls'] for o in objects])))
    K = len(unique_classes)
    cls_to_idx = {name: i for i, name in enumerate(unique_classes)}

    # 2. 构建矩阵
    # 目标向量 Y: [log(s)]
    Y = np.array([o['log_s'] for o in objects])
    
    # 设计矩阵 A: [x_norm, y_norm, one_hot_classes...]
    # 为了数值稳定，坐标归一化到 [-1, 1]
    X_raw = np.array([o['x'] for o in objects])
    Y_raw = np.array([o['y'] for o in objects])
    X_norm = (X_raw - IMG_W/2) / (IMG_W/2)
    Y_norm = (Y_raw - IMG_H/2) / (IMG_H/2)
    
    A = np.zeros((N, 2 + K))
    A[:, 0] = X_norm # 共享斜率 wx
    A[:, 1] = Y_norm # 共享斜率 wy
    
    for i, obj in enumerate(objects):
        col_idx = 2 + cls_to_idx[obj['cls']]
        A[i, col_idx] = 1 # 独立截距 b_class

    # 3. 最小二乘求解 (theta = [wx, wy, b0, b1...])
    # 加一点正则化 (rcond) 防止奇异矩阵
    theta, residuals, rank, s = np.linalg.lstsq(A, Y, rcond=None)
    
    params = {
        'wx': theta[0],
        'wy': theta[1],
        'intercepts': {name: theta[2+i] for i, name in enumerate(unique_classes)},
        'cls_to_idx': cls_to_idx
    }
    
    # 计算每个样本的理论值和误差
    Y_pred = A @ theta
    errors = np.abs(Y - Y_pred)
    
    return params, errors, Y_pred

def visualize(file_name, objects, params, errors, Y_pred):
    """生成三合一可视化图"""
    fig = plt.figure(figsize=(18, 6))
    plt.suptitle(f"File: {file_name} | Objects: {len(objects)}", fontsize=14)

    # --- 子图 1: 透视热力场 (Perspective Field) ---
    ax1 = fig.add_subplot(131)
    ax1.set_title(f"Perspective Field\nSlope: wx={params['wx']:.2f}, wy={params['wy']:.2f}")
    
    # 创建网格
    xx = np.linspace(-1, 1, 100)
    yy = np.linspace(-1, 1, 100)
    XX, YY = np.meshgrid(xx, yy)
    # 计算透视势能 Z = wx*x + wy*y (不包含截距，只看倾斜)
    ZZ = params['wx'] * XX + params['wy'] * YY
    
    im = ax1.imshow(ZZ, extent=[0, IMG_W, IMG_H, 0], cmap='coolwarm', alpha=0.6)
    plt.colorbar(im, ax=ax1, label='Log Scale Factor')
    
    # 画出物体位置
    cls_colors = {}
    cmap_scatter = plt.get_cmap('tab10')
    for i, cls_name in enumerate(params['intercepts'].keys()):
        cls_colors[cls_name] = cmap_scatter(i)
        
    for obj in objects:
        ax1.scatter(obj['x'], obj['y'], c='black', s=10, alpha=0.5)
    
    ax1.set_xlim(0, IMG_W)
    ax1.set_ylim(IMG_H, 0) # 图像坐标系 Y轴向下

    # --- 子图 2: 3D 拟合平面 (Normalized Fitting) ---
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.set_title("Normalized 3D Plane Fitting\n(Corrected by Class Intercepts)")
    
    # 画平面
    ax2.plot_surface(XX, YY, ZZ, alpha=0.2, color='gray')
    
    # 画点：Z = log(s) - b_class
    # 这样不同类别的物体应该都落在同一个平面上
    for i, obj in enumerate(objects):
        b_k = params['intercepts'][obj['cls']]
        z_norm = obj['log_s'] - b_k
        x_norm = (obj['x'] - IMG_W/2) / (IMG_W/2)
        y_norm = (obj['y'] - IMG_H/2) / (IMG_H/2)
        
        c = cls_colors[obj['cls']]
        ax2.scatter(x_norm, y_norm, z_norm, color=c, s=20)

    ax2.set_xlabel('X (Position)')
    ax2.set_ylabel('Y (Position)')
    ax2.set_zlabel('Log Size (Normalized)')

    # --- 子图 3: GT 误差残留图 (Residual Map) ---
    ax3 = fig.add_subplot(133)
    mean_err = np.mean(errors)
    ax3.set_title(f"Size Error Map\nMean Error: {mean_err:.4f}")
    
    # 黑色背景
    ax3.set_facecolor('black')
    
    for i, obj in enumerate(objects):
        err = errors[i]
        coords = np.array(obj['coords']).reshape(-1, 2)
        poly = plt.Polygon(coords, fill=False, linewidth=2)
        
        # 颜色编码：绿(好) -> 黄 -> 红(差)
        # 误差 > 0.3 (对数空间) 算比较大
        color_val = min(err / 0.5, 1.0) 
        edge_color = (color_val, 1.0 - color_val, 0) 
        
        poly.set_edgecolor(edge_color)
        ax3.add_patch(poly)
        
        # 标注误差值和类别
        if err > 0.2: # 只标注误差大的，避免拥挤
            ax3.text(obj['x'], obj['y'], f"{err:.2f}\n{obj['cls'][:2]}", 
                     color='white', fontsize=8, ha='center')

    ax3.set_xlim(0, IMG_W)
    ax3.set_ylim(IMG_H, 0)

    # 保存
    if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)
    save_path = os.path.join(SAVE_DIR, file_name.replace('.txt', '.png'))
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path} | Error: {mean_err:.4f}")

def main():
    txt_files = glob.glob(os.path.join(ANN_DIR, '*.txt'))
    print(f"Found {len(txt_files)} annotation files.")
    
    # 随机抽样或者跑全部
    # txt_files = txt_files[:20] 

    for txt_path in txt_files:
        file_name = os.path.basename(txt_path)
        objects = parse_txt(txt_path)
        
        if not objects: continue

        # 核心：拟合平面
        params, errors, Y_pred = fit_perspective_plane(objects)
        
        if params is None:
            continue
            
        # 可视化
        visualize(file_name, objects, params, errors, Y_pred)

if __name__ == '__main__':
    main()
