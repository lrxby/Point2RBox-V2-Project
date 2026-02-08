import os
import glob
import numpy as np
import matplotlib.pyplot as plt
# 修复多进程matplotlib渲染冲突，必须放在导入Axes3D之前
plt.switch_backend('Agg')
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Polygon
import cv2 
from tqdm import tqdm
# 新增多进程依赖
import multiprocessing
from multiprocessing import Pool

# ================= 配置区域 =================
# 1. 全量数据集路径 (用于统计排行榜)
FULL_TRAINVAL_DIR = '/mnt/data/xiekaikai/split_ss_codrone/trainval/labelTxt'
FULL_TEST_DIR = '/mnt/data/xiekaikai/split_ss_codrone/test/labelTxt'

# 2. 待可视化的挑选数据集路径
VIS_DATA_ROOT = './dataset/cd'

# 3. 结果保存根目录
SAVE_ROOT = './visual/cd_iou_vis'

# CODrone 全类别
CLASSES = ('car', 'truck', 'bus', 'traffic-light',
           'traffic-sign', 'bridge', 'people', 'bicycle',
           'motor', 'tricycle', 'boat', 'ship')

# 图像默认尺寸
DEFAULT_W, DEFAULT_H = 1024, 1024

# 正则项
RIDGE_LAMBDA = 1e-4

# 【可视化参数】
# 不再需要 MAX_ERROR_THRESHOLD，因为 IoU 是归一化的 (0~1)
# 多进程进程数
PROCESS_NUM = multiprocessing.cpu_count()
# ===========================================

def polygon_area(coords):
    x = coords[0::2]
    y = coords[1::2]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def parse_txt(txt_path):
    objects = []
    with open(txt_path, 'r') as f:
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
                'cls': cls_name,
                'x': cx,
                'y': cy,
                's': s,
                'log_s': s, # 线性模式，保持 s 不变
                'coords': coords
            })
    return objects

def fit_perspective_plane(objects, img_w, img_h):
    """拟合透视平面 (线性版本)"""
    N = len(objects)
    present_classes = sorted(list(set([o['cls'] for o in objects])))
    K = len(present_classes)
    
    if N < K + 3: return None, None, None

    cls_to_idx = {name: i for i, name in enumerate(present_classes)}

    Y = np.array([o['s'] for o in objects])
    
    X_raw = np.array([o['x'] for o in objects])
    Y_raw = np.array([o['y'] for o in objects])
    
    X_norm = (X_raw - img_w/2) / (img_w/2)
    Y_norm = (Y_raw - img_h/2) / (img_h/2)
    
    A = np.zeros((N, 2 + K))
    A[:, 0] = X_norm
    A[:, 1] = Y_norm
    for i, obj in enumerate(objects):
        A[i, 2 + cls_to_idx[obj['cls']]] = 1

    M = A.T @ A
    I_reg = np.eye(2 + K) * RIDGE_LAMBDA
    
    try:
        theta = np.linalg.inv(M + I_reg) @ (A.T @ Y)
    except np.linalg.LinAlgError:
        return None, None, None
        
    params = {
        'wx': theta[0],
        'wy': theta[1],
        'intercepts': {name: theta[2+i] for i, name in enumerate(present_classes)},
        'present_classes': present_classes
    }
    
    Y_pred = A @ theta
    signed_errors = Y_pred - Y 
    
    return params, signed_errors, Y_pred

# ==========================================
# 模块1: 全量数据统计 (修改为 IoU + MAPE)
# ==========================================
def calculate_dataset_stats(ann_dir, dataset_name="Dataset"):
    print(f"\n启动统计: {dataset_name} ({ann_dir})")
    txt_files = glob.glob(os.path.join(ann_dir, '*.txt'))
    print(f"  - 发现文件数: {len(txt_files)}")
    
    # 存储不同指标
    stats = {c: {'iou': [], 'mape': []} for c in CLASSES}
    valid_images = 0
    
    for txt_path in tqdm(txt_files, desc=f"Analyzing {dataset_name}"):
        objects = parse_txt(txt_path)
        if not objects: continue
        
        # 拟合
        params, signed_errors, Y_pred = fit_perspective_plane(objects, DEFAULT_W, DEFAULT_H)
        
        if params is None: continue
        valid_images += 1
        
        # 计算具体指标
        for i, obj in enumerate(objects):
            s_gt = obj['s']
            s_pred = Y_pred[i]
            
            # 1. 计算 IoU (Size IoU)
            # IoU = min(A1, A2) / max(A1, A2)
            # Area = s^2
            area_gt = s_gt ** 2
            area_pred = s_pred ** 2 # 线性模式直接是 s
            # 避免负数预测导致面积计算错误（虽然线性拟合一般在正数区间，但加上保护）
            area_pred = max(area_pred, 1e-6) 
            
            iou = min(area_gt, area_pred) / max(area_gt, area_pred)
            
            # 2. 计算 MAPE
            mape = abs(s_gt - s_pred) / (s_gt + 1e-6)
            
            stats[obj['cls']]['iou'].append(iou)
            stats[obj['cls']]['mape'].append(mape)
            
    print(f"  - 有效透视拟合图片数: {valid_images}")
    return stats

def print_leaderboard(stats, title="Leaderboard"):
    print("\n" + "="*80)
    print(f"{title:^80}")
    print("="*80)
    # 修改表头，展示 IoU 和 MAPE
    print(f"{'Class':<15} | {'Mean IoU (%)':<15} | {'Mean MAPE (%)':<15} | {'Samples':<10}")
    print("-" * 80)
    
    ranking = []
    for cls in CLASSES:
        ious = stats.get(cls, {}).get('iou', [])
        mapes = stats.get(cls, {}).get('mape', [])
        
        if len(ious) > 0:
            mean_iou = np.mean(ious) * 100 # 转百分比
            mean_mape = np.mean(mapes) * 100 # 转百分比
            ranking.append((cls, mean_iou, mean_mape, len(ious)))
        else:
            ranking.append((cls, -1, -1, 0))
            
    # 按 IoU 降序排列 (越大约好)
    ranking.sort(key=lambda x: x[1], reverse=True)
    
    for cls, mean_iou, mean_mape, cnt in ranking:
        if cnt == 0:
            print(f"{cls:<15} | {'N/A':<15} | {'N/A':<15} | {cnt:<10}")
        else:
            print(f"{cls:<15} | {mean_iou:<15.2f} | {mean_mape:<15.2f} | {cnt:<10}")
    print("="*80)

# ==========================================
# 模块2: 挑选数据可视化 (修改为 IoU 可视化)
# ==========================================
def get_image_path(ann_path):
    basename = os.path.splitext(os.path.basename(ann_path))[0]
    dirname = os.path.dirname(ann_path)
    for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tif']:
        img_path = os.path.join(dirname, basename + ext)
        if os.path.exists(img_path):
            return img_path
    return None

def visualize_sample(cls_folder_name, file_name, img_path, objects, params, signed_errors):
    img = cv2.imread(img_path)
    if img is None: return
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H, W = img.shape[:2]
    
    fig = plt.figure(figsize=(20, 6))
    plt.suptitle(f"Class: {cls_folder_name} | File: {file_name} ({W}x{H}) | Objects: {len(objects)}", fontsize=14)

    # 1. 热力图
    ax1 = fig.add_subplot(131)
    ax1.set_title(f"Perspective Field (Slope)\nwx={params['wx']:.2f}, wy={params['wy']:.2f}", pad=20)
    ax1.set_facecolor('black')
    xx = np.linspace(-1, 1, 100)
    yy = np.linspace(-1, 1, 100)
    XX, YY = np.meshgrid(xx, yy)
    ZZ = params['wx'] * XX + params['wy'] * YY
    im = ax1.imshow(ZZ, extent=[0, W, H, 0], cmap='coolwarm') 
    plt.colorbar(im, ax=ax1, label='Scale Variance')
    ax1.scatter([o['x'] for o in objects], [o['y'] for o in objects], c='white', s=10, alpha=0.6)
    ax1.xaxis.tick_top()
    ax1.set_xlim(0, W); ax1.set_ylim(H, 0)

    # 2. 3D平面
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.set_title("3D Plane Fitting", pad=20)
    ax2.plot_surface(XX, YY, ZZ, alpha=0.2, color='gray')
    unique_cls = sorted(list(set(CLASSES)))
    cmap = plt.get_cmap('tab20')
    cls_colors = {c: cmap(i/len(unique_cls)) for i, c in enumerate(unique_cls)}
    for i, obj in enumerate(objects):
        cls = obj['cls']
        b_k = params['intercepts'][cls]
        x_n = (obj['x'] - W/2) / (W/2)
        y_n = (obj['y'] - H/2) / (H/2)
        z_n = obj['s'] - b_k 
        c = cls_colors.get(cls, 'k')
        ax2.scatter(x_n, y_n, z_n, color=c, s=15, alpha=0.8)
    
    # 翻转坐标轴以符合直觉
    ax2.invert_yaxis()
    # ax2.invert_xaxis()
    ax2.set_xlabel('X'); ax2.set_ylabel('Y'); ax2.set_zlabel('Size')

    # 3. 误差图 (IoU Alpha + Red/Blue Bias)
    ax3 = fig.add_subplot(133)
    
    # 计算平均 IoU 用于标题
    ious = []
    for i, obj in enumerate(objects):
        s_gt = obj['s']
        s_pred = obj['s'] + signed_errors[i] # 还原预测值
        area_gt = s_gt**2
        area_pred = max(s_pred**2, 1e-6)
        iou = min(area_gt, area_pred) / max(area_gt, area_pred)
        ious.append(iou)
    mean_iou = np.mean(ious)
    
    ax3.set_title(f"IoU Visualizer (Red=Pred>GT, Blue=Pred<GT)\nMean IoU: {mean_iou:.2f}", pad=20)
    ax3.imshow(img, extent=[0, W, H, 0], alpha=0.6) 
    
    all_x, all_y = [], []
    for i, obj in enumerate(objects):
        diff = signed_errors[i] 
        iou = ious[i]
        
        coords = np.array(obj['coords']).reshape(-1, 2)
        all_x.extend(coords[:, 0]); all_y.extend(coords[:, 1])
        
        # --- 颜色逻辑：红蓝表示误差方向 ---
        if diff > 0:
            color = (1.0, 0.0, 0.0) # 红 (偏大)
        else:
            color = (0.0, 0.0, 1.0) # 蓝 (偏小)
            
        # --- 透明度逻辑：IoU 决定深浅 ---
        # IoU 越小(误差越大)，Alpha 越大(越不透明)
        # IoU = 1.0 -> Alpha = 0.1 (很淡)
        # IoU = 0.5 -> Alpha = 0.5
        # IoU = 0.0 -> Alpha = 0.9 (很深)
        alpha_val = (1.0 - iou) * 0.8 + 0.1 
        alpha_val = min(max(alpha_val, 0.1), 0.9)
        
        poly = Polygon(coords, closed=True, 
                       facecolor=color, 
                       alpha=alpha_val, 
                       edgecolor=color, 
                       linewidth=1)
        ax3.add_patch(poly)
        
        # 标注 IoU 值 (保留两位小数)
        # 仅当 IoU < 0.9 (有明显误差) 时才显示，避免太乱
        if iou < 0.9:
            ax3.text(obj['x'], obj['y'], f"{iou:.2f}", 
                     color='white', fontsize=7, ha='center', fontweight='bold',
                     bbox=dict(facecolor=color, alpha=0.7, edgecolor='none', pad=0.5)) 
    
    margin = 50
    if all_x:
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        ax3.set_xlim(min(0, min_x)-margin, max(W, max_x)+margin)
        ax3.set_ylim(max(H, max_y)+margin, min(0, min_y)-margin)
    else:
        ax3.set_xlim(0, W); ax3.set_ylim(H, 0)
    ax3.xaxis.tick_top()
    ax3.xaxis.set_label_position('top')

    save_folder = os.path.join(SAVE_ROOT, cls_folder_name)
    if not os.path.exists(save_folder): os.makedirs(save_folder)
    save_path = os.path.join(save_folder, file_name.replace('.txt', '.png'))
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def vis_worker(task_args):
    """多进程工作函数"""
    cls_name, txt_path = task_args
    try:
        file_name = os.path.basename(txt_path)
        img_path = get_image_path(txt_path)
        if not img_path: return 0
        
        img = cv2.imread(img_path)
        if img is None: return 0
        H, W = img.shape[:2]
        
        objects = parse_txt(txt_path)
        if not objects: return 0
        
        params, signed_errors, _ = fit_perspective_plane(objects, W, H)
        if params is None: return 0
        
        visualize_sample(cls_name, file_name, img_path, objects, params, signed_errors)
        return 1
    except Exception:
        return 0

def run_visualization_on_selected():
    print(f"\n启动可视化绘制 (基于挑选的样本: {VIS_DATA_ROOT})...")
    if not os.path.exists(VIS_DATA_ROOT):
        print("Error: 挑选数据集不存在，请先运行挑选脚本。")
        return

    task_list = []
    for cls_name in CLASSES:
        cls_dir = os.path.join(VIS_DATA_ROOT, cls_name)
        if not os.path.exists(cls_dir): continue
        txt_files = glob.glob(os.path.join(cls_dir, '*.txt'))
        for txt_path in txt_files:
            task_list.append((cls_name, txt_path))
    
    total_task = len(task_list)
    print(f"  - 待可视化文件数: {total_task}")
    if total_task == 0: return

    with Pool(processes=PROCESS_NUM) as pool:
        results = list(tqdm(
            pool.imap(vis_worker, task_list),
            total=total_task,
            desc="Drawing Visualizations"
        ))
    
    print(f"可视化完成！共生成 {sum(results)} 张图表，保存在 {SAVE_ROOT}")

def main():
    # 1. 运行 Trainval 统计
    stats_trainval = calculate_dataset_stats(FULL_TRAINVAL_DIR, "TrainVal Set (Linear)")
    
    # 2. 运行 Test 统计
    stats_test = calculate_dataset_stats(FULL_TEST_DIR, "Test Set (Linear)")
    
    # 3. 汇总全量统计 (Global)
    stats_total = {c: {'iou': [], 'mape': []} for c in CLASSES}
    for c in CLASSES:
        stats_total[c]['iou'].extend(stats_trainval[c]['iou'])
        stats_total[c]['mape'].extend(stats_trainval[c]['mape'])
        stats_total[c]['iou'].extend(stats_test[c]['iou'])
        stats_total[c]['mape'].extend(stats_test[c]['mape'])
        
    # 4. 打印三张排行榜 (含 IoU 和 MAPE)
    print_leaderboard(stats_trainval, "Leaderboard: TrainVal Set (IoU & MAPE)")
    print_leaderboard(stats_test, "Leaderboard: Test Set (IoU & MAPE)")
    print_leaderboard(stats_total, "Leaderboard: GLOBAL (IoU & MAPE)")
    
    # 5. 运行多进程可视化
    run_visualization_on_selected()

if __name__ == '__main__':
    main()