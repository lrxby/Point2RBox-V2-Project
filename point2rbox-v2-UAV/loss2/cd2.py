import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Polygon
import cv2 
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# ================= 配置区域 =================
FULL_TRAINVAL_DIR = '/mnt/data/xiekaikai/split_ss_codrone/trainval/labelTxt'
FULL_TEST_DIR = '/mnt/data/xiekaikai/split_ss_codrone/test/labelTxt'
VIS_DATA_ROOT = './dataset/cd'
SAVE_ROOT = './visual/cd_redblue_vis'
CLASSES = ('car', 'truck', 'bus', 'traffic-light',
           'traffic-sign', 'bridge', 'people', 'bicycle',
           'motor', 'tricycle', 'boat', 'ship')
DEFAULT_W, DEFAULT_H = 1024, 1024
RIDGE_LAMBDA = 1e-4
MAX_ERROR_THRESHOLD = 30.0 
MAX_WORKERS = 24
# ===========================================

plt_lock = threading.Lock()

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
                'log_s': s,
                'coords': coords
            })
    return objects

def fit_perspective_plane(objects, img_w, img_h):
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

def process_single_stats_file(txt_path):
    file_result = {c: [] for c in CLASSES}
    valid_flag = False
    try:
        objects = parse_txt(txt_path)
        if not objects:
            return file_result, valid_flag
        params, signed_errors, _ = fit_perspective_plane(objects, DEFAULT_W, DEFAULT_H)
        if params is None:
            return file_result, valid_flag
        valid_flag = True
        abs_errors = np.abs(signed_errors)
        for i, obj in enumerate(objects):
            file_result[obj['cls']].append(abs_errors[i])
    except Exception:
        pass
    return file_result, valid_flag

def calculate_dataset_stats(ann_dir, dataset_name="Dataset"):
    print(f"\n启动统计: {dataset_name} ({ann_dir})")
    txt_files = glob.glob(os.path.join(ann_dir, '*.txt'))
    print(f"  - 发现文件数: {len(txt_files)}")
    class_errors = {c: [] for c in CLASSES}
    valid_images = 0
    if not txt_files:
        print(f"  - 有效透视拟合图片数: {valid_images}")
        return class_errors
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_file = {executor.submit(process_single_stats_file, path): path for path in txt_files}
        for future in tqdm(as_completed(future_to_file), total=len(txt_files), desc=f"Analyzing {dataset_name}"):
            file_res, is_valid = future.result()
            if is_valid:
                valid_images += 1
                for cls in CLASSES:
                    class_errors[cls].extend(file_res[cls])
    print(f"  - 有效透视拟合图片数: {valid_images}")
    return class_errors

def print_leaderboard(stats, title="Leaderboard"):
    print("\n" + "="*60)
    print(f"{title:^60}")
    print("="*60)
    print(f"{'Class':<15} | {'Mean Error (Px)':<15} | {'Samples':<10} | {'Status'}")
    print("-" * 60)
    ranking = []
    for cls in CLASSES:
        errs = stats.get(cls, [])
        if len(errs) > 0:
            mean_e = np.mean(errs)
            ranking.append((cls, mean_e, len(errs)))
        else:
            ranking.append((cls, 999.0, 0))
    ranking.sort(key=lambda x: x[1])
    for cls, mean_e, cnt in ranking:
        if cnt == 0:
            status = "No Samples"
            print(f"{cls:<15} | {'N/A':<15} | {cnt:<10} | {status}")
        else:
            if mean_e < 10.0: status = "Excellent ✅"
            elif mean_e < 20.0: status = "Good ⭕"
            else: status = "Bad ❌"
            print(f"{cls:<15} | {mean_e:.4f}           | {cnt:<10} | {status}")
    print("="*60)

def process_single_vis_file(cls_name, txt_path):
    try:
        file_name = os.path.basename(txt_path)
        img_path = get_image_path(txt_path)
        if not img_path:
            return 0
        img = cv2.imread(img_path)
        if img is None:
            return 0
        H, W = img.shape[:2]
        objects = parse_txt(txt_path)
        if not objects:
            return 0
        params, signed_errors, _ = fit_perspective_plane(objects, W, H)
        if params is None:
            return 0
        with plt_lock:
            visualize_sample(cls_name, file_name, img_path, objects, params, signed_errors)
        return 1
    except Exception:
        return 0

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

    # 2. 3D平面 - 核心修改：强制锁定X/Y轴方向
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
    # 关键修改：先翻转X/Y轴，再强制设置刻度范围，确保方向正确
    ax2.invert_xaxis()
    ax2.invert_yaxis()
    # 强制锁定Y轴范围：右上为1，左下为-1
    ax2.set_ylim(1, -1)
    ax2.set_xlim(1, -1)
    ax2.set_xlabel('X'); ax2.set_ylabel('Y'); ax2.set_zlabel('Size')

    # 3. 误差图
    ax3 = fig.add_subplot(133)
    mae = np.mean(np.abs(signed_errors))
    ax3.set_title(f"Bias Map (Red=Pred>GT, Blue=Pred<GT)\nMAE: {mae:.1f} px", pad=20)
    ax3.imshow(img, extent=[0, W, H, 0], alpha=0.6)
    all_x, all_y = [], []
    for i, obj in enumerate(objects):
        diff = signed_errors[i]
        abs_diff = abs(diff)
        coords = np.array(obj['coords']).reshape(-1, 2)
        all_x.extend(coords[:, 0]); all_y.extend(coords[:, 1])
        if diff > 0:
            color = (1.0, 0.0, 0.0) 
        else:
            color = (0.0, 0.0, 1.0)
        alpha_val = min(abs_diff / MAX_ERROR_THRESHOLD, 0.8)
        alpha_val = max(alpha_val, 0.1)
        poly = Polygon(coords, closed=True, 
                       facecolor=color, 
                       alpha=alpha_val, 
                       edgecolor=color,
                       linewidth=1)
        ax3.add_patch(poly)
        if abs_diff > MAX_ERROR_THRESHOLD / 2:
            ax3.text(obj['x'], obj['y'], f"{diff:.0f}", 
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
    os.makedirs(save_folder, exist_ok=True)
    save_path = os.path.join(save_folder, file_name.replace('.txt', '.png'))
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def run_visualization_on_selected():
    print(f"\n启动可视化绘制 (基于挑选的样本: {VIS_DATA_ROOT})...")
    if not os.path.exists(VIS_DATA_ROOT):
        print("Error: 挑选数据集不存在，请先运行挑选脚本。")
        return
    vis_tasks = []
    for cls_name in CLASSES:
        cls_dir = os.path.join(VIS_DATA_ROOT, cls_name)
        if not os.path.exists(cls_dir):
            continue
        txt_files = glob.glob(os.path.join(cls_dir, '*.txt'))
        for txt_path in txt_files:
            vis_tasks.append((cls_name, txt_path))
    total_count = len(vis_tasks)
    success_count = 0
    print(f"  - 待可视化文件数: {total_count}")
    if total_count == 0:
        print("可视化完成！无有效文件需要处理")
        return
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_task = {executor.submit(process_single_vis_file, cls, path): (cls, path) for cls, path in vis_tasks}
        for future in tqdm(as_completed(future_to_task), total=total_count, desc="Drawing Visualizations"):
            success_count += future.result()
    print(f"可视化完成！共生成 {success_count} 张图表，保存在 {SAVE_ROOT}")

def main():
    stats_trainval = calculate_dataset_stats(FULL_TRAINVAL_DIR, "TrainVal Set (Linear)")
    stats_test = calculate_dataset_stats(FULL_TEST_DIR, "Test Set (Linear)")
    stats_total = {c: [] for c in CLASSES}
    for c in CLASSES:
        stats_total[c].extend(stats_trainval[c])
        stats_total[c].extend(stats_test[c])
    print_leaderboard(stats_trainval, "Leaderboard: TrainVal Set (Linear Px Error)")
    print_leaderboard(stats_test, "Leaderboard: Test Set (Linear Px Error)")
    print_leaderboard(stats_total, "Leaderboard: GLOBAL (Linear Px Error)")
    run_visualization_on_selected()

if __name__ == '__main__':
    plt.rcParams['backend'] = 'Agg'
    main()