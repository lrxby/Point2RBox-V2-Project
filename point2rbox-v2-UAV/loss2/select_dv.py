import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing

# ================= 配置区域 =================
# 1. 数据集根路径 (包含 train、val、test 文件夹的上级目录)
DATASET_ROOT = '/mnt/data/xiekaikai/DroneVehicle'
# 指定需要合并统计的子集
TARGET_SUBSETS = ['train', 'val', 'test']

# 2. 这里的图像尺寸用于 'image-norm' (DOTA 切图通常是 1024)
IMG_W, IMG_H = 1024, 1024

# 3. 超参数搜索空间 (Grid Search Space)
# (1) 正则化强度: 覆盖从极小到较大的范围
LAMBDA_LIST = [1e-8, 1e-7, 1e-6, 1e-4, 1e-3, 0.01, 0.1, 1.0, 2, 3, 4, 5.0]

# (2) 归一化方式: 对比 Z-Score 和 Image-Norm
NORM_TYPES = ['z-score', 'image-norm', 'none']

# (3) 拟合形式: 再次确认 Log 是否稳坐第一
MODES = ['log', 'linear', 'sqrt', 'square']

# 4. 其他配置
CLASSES = ('car', 'bus', 'truck', 'van', 'freight_car')
EPS = 1e-6
WORKER_NUM = max(1, multiprocessing.cpu_count() - 4) # 留点余地
MAX_FILES = None # 设置为 None 则跑全量数据，设置为 2000 可快速验证
# ===========================================

def polygon_area(coords):
    x = coords[0::2]
    y = coords[1::2]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def parse_txt(txt_path):
    """解析 DroneVehicle 格式 LabelTxt 文件"""
    bboxes = []
    labels = []
    try:
        with open(txt_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 9: continue
                
                # DroneVehicle 格式: x1 y1 ... x4 y4 类别索引
                try:
                    cls_id = int(parts[8])  # 第9个字段是类别索引
                except ValueError:
                    continue
                
                # 过滤非法类别索引
                if cls_id < 0 or cls_id >= len(CLASSES):
                    continue
                cls_name = CLASSES[cls_id]  # 转换为类别名称（保持后续逻辑兼容）
                
                coords = list(map(float, parts[:8]))
                area = polygon_area(coords)
                if area <= 1: continue 
                
                s = np.sqrt(area)
                s = max(s, 1e-2)
                
                cx = sum(coords[0::2]) / 4.0
                cy = sum(coords[1::2]) / 4.0
                
                bboxes.append([cx, cy, s, s]) 
                labels.append(cls_name)
    except Exception:
        pass
    return np.array(bboxes), labels

def solve_perspective(bboxes, labels_str, mode, ridge_lambda, norm_type):
    """
    核心解算器：针对一种特定的参数组合进行拟合和评估
    """
    N = len(bboxes)
    unique_labels = sorted(list(set(labels_str)))
    K = len(unique_labels)
    
    # 约束检查
    if N < K + 3: return None

    cls_to_idx = {name: i for i, name in enumerate(unique_labels)}
    labels_idx = np.array([cls_to_idx[name] for name in labels_str])

    # === 1. 准备数据 ===
    x_c = bboxes[:, 0]
    y_c = bboxes[:, 1]
    w = bboxes[:, 2] # w=s
    h = bboxes[:, 3] # h=s
    s_gt = np.sqrt(w * h)

    # === 2. 目标变量变换 (Mode) ===
    if mode == 'log':
        Y = 0.5 * np.log(w * h)
    elif mode == 'linear':
        Y = s_gt
    elif mode == 'sqrt':
        Y = np.sqrt(s_gt)
    elif mode == 'square':
        Y = w * h

    # === 3. 坐标归一化 (Norm Type) ===
    if norm_type == 'z-score':
        x_mean, x_std = np.mean(x_c), np.std(x_c)
        y_mean, y_std = np.mean(y_c), np.std(y_c)
        x_std = max(x_std, EPS)
        y_std = max(y_std, EPS)
        x_norm = (x_c - x_mean) / x_std
        y_norm = (y_c - y_mean) / y_std
        
    elif norm_type == 'image-norm':
        # 映射到 [-1, 1]
        x_norm = (x_c - IMG_W / 2.0) / (IMG_W / 2.0)
        y_norm = (y_c - IMG_H / 2.0) / (IMG_H / 2.0)
        
    else: # 'none'
        x_norm = x_c
        y_norm = y_c

    # === 4. 构建矩阵与求解 ===
    A = np.zeros((N, 2 + K))
    A[:, 0] = x_norm
    A[:, 1] = y_norm
    for i, idx in enumerate(labels_idx):
        A[i, 2 + idx] = 1.0

    M = A.T @ A
    I_reg = np.eye(2 + K) * ridge_lambda # 使用传入的 lambda
    
    try:
        theta = np.linalg.inv(M + I_reg) @ (A.T @ Y)
    except np.linalg.LinAlgError:
        return None # 奇异矩阵

    # === 5. 还原预测值 ===
    Y_hat = A @ theta
    
    if mode == 'log':
        s_pred = np.exp(Y_hat)
    elif mode == 'linear':
        s_pred = Y_hat
    elif mode == 'sqrt':
        s_pred = np.maximum(Y_hat, 0) ** 2
    elif mode == 'square':
        s_pred = np.sqrt(np.maximum(Y_hat, 0))
    
    # === 6. 计算指标 ===
    # IoU
    area_gt = s_gt ** 2
    area_pred = s_pred ** 2
    area_gt = np.maximum(area_gt, EPS)
    area_pred = np.maximum(area_pred, EPS)
    iou = np.minimum(area_gt, area_pred) / np.maximum(area_gt, area_pred)
    mean_iou = np.mean(iou)

    # MAPE
    diff_abs = np.abs(s_gt - s_pred)
    mape = diff_abs / (s_gt + EPS)
    mean_mape = np.mean(mape)

    return mean_iou, mean_mape

def process_file_grid_search(txt_path):
    """
    单个文件处理函数：对该文件跑完所有的参数组合
    返回：{ (mode, lambda, norm): (iou, mape, count=1), ... }
    """
    bboxes, labels = parse_txt(txt_path)
    if len(bboxes) == 0: return None

    results = {}
    
    # 三层循环遍历所有组合
    for mode in MODES:
        for ridge_lambda in LAMBDA_LIST:
            for norm_type in NORM_TYPES:
                
                res = solve_perspective(bboxes, labels, mode, ridge_lambda, norm_type)
                
                key = (mode, ridge_lambda, norm_type)
                if res is not None:
                    # 记录 (IoU, MAPE, 有效样本数)
                    results[key] = (res[0], res[1], len(bboxes))
                else:
                    results[key] = (0.0, 0.0, 0) # 失败标记

    return results

def main():
    print(f"🚀 启动超参数自动搜索...")
    print(f"📂 数据集根目录: {DATASET_ROOT}")
    print(f"🎯 目标子集: {TARGET_SUBSETS}")
    print(f"⚙️  搜索空间: {len(MODES)} Modes x {len(LAMBDA_LIST)} Lambdas x {len(NORM_TYPES)} Norms = {len(MODES)*len(LAMBDA_LIST)*len(NORM_TYPES)} Combinations")
    
    # 收集所有文件
    txt_files = []
    for subset in TARGET_SUBSETS:
        # 适配 DroneVehicle 目录结构: subset/annfiles
        subset_path = os.path.join(DATASET_ROOT, subset, 'annfiles')
        if os.path.exists(subset_path):
            files = glob.glob(os.path.join(subset_path, '*.txt'))
            txt_files.extend(files)
            print(f"  - {subset}: 找到 {len(files)} 个标注文件")
        else:
            print(f"  - {subset}: 路径不存在 {subset_path}")

    if len(txt_files) == 0:
        print("❌ 未找到任何标注文件，请检查路径。")
        return

    if MAX_FILES:
        txt_files = txt_files[:MAX_FILES]
        print(f"⚠️  仅使用前 {MAX_FILES} 个文件进行快速验证")
    else:
        print(f"✅ 将计算所有 {len(txt_files)} 个文件")
    
    # 初始化全局统计字典
    # global_stats[key] = {'total_iou': 0, 'total_mape': 0, 'total_samples': 0}
    global_stats = {}
    # 初始化 keys
    for mode in MODES:
        for l in LAMBDA_LIST:
            for n in NORM_TYPES:
                global_stats[(mode, l, n)] = {'sum_iou': 0.0, 'sum_mape': 0.0, 'total_samples': 0}

    # 多进程处理
    with multiprocessing.Pool(WORKER_NUM) as pool:
        for file_res in tqdm(pool.imap_unordered(process_file_grid_search, txt_files), total=len(txt_files)):
            if file_res is None: continue
            
            for key, val in file_res.items():
                mean_iou, mean_mape, n_objs = val
                if n_objs > 0:
                    # 还原为 sum，以便全局累加
                    global_stats[key]['sum_iou'] += mean_iou * n_objs
                    global_stats[key]['sum_mape'] += mean_mape * n_objs
                    global_stats[key]['total_samples'] += n_objs

    # 汇总结果为 DataFrame
    rows = []
    for key, val in global_stats.items():
        mode, ridge_lambda, norm_type = key
        total = val['total_samples']
        if total > 0:
            final_iou = val['sum_iou'] / total
            final_mape = val['sum_mape'] / total
            rows.append({
                'Mode': mode,
                'Lambda': ridge_lambda,
                'Norm': norm_type,
                'IoU (%)': final_iou * 100,
                'MAPE (%)': final_mape * 100,
                'Samples': total
            })
    
    df = pd.DataFrame(rows)
    
    # 排序：优先看 IoU (降序)，其次看 MAPE (升序)
    df = df.sort_values(by=['IoU (%)', 'MAPE (%)'], ascending=[False, True])
    
    print("\n" + "="*100)
    print("🏆 超参数搜索最佳结果 Top 20 (按 IoU 排序)")
    print("="*100)
    pd.set_option('display.max_rows', 20)
    pd.set_option('display.width', 1000)
    print(df.head(20).to_string(index=False))
    
    # 找出每个 Mode 的最佳配置
    print("\n" + "="*100)
    print("🥇 各拟合模式的最佳配置")
    print("="*100)
    best_per_mode = df.loc[df.groupby('Mode')['IoU (%)'].idxmax()]
    print(best_per_mode.sort_values(by='IoU (%)', ascending=False).to_string(index=False))

    # 保存完整结果
    save_path = './select/dv.csv'
    # 确保目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"\n💾 完整结果已保存至: {save_path}")

if __name__ == '__main__':
    multiprocessing.set_start_method('fork', force=True)
    main()
