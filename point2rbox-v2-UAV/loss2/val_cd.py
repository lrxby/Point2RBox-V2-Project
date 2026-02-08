import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing

# ================= 配置区域 =================
# 1. 数据集路径
FULL_TRAINVAL_DIR = '/mnt/data/xiekaikai/split_ss_codrone/trainval/labelTxt'
FULL_TEST_DIR = '/mnt/data/xiekaikai/split_ss_codrone/test/labelTxt'

# 2. 对比模式
MODES = ['log', 'linear', 'sqrt', 'square']

# 3. 必须与 Loss 保持一致的参数
CLASSES = ('car', 'truck', 'bus', 'traffic-light',
           'traffic-sign', 'bridge', 'people', 'bicycle',
           'motor', 'tricycle', 'boat', 'ship')
RIDGE_LAMBDA = 1e-4  # 与 Loss 保持一致
EPS = 1e-6           # 与 Loss 的 clamp 保持一致

# 4. 结果保存路径
SAVE_DIR = '/mnt/data/liurunxiang/workplace/point2rbox-v2-UAV/loss2/val/codrone'

# 5. 多进程配置：自动获取CPU核心数，预留2核保证系统稳定
WORKER_NUM = max(1, multiprocessing.cpu_count() - 2)
# ===========================================

def polygon_area(coords):
    x = coords[0::2]
    y = coords[1::2]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def parse_txt(txt_path):
    """解析 GT 数据"""
    bboxes = []
    labels = []
    
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
            
            bboxes.append([cx, cy, s, s]) # [x, y, w, h]
            labels.append(cls_name)
            
    return np.array(bboxes), labels

def fit_by_loss_logic(bboxes, labels_str, mode='log'):
    """损失计算逻辑"""
    if len(bboxes) == 0: return None
    
    # === 1. 准备数据 ===
    x_c = bboxes[:, 0]
    y_c = bboxes[:, 1]
    w = bboxes[:, 2]
    h = bboxes[:, 3]
    
    # 物理真值 (Ground Truth Size)
    s_gt = np.sqrt(w * h)

    # === 构造 Target Y ===
    if mode == 'log':
        Y = 0.5 * np.log(w * h)
    elif mode == 'linear':
        Y = s_gt
    elif mode == 'sqrt':
        Y = np.sqrt(s_gt)
    elif mode == 'square':
        Y = w * h
    
    # === 2. 检查约束 ===
    unique_labels = sorted(list(set(labels_str)))
    K = len(unique_labels)
    N = len(bboxes)
    
    if N < K + 3: return None

    cls_to_idx = {name: i for i, name in enumerate(unique_labels)}
    labels_idx = np.array([cls_to_idx[name] for name in labels_str])

    # === 3. 归一化 (Z-Score) ===
    x_mean, x_std = np.mean(x_c), np.std(x_c)
    y_mean, y_std = np.mean(y_c), np.std(y_c)
    x_std = max(x_std, EPS)
    y_std = max(y_std, EPS)
    x_norm = (x_c - x_mean) / x_std
    y_norm = (y_c - y_mean) / y_std

    # === 4. 构建矩阵 A ===
    A = np.zeros((N, 2 + K))
    A[:, 0] = x_norm
    A[:, 1] = y_norm
    for i, idx in enumerate(labels_idx):
        A[i, 2 + idx] = 1.0

    # === 5. 求解 theta ===
    M = A.T @ A
    I_reg = np.eye(2 + K) * RIDGE_LAMBDA
    try:
        theta = np.linalg.inv(M + I_reg) @ (A.T @ Y)
    except np.linalg.LinAlgError:
        return None

    # === 6. 预测与还原 (Inverse Transform) ===
    Y_hat = A @ theta
    
    if mode == 'log':
        s_pred = np.exp(Y_hat)
    elif mode == 'linear':
        s_pred = Y_hat
    elif mode == 'sqrt':
        s_pred = np.maximum(Y_hat, 0) ** 2
    elif mode == 'square':
        s_pred = np.sqrt(np.maximum(Y_hat, 0))
        
    # === 7. 计算多种误差指标 (按物体) ===
    # 7.1 绝对误差 MAE (px)
    diff_abs = np.abs(s_gt - s_pred)
    
    # 7.2 相对误差 MAPE (%)
    diff_rel = diff_abs / (s_gt + EPS)
    
    # 7.3 均方误差 MSE (px^2)
    diff_mse = (s_gt - s_pred) ** 2
    
    # 7.4 Size IoU (0~1)
    area_gt = s_gt ** 2
    area_pred = s_pred ** 2
    area_gt = np.maximum(area_gt, EPS)
    area_pred = np.maximum(area_pred, EPS)
    
    inter = np.minimum(area_gt, area_pred)
    union = np.maximum(area_gt, area_pred)
    iou = inter / union
    
    return diff_abs, diff_rel, diff_mse, iou, labels_str

# ================= 单文件并行处理函数 =================
def process_single_file(txt_path):
    """封装单个文件的完整处理逻辑"""
    try:
        bboxes, labels = parse_txt(txt_path)
        current_res = {}
        # 遍历所有模式计算
        for mode in MODES:
            res = fit_by_loss_logic(bboxes, labels, mode)
            if res is None:
                return None
            current_res[mode] = res
        return current_res
    except Exception:
        return None

# ================= 多进程数据收集函数 =================
def collect_dataset_stats(ann_dir, dataset_name):
    """多进程并行收集数据集统计信息"""
    print(f"\n🚀 正在分析数据集: {dataset_name} ...")
    txt_files = glob.glob(os.path.join(ann_dir, '*.txt'))
    
    # 存储结构
    metrics = ['abs', 'rel', 'mse', 'iou']
    stats = {m: {c: {met: [] for met in metrics} for c in CLASSES} for m in MODES}
    
    valid_count = 0
    # 多进程池并行处理文件
    with multiprocessing.Pool(WORKER_NUM) as pool:
        # 并行执行 + 进度条展示
        results = list(tqdm(
            pool.imap(process_single_file, txt_files),
            total=len(txt_files),
            desc=f"Fitting {dataset_name}"
        ))

    # 聚合所有有效计算结果
    for res in results:
        if res is not None:
            valid_count += 1
            # 按模式、类别归档指标数据
            for mode in MODES:
                d_abs, d_rel, d_mse, d_iou, obj_labels = res[mode]
                for i, cls in enumerate(obj_labels):
                    stats[mode][cls]['abs'].append(d_abs[i])
                    stats[mode][cls]['rel'].append(d_rel[i])
                    stats[mode][cls]['mse'].append(d_mse[i])
                    stats[mode][cls]['iou'].append(d_iou[i])

    print(f"✅ {dataset_name} 有效图片数: {valid_count}")
    return stats

def merge_stats(stats1, stats2):
    """合并两个统计字典"""
    merged = {m: {c: {met: [] for met in ['abs', 'rel', 'mse', 'iou']} for c in CLASSES} for m in MODES}
    for mode in MODES:
        for cls in CLASSES:
            for met in ['abs', 'rel', 'mse', 'iou']:
                merged[mode][cls][met] = stats1[mode][cls][met] + stats2[mode][cls][met]
    return merged

def print_and_save_stats(stats, title="Dataset Report", save_filename="report.txt"):
    """打印与保存TXT报告（已新增 MAE 指标）"""
    summary_data = []
    
    # 遍历每个类别构建数据行
    for cls in CLASSES:
        row = {'Class': cls}
        sample_count = len(stats['log'][cls]['abs'])
        row['Samples'] = sample_count
        
        if sample_count == 0:
            for mode in MODES:
                row[f'{mode}_IoU'] = np.nan
                row[f'{mode}_MAE'] = np.nan  # [新增]
                row[f'{mode}_MAPE'] = np.nan
                row[f'{mode}_MSE'] = np.nan
            summary_data.append(row)
            continue

        for mode in MODES:
            row[f'{mode}_IoU'] = np.mean(stats[mode][cls]['iou']) * 100
            row[f'{mode}_MAE'] = np.mean(stats[mode][cls]['abs'])       # [新增] 平均绝对误差 (像素)
            row[f'{mode}_MAPE'] = np.mean(stats[mode][cls]['rel']) * 100
            row[f'{mode}_MSE'] = np.mean(stats[mode][cls]['mse'])
            
        summary_data.append(row)
        
    # 计算全局汇总指标
    total_row = {'Class': 'GLOBAL_ALL', 'Samples': 0}
    for mode in MODES:
        all_metrics = {'abs':[], 'rel':[], 'mse':[], 'iou':[]}
        for cls in CLASSES:
            for met in all_metrics:
                all_metrics[met].extend(stats[mode][cls][met])
        
        total_row['Samples'] = len(all_metrics['abs'])
        if total_row['Samples'] > 0:
            total_row[f'{mode}_IoU'] = np.mean(all_metrics['iou']) * 100
            total_row[f'{mode}_MAE'] = np.mean(all_metrics['abs'])      # [新增]
            total_row[f'{mode}_MAPE'] = np.mean(all_metrics['rel']) * 100
            total_row[f'{mode}_MSE'] = np.mean(all_metrics['mse'])
        else:
            total_row[f'{mode}_IoU'] = np.nan
            total_row[f'{mode}_MAE'] = np.nan
            total_row[f'{mode}_MAPE'] = np.nan
            total_row[f'{mode}_MSE'] = np.nan
        
    summary_data.append(total_row)
    
    df = pd.DataFrame(summary_data)
    # 调整列顺序，新增 MAE
    cols = ['Class', 'Samples']
    # 顺序：IoU (越高越好), MAE (越低越好), MAPE (越低越好), MSE (越低越好)
    for met in ['IoU', 'MAE', 'MAPE', 'MSE']:
        for mode in MODES:
            cols.append(f'{mode}_{met}')
    df = df[cols]

    # 配置格式化输出参数
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 2000)
    pd.set_option('display.float_format', '{:.2f}'.format)
    
    # 构建完整输出文本内容
    split_line = "="*160
    content = f"\n{split_line}\n📊 {title}\n{split_line}\n"
    content += df.to_string(index=False)
    content += f"\n{split_line}\n"

    # 控制台打印
    print(content)

    # 保存为格式化TXT文件
    save_path = os.path.join(SAVE_DIR, save_filename)
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"\n💾 报表已保存至: {save_path}")

def main():
    # 自动创建输出目录
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"📂 结果将保存至目录: {SAVE_DIR}")
    print(f"⚡ 多进程加速已启用，工作进程数：{WORKER_NUM}")
    
    # 执行数据分析
    stats_trainval = collect_dataset_stats(FULL_TRAINVAL_DIR, "TrainVal Set")
    stats_test = collect_dataset_stats(FULL_TEST_DIR, "Test Set")
    stats_global = merge_stats(stats_trainval, stats_test)
    
    # 打印并保存三份报告
    print_and_save_stats(
        stats_trainval,
        title="详细对比报告: TrainVal Set | 指标：IoU(↑), MAE(px)(↓), MAPE(%)(↓), MSE(↓)",
        save_filename="trainval_set_report.txt"
    )
    print_and_save_stats(
        stats_test,
        title="详细对比报告: Test Set | 指标：IoU(↑), MAE(px)(↓), MAPE(%)(↓), MSE(↓)",
        save_filename="test_set_report.txt"
    )
    print_and_save_stats(
        stats_global,
        title="详细对比报告: GLOBAL (TrainVal + Test) | 指标：IoU(↑), MAE(px)(↓), MAPE(%)(↓), MSE(↓)",
        save_filename="global_set_report.txt"
    )

    # 结果解读提示
    print("\n💡 结果解读建议:")
    print("1. [IoU]: 最重要的准确性指标。")
    print("2. [MAE]: 平均像素误差。Log 模式若在此指标上也领先，说明它不仅比例准，绝对值也准。")
    print("3. [MAPE]: 相对误差，体现对小目标的友好程度。")
    print("4. [MSE]: 对离群点敏感。")

if __name__ == '__main__':
    # 适配Linux系统，设置多进程启动方式
    multiprocessing.set_start_method('fork', force=True)
    main()