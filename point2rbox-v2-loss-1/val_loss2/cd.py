import os
import glob
import numpy as np
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# ================= 配置区域 =================
DATASET_ROOT = '/mnt/data/xiekaikai/split_ss_codrone'
SUBSETS = ['trainval', 'test']
OUTPUT_DIR = 'codrone'

CLASSES = ('car', 'truck', 'bus', 'traffic-light',
           'traffic-sign', 'bridge', 'people', 'bicycle',
           'motor', 'tricycle', 'boat', 'ship')

# CODrone 中大部分类别受透视影响，可视为广义刚性
RIGID_CLASSES = {'car', 'truck', 'bus', 'traffic-light', 'traffic-sign'}

MIN_INSTANCES = 2 
CLASSES_SET = set(CLASSES)

# ================= 核心算法 =================
def compute_metrics(y_list, s_list):
    """
    计算 PASCL 相对损失 (Relative Error)
    """
    n = len(y_list)
    if n < MIN_INSTANCES:
        return []

    y = np.array(y_list, dtype=np.float32)
    s = np.array(s_list, dtype=np.float32)
    
    # 拟合趋势线
    y_bar = np.mean(y)
    s_bar = np.mean(s)
    
    numerator = np.sum((y - y_bar) * (s - s_bar))
    denominator = np.sum((y - y_bar) ** 2)
    
    # CODrone 具有非常强的透视性，k 会显著大于 0
    if denominator < 1e-6:
        k = 0.0
    else:
        k = numerator / denominator

    b = s_bar - k * y_bar
    s_hat = k * y + b
    
    # 计算相对误差 (Relative Error)
    abs_loss = np.abs(s - s_hat)
    s_safe = np.maximum(s, 1e-2)
    rel_loss = abs_loss / s_safe
    
    return rel_loss.tolist()

# ================= 工具函数 =================
def fast_poly2metrics(p):
    """纯 Python 几何计算"""
    cy = (p[1] + p[3] + p[5] + p[7]) / 4.0
    term1 = p[0]*p[3] + p[2]*p[5] + p[4]*p[7] + p[6]*p[1]
    term2 = p[1]*p[2] + p[3]*p[4] + p[5]*p[6] + p[7]*p[0]
    area = 0.5 * abs(term1 - term2)
    return cy, math.sqrt(area)

def process_single_file(file_path):
    """
    处理单个 CODrone 标注文件
    """
    local_rel = {c: [] for c in CLASSES}
    temp_data = {c: {'y': [], 's': []} for c in CLASSES}
    
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            parts = line.strip().split()
            # CODrone: x1 y1 ... x4 y4 classname difficulty
            if len(parts) < 10: continue
            
            classname = parts[8]
            
            # 过滤 ignored
            if classname == 'ignored': continue
            if classname not in CLASSES_SET: continue
            
            # 过滤 difficulty
            try:
                difficulty = int(parts[9])
            except:
                difficulty = 0
            if difficulty != 0: continue

            try:
                poly = [float(x) for x in parts[:8]]
                
                # 计算指标
                cy, s = fast_poly2metrics(poly)
                if s < 1: continue 

                temp_data[classname]['y'].append(cy)
                temp_data[classname]['s'].append(s)
            except:
                continue
            
        for cls_name in CLASSES:
            data = temp_data[cls_name]
            if len(data['y']) >= MIN_INSTANCES:
                r_loss = compute_metrics(data['y'], data['s'])
                local_rel[cls_name].extend(r_loss)
                
    except Exception:
        pass
        
    return local_rel

def plot_histogram_advanced(loss_data, class_name, save_dir, stats):
    """绘制高级分布直方图 (相对误差)"""
    if len(loss_data) == 0: return

    plt.figure(figsize=(16, 9))
    
    # 限制显示范围
    display_data = loss_data[loss_data <= 2.0]
    if len(display_data) == 0: display_data = loss_data
    
    bins = np.linspace(0, 2.0, 50) 
    n, bins, patches = plt.hist(display_data, bins=bins, color='#ff7043', edgecolor='black', alpha=0.8)
    
    max_height = 0
    for rect in patches:
        height = rect.get_height()
        max_height = max(max_height, height)

    plt.axvline(stats['mean'], color='red', linestyle='dashed', linewidth=2, label=f"Mean: {stats['mean']:.4f}")
    plt.axvline(stats['p50'], color='blue', linestyle='dashed', linewidth=2, label=f"Median: {stats['p50']:.4f}")
    
    # 详细统计数据展板
    stats_text = (
        f"Class: {class_name}\n"
        f"Type: {'RIGID' if class_name in RIGID_CLASSES else 'Non-Rigid'}\n"
        f"Count: {stats['count']}\n"
        f"{'-'*20}\n"
        f"Mean: {stats['mean']:.4f}\n"
        f"Min:  {stats['min']:.4f}\n"
        f"Max:  {stats['max']:.4f}\n"
        f"{'-'*20}\n"
        f"P10:  {stats['p10']:.4f}\n"
        f"P20:  {stats['p20']:.4f}\n"
        f"P30:  {stats['p30']:.4f}\n"
        f"P40:  {stats['p40']:.4f}\n"
        f"P50:  {stats['p50']:.4f}\n"
        f"P60:  {stats['p60']:.4f}\n"
        f"P70:  {stats['p70']:.4f}\n"
        f"P80:  {stats['p80']:.4f}\n"
        f"P90:  {stats['p90']:.4f}"
    )
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
    plt.text(0.98, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='right', bbox=props, fontfamily='monospace')

    plt.ylim(0, max_height * 1.15)
    plt.title(f'PASCL Relative Error: {class_name}', fontsize=18)
    plt.xlabel('Relative Error (|s - s_hat| / s)', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.legend(fontsize=12, loc='upper center')
    plt.grid(axis='y', alpha=0.3)
    
    save_path = os.path.join(save_dir, f'{class_name}_dist.png')
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()

# ================= 主流程 =================
def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    files = []
    print("Scanning CODrone files...")
    for subset in SUBSETS:
        path = os.path.join(DATASET_ROOT, subset, 'annfiles', '*.txt')
        files.extend(glob.glob(path))
    
    print(f"Found {len(files)} files.")
    num_workers = max(1, cpu_count() - 2)
    print(f"Processing with {num_workers} workers...")
    
    agg_rel = {c: [] for c in CLASSES}
    
    with Pool(processes=num_workers) as pool:
        results = list(tqdm(pool.imap_unordered(process_single_file, files, chunksize=10), total=len(files)))
        
    print("Aggregating results...")
    for res_rel in results:
        for c in CLASSES:
            if res_rel[c]: agg_rel[c].extend(res_rel[c])

    # 生成报告头
    header_str = "{:<20} | {:<8} | {:<8} | {:<8} | {:<8} | {:<8} | {:<8} | {:<8} | {:<8} | {:<8} | {:<8} | {:<8} | {:<8} | {:<8}".format(
        "Class", "Count", "Mean", "Min", "Max", 
        "P10", "P20", "P30", "P40", "P50", "P60", "P70", "P80", "P90"
    )
    row_template = "{:<20} | {:<8} | {:<8.4f} | {:<8.4f} | {:<8.4f} | {:<8.4f} | {:<8.4f} | {:<8.4f} | {:<8.4f} | {:<8.4f} | {:<8.4f} | {:<8.4f} | {:<8.4f} | {:<8.4f}"
    
    print("\n" + "="*160)
    print(header_str)
    print("-" * 160)
    
    report_path = os.path.join(OUTPUT_DIR, 'statistics_report.txt')
    
    with open(report_path, 'w') as f_report:
        f_report.write(header_str + "\n" + "-"*160 + "\n")
        
        for cls in CLASSES:
            data = np.array(agg_rel[cls])
            
            if len(data) == 0:
                continue
            
            # 计算统计量
            mean_val = np.mean(data)
            min_val = np.min(data)
            max_val = np.max(data)
            pcts = np.percentile(data, np.arange(10, 100, 10))
            
            stats = {
                'count': len(data),
                'mean': mean_val, 'min': min_val, 'max': max_val,
                'p10': pcts[0], 'p20': pcts[1], 'p30': pcts[2],
                'p40': pcts[3], 'p50': pcts[4], 'p60': pcts[5],
                'p70': pcts[6], 'p80': pcts[7], 'p90': pcts[8]
            }
            
            row_str = row_template.format(
                cls, len(data), mean_val, min_val, max_val,
                stats['p10'], stats['p20'], stats['p30'], stats['p40'], 
                stats['p50'], stats['p60'], stats['p70'], stats['p80'], stats['p90']
            )
            
            print(row_str)
            f_report.write(row_str + "\n")
            
            # 绘制直方图
            plot_histogram_advanced(data, cls, OUTPUT_DIR, stats)

    print("="*160)
    print(f"Report saved to: {os.path.abspath(report_path)}")
    print(f"Plots saved to: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == '__main__':
    main()