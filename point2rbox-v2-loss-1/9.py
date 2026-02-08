import os
import glob
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ================= 配置区域 =================
# 数据集标注路径 (根据你提供的信息修改)
ANNOTATION_PATH = '/mnt/data/xiekaikai/split_ss_dota/trainval/annfiles'

# 结果保存文件夹
OUT_DIR = 'work_dirs/analysis_results_split_dota'

# 刚性物体类别
RIGID_CLASSES = [
'plane', 'baseball-diamond', 'bridge', 'ground-track-field',
         'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
         'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout',
         'harbor', 'swimming-pool', 'helicopter'
]
# ===========================================

def parse_dota_line(line):
    """解析 DOTA 标注行"""
    parts = line.strip().split()
    if len(parts) < 10: return None, None
    category = parts[8]
    if category not in RIGID_CLASSES: return None, None
        
    try:
        poly = list(map(float, parts[:8]))
        poly = np.array(poly, dtype=np.float32).reshape(4, 2)
        cy = np.mean(poly[:, 1]) # Y坐标
        # 鞋带公式算面积 -> 开根号得到 Scale
        x, y = poly[:, 0], poly[:, 1]
        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        scale = np.sqrt(area)
        return category, (cy, scale)
    except ValueError:
        return None, None

def generate_text_report(stats, out_path):
    """生成并打印数值统计报告"""
    lines = []
    lines.append("="*60)
    lines.append(f"{'Class':<15} | {'Images':<6} | {'Median CV':<10} | {'CV<0.15(%)':<10} | {'Avg |Corr|':<10}")
    lines.append("-" * 60)
    
    # 阈值定义
    CV_THRESHOLD = 0.15
    CORR_THRESHOLD = 0.6 

    for c in RIGID_CLASSES:
        cvs = np.array(stats[c]['cvs'])
        corrs = np.array(stats[c]['corrs'])
        
        n_imgs = len(cvs)
        if n_imgs == 0:
            lines.append(f"{c:<15} | {'0':<6} | {'N/A':<10} | {'N/A':<10} | {'N/A':<10}")
            continue
            
        # 1. 尺寸一致性指标
        median_cv = np.median(cvs)
        consistent_ratio = np.sum(cvs < CV_THRESHOLD) / n_imgs * 100
        
        # 2. 透视线性指标 (只统计有一定尺寸波动的图片)
        # 如果 CV 很小，相关系数是噪声，不计入
        valid_indices = np.where(cvs > 0.05)[0]
        if len(valid_indices) > 0:
            valid_corrs = corrs[valid_indices]
            # 计算相关系数的绝对值平均数 (不管正相关还是负相关，只要有相关性就行)
            avg_abs_corr = np.mean(np.abs(valid_corrs))
        else:
            avg_abs_corr = 0.0
            
        lines.append(f"{c:<15} | {n_imgs:<6} | {median_cv:<10.3f} | {consistent_ratio:<10.1f} | {avg_abs_corr:<10.3f}")

    lines.append("="*60)
    lines.append("\n[结论判定参考]:")
    lines.append("1. Median CV < 0.15 且 CV<0.15(%) > 80%  -> 强一致性 (适合用全图均值约束)")
    lines.append("2. Avg |Corr| > 0.6                    -> 强透视性 (适合用线性回归约束)")
    lines.append("3. 都不满足                            -> 建议用局部(邻域)一致性约束")
    
    report_str = "\n".join(lines)
    print(report_str) # 打印到终端
    
    with open(out_path, 'w') as f:
        f.write(report_str)
    print(f"\n[提示] 完整报告已保存至: {out_path}")

def verify_prior():
    if not os.path.exists(ANNOTATION_PATH):
        print(f"错误: 路径不存在 -> {ANNOTATION_PATH}")
        return
    txt_files = glob.glob(os.path.join(ANNOTATION_PATH, '*.txt'))
    print(f"-> 发现 {len(txt_files)} 个标注文件，正在扫描数据...")
    
    os.makedirs(OUT_DIR, exist_ok=True)
    stats = {c: {'cvs': [], 'corrs': []} for c in RIGID_CLASSES}
    vis_samples = []

    for txt_file in tqdm(txt_files):
        filename = os.path.basename(txt_file)
        with open(txt_file, 'r') as f:
            lines = f.readlines()
            
        img_data = {c: [] for c in RIGID_CLASSES}
        for line in lines:
            cat, data = parse_dota_line(line)
            if cat and data[1] > 5: 
                img_data[cat].append(data)
                
        for cls_name, points in img_data.items():
            if len(points) < 5: continue # 样本太少不统计
                
            points = np.array(points)
            ys, scales = points[:, 0], points[:, 1]
            
            # 指标计算
            mean_s = np.mean(scales)
            std_s = np.std(scales)
            cv = std_s / (mean_s + 1e-6)
            
            corr = 0
            if std_s > 0 and np.std(ys) > 0:
                corr = np.corrcoef(ys, scales)[0, 1]
            
            if cv < 1.0: # 过滤异常
                stats[cls_name]['cvs'].append(cv)
                if not np.isnan(corr):
                    stats[cls_name]['corrs'].append(corr)
            
            # 绘图采样
            if len(vis_samples) < 24 and (cv < 0.1 or abs(corr) > 0.7):
                vis_samples.append({'file': filename, 'class': cls_name, 
                                    'ys': ys, 'scales': scales, 'cv': cv, 'corr': corr})

    # 生成文本报告 (核心!)
    report_path = os.path.join(OUT_DIR, 'analysis_report.txt')
    generate_text_report(stats, report_path)

    # 绘图部分 (保留以备不时之需)
    print("-> 正在生成图表...")
    # 图1: CV箱线图
    plt.figure(figsize=(12, 6))
    cv_data, cv_labels = [], []
    for c in RIGID_CLASSES:
        vals = stats[c]['cvs']
        cv_data.extend(vals)
        cv_labels.extend([c]*len(vals))
    if cv_data:
        sns.boxplot(x=cv_labels, y=cv_data, showfliers=False)
        plt.axhline(0.15, color='r', linestyle='--')
        plt.title('Scale Consistency (CV) Distribution'); plt.xticks(rotation=45); plt.tight_layout()
        plt.savefig(f'{OUT_DIR}/1_cv_boxplot.png'); plt.close()
    
    # 图3: 相关系数直方图
    plt.figure(figsize=(8, 5))
    corr_data = []
    for c in RIGID_CLASSES:
        # 只看那些大小确实有差异的样本(CV>0.05)
        valid_idxs = np.where(np.array(stats[c]['cvs']) > 0.05)[0]
        if len(valid_idxs) > 0:
            corr_data.extend(np.array(stats[c]['corrs'])[valid_idxs])
    if corr_data:
        sns.histplot(corr_data, bins=30, kde=True)
        plt.axvline(0, color='k', linestyle='--'); plt.title('Correlation Distribution'); plt.tight_layout()
        plt.savefig(f'{OUT_DIR}/3_correlation_hist.png'); plt.close()

    print(f"-> 结束。请复制上方打印的表格数据。")

if __name__ == '__main__':
    verify_prior()