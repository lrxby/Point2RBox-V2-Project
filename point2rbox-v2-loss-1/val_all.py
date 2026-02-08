import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from tqdm import tqdm
import math

# ================= 1. 配置区域 (Configuration) =================

DATASETS_CONFIG = {
    'DOTA': {
        'root': '/mnt/data/xiekaikai/split_ss_dota',
        'subsets': ['trainval'],
        'classes': ('plane', 'baseball-diamond', 'bridge', 'ground-track-field',
                    'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
                    'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout',
                    'harbor', 'swimming-pool', 'helicopter'),
        'format': 'dota',
        'ext': 'txt',
        'rigid_classes': ['small-vehicle', 'large-vehicle', 'plane', 'storage-tank', 'helicopter']
    },
    'DroneVehicle': {
        'root': '/mnt/data/xiekaikai/DroneVehicle',
        'subsets': ['train', 'val'],
        'classes': ('car', 'bus', 'truck', 'van', 'freight_car'),
        'format': 'dronevehicle',
        'ext': 'txt',
        'rigid_classes': ['car', 'bus', 'truck', 'van', 'freight_car'] 
    },
    'CODrone': {
        'root': '/mnt/data/xiekaikai/split_ss_codrone',
        'subsets': ['trainval'],
        'classes': ('car', 'truck', 'bus', 'traffic-light',
                    'traffic-sign', 'bridge', 'people', 'bicycle',
                    'motor', 'tricycle', 'boat', 'ship'),
        'format': 'codrone',
        'ext': 'txt',
        'rigid_classes': ['car', 'truck', 'bus', 'traffic-light', 'traffic-sign']
    }
}

# 统计阈值
MIN_INSTANCES_PER_CLASS = 5   # 单张图单类别最少实例数
MIN_AREA_THRESHOLD = 0        # 面积过滤阈值
VISUALIZE_TOP_K = 3           # 可视化样本数

# ================= 2. 核心计算工具 (Core Utils) =================

def get_box_metrics(pts):
    """
    计算旋转框的多种尺度指标
    pts: [x1, y1, x2, y2, x3, y3, x4, y4]
    返回: (area_sqrt, long_side, short_side, center_y)
    """
    pts = np.array(pts).reshape(4, 2)
    
    # 1. 几何中心 Y
    cy = np.mean(pts[:, 1])
    
    # 2. 边长计算 (假设点序是顺时针或逆时针)
    # d01, d12, d23, d30
    dists = [np.linalg.norm(pts[i] - pts[(i+1)%4]) for i in range(4)]
    
    # 简单的边长估计：取平均
    side_a = (dists[0] + dists[2]) / 2
    side_b = (dists[1] + dists[3]) / 2
    
    long_side = max(side_a, side_b)
    short_side = min(side_a, side_b)
    
    # 3. 面积 (Shoelace formula)
    x = pts[:, 0]
    y = pts[:, 1]
    area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    area_sqrt = np.sqrt(area)
    
    return area_sqrt, long_side, short_side, cy, area

def parse_line(line, fmt, class_names):
    parts = line.strip().split()
    obj = None
    
    try:
        if fmt == 'dota':
            if len(parts) >= 10:
                pts = [float(p) for p in parts[:8]]
                cls = parts[8]
                diff = int(parts[9])
                obj = {'points': pts, 'class': cls, 'difficulty': diff}
                
        elif fmt == 'dronevehicle':
            if len(parts) >= 9:
                pts = [float(p) for p in parts[:8]]
                cls_id = int(parts[8])
                if 0 <= cls_id < len(class_names):
                    obj = {'points': pts, 'class': class_names[cls_id], 'difficulty': 0}
                    
        elif fmt == 'codrone':
            if len(parts) >= 10:
                pts = [float(p) for p in parts[:8]]
                cls = parts[8]
                if cls != 'ignored':
                    try: diff = int(parts[9])
                    except: diff = 0
                    obj = {'points': pts, 'class': cls, 'difficulty': diff}
    except:
        pass
        
    return obj

# ================= 3. 分析主逻辑 (Analysis Logic) =================

def analyze_dataset(dataset_name, config):
    print(f"Analyzing {dataset_name}...")
    
    metrics_types = ['area', 'long', 'short']
    
    # 数据存储结构: 每一层增加 metrics 维度
    stats_storage = {}
    for c in config['classes']:
        stats_storage[c] = {}
        for m in metrics_types:
            stats_storage[c][m] = {
                'cv': [],
                'corr': [],
                'slope': []
            }
        stats_storage[c]['count'] = 0
        
    # 用于可视化的典型样本缓存 (仍然基于 area 的相关性来筛选典型样本，因为它最鲁棒)
    vis_cache = []
    
    file_list = []
    for subset in config['subsets']:
        path_pattern = os.path.join(config['root'], subset, 'annfiles', f"*.{config['ext']}")
        file_list.extend(glob.glob(path_pattern))
        
    for ann_file in tqdm(file_list, leave=False):
        with open(ann_file, 'r') as f:
            lines = f.readlines()
            
        img_objs = {c: [] for c in config['classes']}
        
        for line in lines:
            obj = parse_line(line, config['format'], config['classes'])
            if not obj: continue
            if obj['difficulty'] != 0: continue 
            if obj['class'] not in config['classes']: continue
            
            s_area, s_long, s_short, cy, raw_area = get_box_metrics(obj['points'])
            
            if raw_area < MIN_AREA_THRESHOLD: continue
            
            img_objs[obj['class']].append({
                's_area': s_area,
                's_long': s_long,
                's_short': s_short,
                'cy': cy
            })
            
        # 统计当前图片
        for cls_name, items in img_objs.items():
            if len(items) < MIN_INSTANCES_PER_CLASS: continue
            
            cys = np.array([x['cy'] for x in items])
            
            # 分别计算三种指标的 CV 和 Correlation
            for m in metrics_types:
                sizes = np.array([x[f's_{m}'] for x in items])
                
                # 1. CV
                mean_s = np.mean(sizes)
                std_s = np.std(sizes)
                cv = std_s / mean_s if mean_s > 0 else 0
                
                # 2. Correlation
                if np.std(cys) > 1e-4 and np.std(sizes) > 1e-4:
                    corr, _ = stats.pearsonr(cys, sizes)
                    slope, _, _, _, _ = stats.linregress(cys, sizes)
                else:
                    corr, slope = 0, 0
                
                stats_storage[cls_name][m]['cv'].append(cv)
                stats_storage[cls_name][m]['corr'].append(corr)
                stats_storage[cls_name][m]['slope'].append(slope)
            
            stats_storage[cls_name]['count'] += 1
            
            # 缓存典型样本 (用 area 的相关性作为代表)
            area_corr = stats_storage[cls_name]['area']['corr'][-1]
            if area_corr > 0.6 and len(items) > 10: 
                vis_cache.append({
                    'score': area_corr,
                    'file': os.path.basename(ann_file),
                    'y': cys,
                    's': np.array([x['s_area'] for x in items]),
                    'cls': cls_name
                })
                
    vis_cache.sort(key=lambda x: x['score'], reverse=True)
    return stats_storage, vis_cache[:VISUALIZE_TOP_K]

# ================= 4. 报告与绘图 (Report & Plot) =================

def generate_report(all_results):
    # 1. 基础分析报告 (基于 S_area)
    print("\n" + "="*95)
    print(f"{'DATASET':<12} | {'CLASS':<16} | {'TYPE':<9} | {'CV(Area)':<9} | {'Corr(Area)':<10} | {'PosRatio':<8}")
    print("="*95)
    
    for d_name, (d_stats, _) in all_results.items():
        rigid_list = DATASETS_CONFIG[d_name].get('rigid_classes', [])
        sorted_classes = sorted(d_stats.keys())
        
        total_cv, total_corr, total_cnt = 0, 0, 0
        
        for cls_name in sorted_classes:
            metrics = d_stats[cls_name]['area'] # 默认展示 Area
            n = len(metrics['cv'])
            if n == 0: continue
            
            m_cv = np.mean(metrics['cv'])
            valid_corrs = [c for c in metrics['corr'] if not np.isnan(c)]
            if not valid_corrs: 
                m_corr, pos_ratio = 0, 0
            else:
                m_corr = np.mean(valid_corrs)
                pos_ratio = sum(1 for c in valid_corrs if c > 0) / len(valid_corrs)
            
            ctype = "Rigid" if cls_name in rigid_list else "Non-Rigid"
            print(f"{d_name:<12} | {cls_name:<16} | {ctype:<9} | {m_cv:.3f}     | {m_corr:.3f}      | {pos_ratio:.2f}")
            
            total_cv += np.sum(metrics['cv'])
            total_corr += np.sum(valid_corrs)
            total_cnt += len(valid_corrs)
            
        if total_cnt > 0:
            print("-" * 95)
            print(f"{d_name:<12} | {'[OVERALL]':<16} | {'ALL':<9} | {total_cv/total_cnt:.3f}     | {total_corr/total_cnt:.3f}      | -")
            print("=" * 95)

    # 2. 指标对比分析 (Area vs Long vs Short)
    print("\n" + "#"*40 + " METRIC COMPARISON ANALYSIS " + "#"*40)
    print("which metric has the strongest correlation with Y-coordinate?\n")
    
    print(f"{'DATASET':<12} | {'Metric':<8} | {'Avg Correlation (Higher is Better)':<35} | {'Best?'}")
    print("-" * 85)
    
    for d_name, (d_stats, _) in all_results.items():
        # 汇总该数据集下所有样本的指标
        agg_corr = {'area': [], 'long': [], 'short': []}
        
        for cls_name in d_stats:
            for m in ['area', 'long', 'short']:
                valid_c = [c for c in d_stats[cls_name][m]['corr'] if not np.isnan(c)]
                agg_corr[m].extend(valid_c)
        
        if not agg_corr['area']: continue
        
        means = {m: np.mean(vals) for m, vals in agg_corr.items()}
        best_metric = max(means, key=means.get)
        
        for m in ['area', 'long', 'short']:
            is_best = "YES <--" if m == best_metric else ""
            print(f"{d_name:<12} | {m:<8} | {means[m]:.4f}                              | {is_best}")
        print("-" * 85)

def plot_visualizations(all_results):
    # 1. 散点回归图
    n_datasets = len(all_results)
    if n_datasets > 0:
        fig, axes = plt.subplots(1, n_datasets, figsize=(6*n_datasets, 5))
        if n_datasets == 1: axes = [axes]
        
        for ax, (d_name, (_, top_samples)) in zip(axes, all_results.items()):
            if not top_samples:
                ax.text(0.5, 0.5, 'No samples', ha='center')
                ax.set_title(d_name)
                continue
                
            sample = top_samples[0] 
            sns.regplot(x=sample['y'], y=sample['s'], ax=ax, scatter_kws={'s': 50, 'alpha':0.6}, line_kws={'color':'red'})
            ax.set_title(f"{d_name}\nFile: {sample['file']}\nClass: {sample['cls']} (r={sample['score']:.2f})")
            ax.set_xlabel("Y Coordinate")
            ax.set_ylabel("Scale (sqrt(Area))")
            ax.grid(True, linestyle='--', alpha=0.5)
            
        plt.tight_layout()
        plt.savefig('val/perspective_regression_examples.png')
        print("Saved perspective_regression_examples.png")
    
    # 2. CV 对比箱线图 (Area 指标)
    plt.figure(figsize=(10, 6))
    plot_data = []
    labels = []
    for d_name, (d_stats, _) in all_results.items():
        vals = []
        for c in d_stats:
            vals.extend(d_stats[c]['area']['cv'])
        if vals:
            plot_data.append(vals)
            labels.append(d_name)
            
    plt.boxplot(plot_data, labels=labels, showfliers=False, patch_artist=True)
    plt.title("Intra-Image Size Consistency (Using S_area)")
    plt.ylabel("CV")
    plt.grid(axis='y', alpha=0.5)
    plt.savefig('val/cv_comparison_boxplot.png')
    print("Saved cv_comparison_boxplot.png")

if __name__ == "__main__":
    all_res = {}
    for name, conf in DATASETS_CONFIG.items():
        if os.path.exists(conf['root']):
            all_res[name] = analyze_dataset(name, conf)
        else:
            print(f"Skipping {name}: path not found")
            
    if all_res:
        generate_report(all_res)
        plot_visualizations(all_res)
    else:
        print("No datasets analyzed.")