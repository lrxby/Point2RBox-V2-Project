# import os
# import glob
# import math
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from scipy.stats import pearsonr
# from tqdm import tqdm

# # ================= 配置区域（完全复用你的配置）=================
# # 数据集标注路径
# ANNOTATION_PATH = '/mnt/data/xiekaikai/split_ss_dota/trainval/annfiles'
# # 结果保存文件夹
# OUT_DIR = 'work_dirs/analysis_results_split_dota5'
# # 刚性物体类别
# RIGID_CLASSES = [
#     'plane', 'baseball-diamond', 'bridge', 'ground-track-field',
#     'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
#     'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout',
#     'harbor', 'swimming-pool', 'helicopter'
# ]
# # 统计阈值（复用你的阈值，移除了CV>0.05的阈值）
# CV_THRESHOLD = 0.15    # 尺寸一致性阈值
# CORR_THRESHOLD = 0.6   # 透视线性阈值（仅可视化用，无统计计算）
# MIN_SAMPLES = 5        # 单图同类最小样本数
# SCALE_MIN = 0          # 最小尺度（过滤小目标）
# CV_MAX = 1.0           # 最大CV（过滤异常值）
# # ===============================================================

# def parse_dota_line(line):
#     """解析DOTA标注行（复用你的逻辑，优化异常处理）"""
#     parts = line.strip().split()
#     if len(parts) < 10:
#         return None, None
    
#     category = parts[8]
#     if category not in RIGID_CLASSES:
#         return None, None
        
#     try:
#         # 提取多边形坐标
#         poly = list(map(float, parts[:8]))
#         poly = np.array(poly, dtype=np.float32).reshape(4, 2)
#         # 计算中心Y坐标
#         cy = np.mean(poly[:, 1])
#         # 鞋带公式计算面积 -> 开根号得到scale（你的尺度定义）
#         x, y = poly[:, 0], poly[:, 1]
#         area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
#         scale = np.sqrt(area)
        
#         # 过滤过小的scale（你的逻辑）
#         if scale < SCALE_MIN:
#             return None, None
        
#         return category, (cy, scale)
#     except (ValueError, IndexError):
#         return None, None

# def calculate_stats_per_image(img_data):
#     """单图片统计：计算每个类别的CV和相关系数（移除P值无实际使用）"""
#     img_stats = {}
#     for cls_name, points in img_data.items():
#         if len(points) < MIN_SAMPLES:
#             continue
        
#         points = np.array(points)
#         ys, scales = points[:, 0], points[:, 1]
        
#         # 1. 计算尺寸一致性：变异系数CV
#         mean_s = np.mean(scales)
#         std_s = np.std(scales)
#         cv = std_s / (mean_s + 1e-6)
        
#         # 过滤异常CV（你的逻辑）
#         if cv >= CV_MAX:
#             continue
        
#         # 2. 计算透视线性：皮尔逊相关系数（移除P值，因不再统计Sig Corr）
#         corr, _ = pearsonr(ys, scales)
#         if np.isnan(corr):
#             corr = 0.0
        
#         img_stats[cls_name] = {
#             'cv': cv,
#             'corr': corr,
#             'ys': ys,
#             'scales': scales
#         }
#     return img_stats

# def generate_text_report(stats, out_path):
#     """生成文本报告：移除Sig Corr，新增总计行+平均绝对值Corr"""
#     lines = []
#     lines.append("="*120)  # 调整分隔线长度适配新列
#     # 新增Avg |Corr|列
#     lines.append(f"{'Class':<18} | {'Images':<6} | {'Avg CV':<10} | {'Median CV':<10} | {'CV<0.15(%)':<10} | {'Avg Corr':<10} | {'Avg |Corr|':<10} | {'PosRatio(%)':<10}")
#     lines.append("-" * 120)
    
#     # 初始化总计数据容器
#     total_imgs = 0
#     all_cvs = []
#     all_corrs = []
    
#     # 分类别统计
#     for c in RIGID_CLASSES:
#         cvs = np.array(stats[c]['cvs'])
#         corrs = np.array(stats[c]['corrs'])
        
#         n_imgs = len(cvs)
#         total_imgs += n_imgs  # 累加总有效图片数
#         all_cvs.extend(cvs)   # 收集所有CV数据
#         all_corrs.extend(corrs)  # 收集所有Corr数据
        
#         if n_imgs == 0:
#             lines.append(f"{c:<18} | {'0':<6} | {'N/A':<10} | {'N/A':<10} | {'N/A':<10} | {'N/A':<10} | {'N/A':<10} | {'N/A':<10}")
#             continue
        
#         # 计算当前类别指标（新增Avg |Corr|）
#         avg_cv = np.mean(cvs)
#         median_cv = np.median(cvs)
#         consistent_ratio = np.sum(cvs < CV_THRESHOLD) / n_imgs * 100
#         avg_corr = np.mean(corrs)
#         avg_abs_corr = np.mean(np.abs(corrs))  # 新增：平均绝对值Corr
#         pos_ratio = np.sum(corrs > 0) / len(corrs) * 100
        
#         # 格式化输出
#         lines.append(f"{c:<18} | {n_imgs:<6} | {avg_cv:<10.3f} | {median_cv:<10.3f} | {consistent_ratio:<10.1f} | {avg_corr:<10.3f} | {avg_abs_corr:<10.3f} | {pos_ratio:<10.1f}")
    
#     # 计算总计指标（新增Avg |Corr|）
#     lines.append("-" * 120)
#     if total_imgs > 0 and len(all_cvs) > 0 and len(all_corrs) > 0:
#         total_avg_cv = np.mean(all_cvs)
#         total_median_cv = np.median(all_cvs)
#         total_consistent_ratio = np.sum(np.array(all_cvs) < CV_THRESHOLD) / len(all_cvs) * 100
#         total_avg_corr = np.mean(all_corrs)
#         total_avg_abs_corr = np.mean(np.abs(all_corrs))  # 总计：平均绝对值Corr
#         total_pos_ratio = np.sum(np.array(all_corrs) > 0) / len(all_corrs) * 100
        
#         # 输出总计行
#         lines.append(f"{'Total':<18} | {total_imgs:<6} | {total_avg_cv:<10.3f} | {total_median_cv:<10.3f} | {total_consistent_ratio:<10.1f} | {total_avg_corr:<10.3f} | {total_avg_abs_corr:<10.3f} | {total_pos_ratio:<10.1f}")
#     else:
#         lines.append(f"{'Total':<18} | {'0':<6} | {'N/A':<10} | {'N/A':<10} | {'N/A':<10} | {'N/A':<10} | {'N/A':<10} | {'N/A':<10}")
#     lines.append("="*120)
    
#     # 更新结论（补充Avg |Corr|相关判定）
#     lines.append("\n[结论判定参考]:")
#     lines.append("1. Median CV < 0.15 且 CV<0.15(%) > 80%  -> 强一致性 (适合用全图均值约束)")
#     lines.append("2. Avg Corr > 0.6 或 Avg |Corr| > 0.6 -> 强透视性（近大远小）(适合用线性回归约束)")
#     lines.append("3. PosRatio(%) > 70% -> 多数样本符合近大远小规律")
#     lines.append("4. 都不满足                            -> 建议用局部(邻域)一致性约束")
    
#     report_str = "\n".join(lines)
#     print(report_str)
    
#     with open(out_path, 'w', encoding='utf-8') as f:
#         f.write(report_str)
#     print(f"\n[提示] 文本报告已保存至: {out_path}")

# def generate_visualizations(stats, vis_samples, out_dir):
#     """生成可视化图表：移除Sig Corr相关标注，新增Avg |Corr|可视化"""
#     # 确保中文/特殊字符正常显示
#     plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
#     plt.rcParams['axes.unicode_minus'] = False
    
#     # 图1: 类别CV箱线图（新增Avg CV标注）
#     plt.figure(figsize=(14, 7))
#     cv_data, cv_labels = [], []
#     for c in RIGID_CLASSES:
#         vals = stats[c]['cvs']
#         if len(vals) == 0:
#             continue
#         cv_data.extend(vals)
#         cv_labels.extend([c]*len(vals))
    
#     if cv_data:
#         sns.boxplot(x=cv_labels, y=cv_data, showfliers=False, palette='Set2')
#         plt.axhline(CV_THRESHOLD, color='r', linestyle='--', label=f'Threshold ({CV_THRESHOLD})')
#         plt.title('Scale Consistency (CV) Distribution per Class (DOTA) - All Samples', fontsize=12)
#         plt.xlabel('Class', fontsize=10)
#         plt.ylabel('Coefficient of Variation (CV)', fontsize=10)
#         plt.xticks(rotation=45, ha='right')
#         plt.legend()
#         plt.tight_layout()
#         plt.savefig(os.path.join(out_dir, '1_cv_boxplot.png'), dpi=300)
#         plt.close()
    
#     # 图2-1 分类别相关系数子图（基于全部样本，标注Avg |Corr|）
#     valid_classes = [c for c in RIGID_CLASSES if len(stats[c]['cvs']) > 0]
#     n_classes = len(valid_classes)
#     if n_classes > 0:
#         n_rows = 5
#         n_cols = 3
#         plt.figure(figsize=(15, 20))
        
#         for idx, cls_name in enumerate(valid_classes):
#             # 直接使用全部相关系数
#             corr_data = np.array(stats[cls_name]['corrs'])
#             n_samples = len(corr_data)  # 全部样本数量
            
#             plt.subplot(n_rows, n_cols, idx+1)
#             if len(corr_data) > 0:
#                 sns.histplot(corr_data, bins=20, kde=True, color='steelblue')
#                 plt.axvline(0, color='k', linestyle='--', label='Corr=0')
#                 # 标注原始Corr均值和绝对值均值
#                 avg_corr = np.mean(corr_data)
#                 avg_abs_corr = np.mean(np.abs(corr_data))
#                 plt.axvline(avg_corr, color='orange', linestyle='-', label=f'Avg Corr={avg_corr:.2f}')
#                 plt.axvline(avg_abs_corr, color='green', linestyle=':', label=f'Avg |Corr|={avg_abs_corr:.2f}')
#                 plt.axvline(-avg_abs_corr, color='green', linestyle=':')
#             plt.title(f'Correlation Distribution: {cls_name} (n={n_samples})', fontsize=10)
#             plt.xlabel('Pearson Correlation (Signed)', fontsize=8)
#             plt.ylabel('Count', fontsize=8)
#             plt.tick_params(axis='both', labelsize=7)
#             if idx == 0:
#                 plt.legend(fontsize=7)
        
#         plt.suptitle('Correlation (Y-Coord vs Scale) Distribution per Class (DOTA) - All Samples', fontsize=14)
#         plt.tight_layout(rect=[0, 0, 1, 0.96])
#         plt.savefig(os.path.join(out_dir, '2_corr_dist_per_class.png'), dpi=300)
#         plt.close()

#     # 图2-2 全类别统一相关系数直方图（基于全部样本，标注Avg |Corr|）
#     plt.figure(figsize=(10, 6))
#     all_corr_data = []
#     # 收集所有类别的全部相关系数
#     for c in RIGID_CLASSES:
#         all_corr_data.extend(stats[c]['corrs'])
    
#     total_samples = len(all_corr_data)
#     if all_corr_data:
#         sns.histplot(all_corr_data, bins=30, kde=True, color='steelblue')
#         plt.axvline(0, color='k', linestyle='--', label='Corr=0')
#         # 标注整体原始Corr均值和绝对值均值
#         overall_avg_corr = np.mean(all_corr_data)
#         overall_avg_abs_corr = np.mean(np.abs(all_corr_data))
#         plt.axvline(overall_avg_corr, color='orange', linestyle='-', label=f'Overall Avg Corr={overall_avg_corr:.2f}')
#         plt.axvline(overall_avg_abs_corr, color='green', linestyle=':', label=f'Overall Avg |Corr|={overall_avg_abs_corr:.2f}')
#         plt.axvline(-overall_avg_abs_corr, color='green', linestyle=':')
#         plt.title(f'Correlation (Y-Coord vs Scale) Distribution (All Classes, DOTA) (Total n={total_samples})', fontsize=12)
#         plt.xlabel('Pearson Correlation Coefficient (Signed)', fontsize=10)
#         plt.ylabel('Count', fontsize=10)
#         plt.legend()
#         plt.tight_layout()
#         plt.savefig(os.path.join(out_dir, '2_corr_dist_all_classes.png'), dpi=300)
#     plt.close()
    
#     # 图3: 示例散点图（直观展示线性关系）
#     if vis_samples:
#         plt.figure(figsize=(12, 8))
#         n_cols = 4
#         n_rows = math.ceil(len(vis_samples) / n_cols)
#         for idx, sample in enumerate(vis_samples[:12]):
#             plt.subplot(n_rows, n_cols, idx+1)
#             plt.scatter(sample['ys'], sample['scales'], s=10, alpha=0.7)
#             # 标题补充绝对值Corr
#             abs_corr = np.abs(sample['corr'])
#             plt.title(f"{sample['file'][:8]} | {sample['class']}\nCV={sample['cv']:.3f} | Corr={sample['corr']:.3f} | |Corr|={abs_corr:.3f}", fontsize=8)
#             plt.xlabel('Y Coord', fontsize=7)
#             plt.ylabel('Scale', fontsize=7)
#             plt.tick_params(axis='both', labelsize=6)
#         plt.tight_layout()
#         plt.savefig(os.path.join(out_dir, '3_sample_scatter.png'), dpi=300)
#         plt.close()
    
#     # 图4: 类别Avg |Corr|对比图（替换原Avg Corr，更具参考价值）
#     plt.figure(figsize=(12, 6))
#     cls_abs_corr = []
#     cls_names = []
#     for c in RIGID_CLASSES:
#         corr_vals = np.array(stats[c]['corrs'])
#         if len(corr_vals) == 0:
#             continue
#         avg_abs_corr = np.mean(np.abs(corr_vals))  # 平均绝对值Corr
#         cls_abs_corr.append(avg_abs_corr)
#         cls_names.append(c)
    
#     if cls_abs_corr:
#         sns.barplot(x=cls_names, y=cls_abs_corr, palette='Set1')
#         plt.axhline(CORR_THRESHOLD, color='r', linestyle='--', label=f'Threshold ({CORR_THRESHOLD})')
#         plt.title('Average |Pearson Correlation| per Class (DOTA) - All Samples', fontsize=12)
#         plt.xlabel('Class', fontsize=10)
#         plt.ylabel('Average |Pearson Correlation|', fontsize=10)
#         plt.xticks(rotation=45, ha='right')
#         plt.legend()
#         plt.tight_layout()
#         plt.savefig(os.path.join(out_dir, '4_cls_abs_corr_bar.png'), dpi=300)
#         plt.close()

# def generate_markdown_report(stats, out_dir):
#     """生成Markdown报告：移除Sig Corr，新增总计行+平均绝对值Corr"""
#     # 先计算总计数据
#     total_imgs = 0
#     all_cvs = []
#     all_corrs = []
#     for c in RIGID_CLASSES:
#         cvs = np.array(stats[c]['cvs'])
#         corrs = np.array(stats[c]['corrs'])
#         total_imgs += len(cvs)
#         all_cvs.extend(cvs)
#         all_corrs.extend(corrs)
    
#     # 计算总计指标（新增Avg |Corr|）
#     if total_imgs > 0 and len(all_cvs) > 0 and len(all_corrs) > 0:
#         total_avg_cv = np.mean(all_cvs)
#         total_median_cv = np.median(all_cvs)
#         total_consistent_ratio = np.sum(np.array(all_cvs) < CV_THRESHOLD) / len(all_cvs) * 100
#         total_avg_corr = np.mean(all_corrs)
#         total_avg_abs_corr = np.mean(np.abs(all_corrs))  # 总计：平均绝对值Corr
#         total_pos_ratio = np.sum(np.array(all_corrs) > 0) / len(all_corrs) * 100
#     else:
#         total_avg_cv = total_median_cv = total_consistent_ratio = total_avg_corr = total_avg_abs_corr = total_pos_ratio = "N/A"
    
#     report = f"""
# # 先验知识统计验证报告（DOTA-split_ss）
# ## 1. 数据集信息
# - 标注路径：{ANNOTATION_PATH}
# - 验证类别：{', '.join(RIGID_CLASSES)}
# - 统计阈值：
#   - 单图同类最小样本数：{MIN_SAMPLES}
#   - 尺寸一致性阈值（CV）：{CV_THRESHOLD}
#   - 最小尺度：{SCALE_MIN}，最大CV：{CV_MAX}
#   - 统计范围：所有有效样本（移除CV>0.05过滤限制）

# ## 2. 核心统计结果
# | 类别 | 有效图片数 | Avg CV | Median CV | CV<0.15比例(%) | Avg Corr（带符号） | Avg |Corr| | PosRatio(%) |
# |------|------------|--------|-----------|---------------|-------------------|-----------|-------------|
# """
#     # 分类别输出（新增Avg |Corr|列）
#     for c in RIGID_CLASSES:
#         cvs = np.array(stats[c]['cvs'])
#         corrs = np.array(stats[c]['corrs'])
        
#         n_imgs = len(cvs)
#         if n_imgs == 0:
#             report += f"| {c} | 0 | N/A | N/A | N/A | N/A | N/A | N/A |\n"
#             continue
        
#         avg_cv = np.mean(cvs)
#         median_cv = np.median(cvs)
#         consistent_ratio = np.sum(cvs < CV_THRESHOLD) / n_imgs * 100
#         avg_corr = np.mean(corrs)
#         avg_abs_corr = np.mean(np.abs(corrs))  # 类别：平均绝对值Corr
#         pos_ratio = np.sum(corrs > 0) / len(corrs) * 100
        
#         report += f"| {c} | {n_imgs} | {avg_cv:.3f} | {median_cv:.3f} | {consistent_ratio:.1f} | {avg_corr:.3f} | {avg_abs_corr:.3f} | {pos_ratio:.1f} |\n"
    
#     # 输出总计行（新增Avg |Corr|）
#     report += f"| **Total** | {total_imgs} | {total_avg_cv if total_avg_cv == 'N/A' else f'{total_avg_cv:.3f}'} | {total_median_cv if total_median_cv == 'N/A' else f'{total_median_cv:.3f}'} | {total_consistent_ratio if total_consistent_ratio == 'N/A' else f'{total_consistent_ratio:.1f}'} | {total_avg_corr if total_avg_corr == 'N/A' else f'{total_avg_corr:.3f}'} | {total_avg_abs_corr if total_avg_abs_corr == 'N/A' else f'{total_avg_abs_corr:.3f}'} | {total_pos_ratio if total_pos_ratio == 'N/A' else f'{total_pos_ratio:.1f}'} |\n"

#     report += """
# ## 3. 结论与建议
# ### 3.1 尺寸一致性先验
# - 强一致性类别：Avg CV < 0.15 且 Median CV < 0.15 且 CV<0.15比例>80% → 适合用全图均值约束Loss；
# - 弱一致性类别：建议用局部邻域一致性约束，或放宽CV阈值；

# ### 3.2 透视线性先验（近大远小规律）
# - 强符合规律：Avg |Corr| > 0.6 且 PosRatio(%) > 70% → 适合用线性回归约束Loss；
# - 中等符合规律：0.3 < Avg |Corr| ≤ 0.6 且 PosRatio(%) > 50% → 可尝试弱线性约束；
# - 弱符合规律：Avg |Corr| ≤ 0.3 → 建议仅保留尺寸约束；

# ### 3.3 整体建议
# 1. 优先对强一致性+强透视性（Avg |Corr|>0.6）的类别加入双约束Loss；
# 2. 对仅满足其一的类别加入单约束Loss；
# 3. 对均不满足的类别，建议先验证局部邻域规律，再设计Loss。
#     """
    
#     with open(os.path.join(out_dir, 'analysis_report.md'), 'w', encoding='utf-8') as f:
#         f.write(report)
#     print(f"[提示] Markdown报告已保存至: {os.path.join(out_dir, 'analysis_report.md')}")

# def verify_prior():
#     # 1. 检查路径
#     if not os.path.exists(ANNOTATION_PATH):
#         print(f"错误: 标注路径不存在 -> {ANNOTATION_PATH}")
#         return
    
#     # 2. 加载标注文件
#     txt_files = glob.glob(os.path.join(ANNOTATION_PATH, '*.txt'))
#     if len(txt_files) == 0:
#         print(f"错误: 未找到标注文件 -> {ANNOTATION_PATH}")
#         return
#     print(f"-> 发现 {len(txt_files)} 个标注文件，正在解析...")
    
#     # 3. 初始化统计容器（移除p_values，因不再使用）
#     os.makedirs(OUT_DIR, exist_ok=True)
#     stats = {c: {'cvs': [], 'corrs': []} for c in RIGID_CLASSES}
#     vis_samples = []  # 可视化样本
    
#     # 4. 逐文件解析+统计
#     for txt_file in tqdm(txt_files, desc='解析标注并统计'):
#         filename = os.path.basename(txt_file)
#         img_data = {c: [] for c in RIGID_CLASSES}
        
#         # 解析单文件
#         with open(txt_file, 'r', encoding='utf-8') as f:
#             lines = f.readlines()
        
#         for line in lines:
#             cat, data = parse_dota_line(line)
#             if cat and data:
#                 img_data[cat].append(data)
        
#         # 单图片统计
#         img_stats = calculate_stats_per_image(img_data)
        
#         # 汇总统计结果
#         for cls_name, cls_stats in img_stats.items():
#             stats[cls_name]['cvs'].append(cls_stats['cv'])
#             stats[cls_name]['corrs'].append(cls_stats['corr'])
            
#             # 收集可视化样本
#             if len(vis_samples) < 24 and (cls_stats['cv'] < 0.1 or abs(cls_stats['corr']) > 0.7):
#                 vis_samples.append({
#                     'file': filename,
#                     'class': cls_name,
#                     'ys': cls_stats['ys'],
#                     'scales': cls_stats['scales'],
#                     'cv': cls_stats['cv'],
#                     'corr': cls_stats['corr']
#                 })
    
#     # 5. 生成报告
#     print("\n" + "="*120)
#     print("核心统计结果（基于全部有效样本）：")
#     print("="*120)
#     text_report_path = os.path.join(OUT_DIR, 'analysis_report.txt')
#     generate_text_report(stats, text_report_path)
    
#     # 6. 生成可视化
#     print("\n-> 正在生成可视化图表...")
#     generate_visualizations(stats, vis_samples, OUT_DIR)
    
#     # 7. 生成Markdown报告
#     generate_markdown_report(stats, OUT_DIR)
    
#     print("\n-> 所有验证完成！结果保存至：", OUT_DIR)
#     print("-> 关键文件：")
#     print(f"   - 文本报告：{text_report_path}")
#     print(f"   - Markdown报告：{os.path.join(OUT_DIR, 'analysis_report.md')}")
#     print(f"   - 分类别相关系数图：{os.path.join(OUT_DIR, '2_corr_dist_per_class.png')}")
#     print(f"   - 全类别相关系数图：{os.path.join(OUT_DIR, '2_corr_dist_all_classes.png')}")
#     print(f"   - 类别Avg |Corr|对比图：{os.path.join(OUT_DIR, '4_cls_abs_corr_bar.png')}")

# if __name__ == '__main__':
#     verify_prior()

import os
import glob
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from tqdm import tqdm

# ================= 配置区域（完全复用你的配置）=================
# 数据集标注路径
ANNOTATION_PATH = '/mnt/data/xiekaikai/split_ss_dota/trainval/annfiles'
# 结果保存文件夹
OUT_DIR = 'work_dirs/analysis_results_split_dota2'
# 刚性物体类别
RIGID_CLASSES = [
    'plane', 'baseball-diamond', 'bridge', 'ground-track-field',
    'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
    'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout',
    'harbor', 'swimming-pool', 'helicopter'
]
# 统计阈值（复用你的阈值，移除了CV>0.05的阈值）
CV_THRESHOLD = 0.15    # 尺寸一致性阈值
CORR_THRESHOLD = 0.6   # 透视线性阈值（仅可视化用，无统计计算）
MIN_SAMPLES = 2        # 单图同类最小样本数
SCALE_MIN = 0          # 最小尺度（过滤小目标）
CV_MAX = 1.0           # 最大CV（过滤异常值）
# ===============================================================

def parse_dota_line(line):
    """解析DOTA标注行（复用你的逻辑，优化异常处理）"""
    parts = line.strip().split()
    if len(parts) < 10:
        return None, None
    
    category = parts[8]
    if category not in RIGID_CLASSES:
        return None, None
        
    try:
        # 提取多边形坐标
        poly = list(map(float, parts[:8]))
        poly = np.array(poly, dtype=np.float32).reshape(4, 2)
        # 计算中心Y坐标
        cy = np.mean(poly[:, 1])
        # 鞋带公式计算面积 -> 开根号得到scale（你的尺度定义）
        x, y = poly[:, 0], poly[:, 1]
        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        scale = np.sqrt(area)
        
        # 过滤过小的scale（你的逻辑）
        if scale < SCALE_MIN:
            return None, None
        
        return category, (cy, scale)
    except (ValueError, IndexError):
        return None, None

def calculate_stats_per_image(img_data):
    """单图片统计：计算每个类别的CV和相关系数（移除P值无实际使用）"""
    img_stats = {}
    for cls_name, points in img_data.items():
        if len(points) < MIN_SAMPLES:
            continue
        
        points = np.array(points)
        ys, scales = points[:, 0], points[:, 1]
        
        # 1. 计算尺寸一致性：变异系数CV
        mean_s = np.mean(scales)
        std_s = np.std(scales)
        cv = std_s / (mean_s + 1e-6)
        
        # 过滤异常CV（你的逻辑）
        if cv >= CV_MAX:
            continue
        
        # 2. 计算透视线性：皮尔逊相关系数（移除P值，因不再统计Sig Corr）
        corr, _ = pearsonr(ys, scales)
        if np.isnan(corr):
            corr = 0.0
        
        img_stats[cls_name] = {
            'cv': cv,
            'corr': corr,
            'ys': ys,
            'scales': scales
        }
    return img_stats

def generate_text_report(stats, out_path):
    """生成文本报告：保留原有统计，新增正负相关数量+正负相关均值"""
    lines = []
    lines.append("="*180)  # 调整分隔线长度适配新列
    # 新增Pos_Count/Neg_Count/Avg_Corr_Pos/Avg_Corr_Neg列
    lines.append(
        f"{'Class':<18} | {'Images':<6} | {'Avg CV':<10} | {'Median CV':<10} | {'CV<0.15(%)':<10} | "
        f"{'Avg Corr':<10} | {'Avg |Corr|':<10} | {'PosRatio(%)':<10} | {'Pos_Count':<10} | {'Neg_Count':<10} | "
        f"{'Avg_Corr_Pos':<10} | {'Avg_Corr_Neg':<10}"
    )
    lines.append("-" * 180)
    
    # 初始化总计数据容器
    total_imgs = 0
    all_cvs = []
    all_corrs = []
    
    # 分类别统计
    for c in RIGID_CLASSES:
        cvs = np.array(stats[c]['cvs'])
        corrs = np.array(stats[c]['corrs'])
        
        n_imgs = len(cvs)
        total_imgs += n_imgs  # 累加总有效图片数
        all_cvs.extend(cvs)   # 收集所有CV数据
        all_corrs.extend(corrs)  # 收集所有Corr数据
        
        if n_imgs == 0:
            lines.append(
                f"{c:<18} | {'0':<6} | {'N/A':<10} | {'N/A':<10} | {'N/A':<10} | {'N/A':<10} | {'N/A':<10} | "
                f"{'N/A':<10} | {'N/A':<10} | {'N/A':<10} | {'N/A':<10} | {'N/A':<10}"
            )
            continue
        
        # 原有统计指标
        avg_cv = np.mean(cvs)
        median_cv = np.median(cvs)
        consistent_ratio = np.sum(cvs < CV_THRESHOLD) / n_imgs * 100
        avg_corr = np.mean(corrs)
        avg_abs_corr = np.mean(np.abs(corrs))
        pos_ratio = np.sum(corrs > 0) / len(corrs) * 100
        
        # 新增统计指标：正负相关数量+正负相关均值
        pos_corrs = corrs[corrs > 0]  # 相关系数>0的样本
        neg_corrs = corrs[corrs < 0]  # 相关系数<0的样本
        pos_count = len(pos_corrs)
        neg_count = len(neg_corrs)
        avg_corr_pos = np.mean(pos_corrs) if pos_count > 0 else 0.0
        avg_corr_neg = np.mean(neg_corrs) if neg_count > 0 else 0.0
        
        # 格式化输出
        lines.append(
            f"{c:<18} | {n_imgs:<6} | {avg_cv:<10.3f} | {median_cv:<10.3f} | {consistent_ratio:<10.1f} | "
            f"{avg_corr:<10.3f} | {avg_abs_corr:<10.3f} | {pos_ratio:<10.1f} | {pos_count:<10} | {neg_count:<10} | "
            f"{avg_corr_pos:<10.3f} | {avg_corr_neg:<10.3f}"
        )
    
    # 计算总计指标
    lines.append("-" * 180)
    if total_imgs > 0 and len(all_cvs) > 0 and len(all_corrs) > 0:
        # 原有总计指标
        total_avg_cv = np.mean(all_cvs)
        total_median_cv = np.median(all_cvs)
        total_consistent_ratio = np.sum(np.array(all_cvs) < CV_THRESHOLD) / len(all_cvs) * 100
        total_avg_corr = np.mean(all_corrs)
        total_avg_abs_corr = np.mean(np.abs(all_corrs))
        total_pos_ratio = np.sum(np.array(all_corrs) > 0) / len(all_corrs) * 100
        
        # 新增总计指标
        total_pos_corrs = np.array(all_corrs)[np.array(all_corrs) > 0]
        total_neg_corrs = np.array(all_corrs)[np.array(all_corrs) < 0]
        total_pos_count = len(total_pos_corrs)
        total_neg_count = len(total_neg_corrs)
        total_avg_corr_pos = np.mean(total_pos_corrs) if total_pos_count > 0 else 0.0
        total_avg_corr_neg = np.mean(total_neg_corrs) if total_neg_count > 0 else 0.0
        
        # 输出总计行
        lines.append(
            f"{'Total':<18} | {total_imgs:<6} | {total_avg_cv:<10.3f} | {total_median_cv:<10.3f} | {total_consistent_ratio:<10.1f} | "
            f"{total_avg_corr:<10.3f} | {total_avg_abs_corr:<10.3f} | {total_pos_ratio:<10.1f} | {total_pos_count:<10} | {total_neg_count:<10} | "
            f"{total_avg_corr_pos:<10.3f} | {total_avg_corr_neg:<10.3f}"
        )
    else:
        lines.append(
            f"{'Total':<18} | {'0':<6} | {'N/A':<10} | {'N/A':<10} | {'N/A':<10} | {'N/A':<10} | {'N/A':<10} | "
            f"{'N/A':<10} | {'N/A':<10} | {'N/A':<10} | {'N/A':<10} | {'N/A':<10}"
        )
    lines.append("="*180)
    
    # 结论部分保留原有逻辑
    lines.append("\n[结论判定参考]:")
    lines.append("1. Median CV < 0.15 且 CV<0.15(%) > 80%  -> 强一致性 (适合用全图均值约束)")
    lines.append("2. Avg Corr > 0.6 或 Avg |Corr| > 0.6 -> 强透视性（近大远小）(适合用线性回归约束)")
    lines.append("3. PosRatio(%) > 70% -> 多数样本符合近大远小规律")
    lines.append("4. 都不满足                            -> 建议用局部(邻域)一致性约束")
    
    report_str = "\n".join(lines)
    print(report_str)
    
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(report_str)
    print(f"\n[提示] 文本报告已保存至: {out_path}")

def generate_visualizations(stats, vis_samples, out_dir):
    """生成可视化图表：保留原有所有可视化逻辑"""
    # 确保中文/特殊字符正常显示
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 图1: 类别CV箱线图（新增Avg CV标注）
    plt.figure(figsize=(14, 7))
    cv_data, cv_labels = [], []
    for c in RIGID_CLASSES:
        vals = stats[c]['cvs']
        if len(vals) == 0:
            continue
        cv_data.extend(vals)
        cv_labels.extend([c]*len(vals))
    
    if cv_data:
        sns.boxplot(x=cv_labels, y=cv_data, showfliers=False, palette='Set2')
        plt.axhline(CV_THRESHOLD, color='r', linestyle='--', label=f'Threshold ({CV_THRESHOLD})')
        plt.title('Scale Consistency (CV) Distribution per Class (DOTA) - All Samples', fontsize=12)
        plt.xlabel('Class', fontsize=10)
        plt.ylabel('Coefficient of Variation (CV)', fontsize=10)
        plt.xticks(rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, '1_cv_boxplot.png'), dpi=300)
        plt.close()
    
    # 图2-1 分类别相关系数子图（基于全部样本，标注Avg |Corr|）
    valid_classes = [c for c in RIGID_CLASSES if len(stats[c]['cvs']) > 0]
    n_classes = len(valid_classes)
    if n_classes > 0:
        n_rows = 5
        n_cols = 3
        plt.figure(figsize=(15, 20))
        
        for idx, cls_name in enumerate(valid_classes):
            # 直接使用全部相关系数
            corr_data = np.array(stats[cls_name]['corrs'])
            n_samples = len(corr_data)  # 全部样本数量
            
            plt.subplot(n_rows, n_cols, idx+1)
            if len(corr_data) > 0:
                sns.histplot(corr_data, bins=20, kde=True, color='steelblue')
                plt.axvline(0, color='k', linestyle='--', label='Corr=0')
                # 标注原始Corr均值和绝对值均值
                avg_corr = np.mean(corr_data)
                avg_abs_corr = np.mean(np.abs(corr_data))
                plt.axvline(avg_corr, color='orange', linestyle='-', label=f'Avg Corr={avg_corr:.2f}')
                plt.axvline(avg_abs_corr, color='green', linestyle=':', label=f'Avg |Corr|={avg_abs_corr:.2f}')
                plt.axvline(-avg_abs_corr, color='green', linestyle=':')
            plt.title(f'Correlation Distribution: {cls_name} (n={n_samples})', fontsize=10)
            plt.xlabel('Pearson Correlation (Signed)', fontsize=8)
            plt.ylabel('Count', fontsize=8)
            plt.tick_params(axis='both', labelsize=7)
            if idx == 0:
                plt.legend(fontsize=7)
        
        plt.suptitle('Correlation (Y-Coord vs Scale) Distribution per Class (DOTA) - All Samples', fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(os.path.join(out_dir, '2_corr_dist_per_class.png'), dpi=300)
        plt.close()

    # 图2-2 全类别统一相关系数直方图（基于全部样本，标注Avg |Corr|）
    plt.figure(figsize=(10, 6))
    all_corr_data = []
    # 收集所有类别的全部相关系数
    for c in RIGID_CLASSES:
        all_corr_data.extend(stats[c]['corrs'])
    
    total_samples = len(all_corr_data)
    if all_corr_data:
        sns.histplot(all_corr_data, bins=30, kde=True, color='steelblue')
        plt.axvline(0, color='k', linestyle='--', label='Corr=0')
        # 标注整体原始Corr均值和绝对值均值
        overall_avg_corr = np.mean(all_corr_data)
        overall_avg_abs_corr = np.mean(np.abs(all_corr_data))
        plt.axvline(overall_avg_corr, color='orange', linestyle='-', label=f'Overall Avg Corr={overall_avg_corr:.2f}')
        plt.axvline(overall_avg_abs_corr, color='green', linestyle=':', label=f'Overall Avg |Corr|={overall_avg_abs_corr:.2f}')
        plt.axvline(-overall_avg_abs_corr, color='green', linestyle=':')
        plt.title(f'Correlation (Y-Coord vs Scale) Distribution (All Classes, DOTA) (Total n={total_samples})', fontsize=12)
        plt.xlabel('Pearson Correlation Coefficient (Signed)', fontsize=10)
        plt.ylabel('Count', fontsize=10)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, '2_corr_dist_all_classes.png'), dpi=300)
    plt.close()
    
    # 图3: 示例散点图（直观展示线性关系）
    if vis_samples:
        plt.figure(figsize=(12, 8))
        n_cols = 4
        n_rows = math.ceil(len(vis_samples) / n_cols)
        for idx, sample in enumerate(vis_samples[:12]):
            plt.subplot(n_rows, n_cols, idx+1)
            plt.scatter(sample['ys'], sample['scales'], s=10, alpha=0.7)
            # 标题补充绝对值Corr
            abs_corr = np.abs(sample['corr'])
            plt.title(f"{sample['file'][:8]} | {sample['class']}\nCV={sample['cv']:.3f} | Corr={sample['corr']:.3f} | |Corr|={abs_corr:.3f}", fontsize=8)
            plt.xlabel('Y Coord', fontsize=7)
            plt.ylabel('Scale', fontsize=7)
            plt.tick_params(axis='both', labelsize=6)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, '3_sample_scatter.png'), dpi=300)
        plt.close()
    
    # 图4: 类别Avg |Corr|对比图（替换原Avg Corr，更具参考价值）
    plt.figure(figsize=(12, 6))
    cls_abs_corr = []
    cls_names = []
    for c in RIGID_CLASSES:
        corr_vals = np.array(stats[c]['corrs'])
        if len(corr_vals) == 0:
            continue
        avg_abs_corr = np.mean(np.abs(corr_vals))  # 平均绝对值Corr
        cls_abs_corr.append(avg_abs_corr)
        cls_names.append(c)
    
    if cls_abs_corr:
        sns.barplot(x=cls_names, y=cls_abs_corr, palette='Set1')
        plt.axhline(CORR_THRESHOLD, color='r', linestyle='--', label=f'Threshold ({CORR_THRESHOLD})')
        plt.title('Average |Pearson Correlation| per Class (DOTA) - All Samples', fontsize=12)
        plt.xlabel('Class', fontsize=10)
        plt.ylabel('Average |Pearson Correlation|', fontsize=10)
        plt.xticks(rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, '4_cls_abs_corr_bar.png'), dpi=300)
        plt.close()

def generate_markdown_report(stats, out_dir):
    """生成Markdown报告：保留原有统计，新增正负相关数量+正负相关均值"""
    # 先计算总计数据
    total_imgs = 0
    all_cvs = []
    all_corrs = []
    for c in RIGID_CLASSES:
        cvs = np.array(stats[c]['cvs'])
        corrs = np.array(stats[c]['corrs'])
        total_imgs += len(cvs)
        all_cvs.extend(cvs)
        all_corrs.extend(corrs)
    
    # 计算总计指标
    if total_imgs > 0 and len(all_cvs) > 0 and len(all_corrs) > 0:
        # 原有总计指标
        total_avg_cv = np.mean(all_cvs)
        total_median_cv = np.median(all_cvs)
        total_consistent_ratio = np.sum(np.array(all_cvs) < CV_THRESHOLD) / len(all_cvs) * 100
        total_avg_corr = np.mean(all_corrs)
        total_avg_abs_corr = np.mean(np.abs(all_corrs))
        total_pos_ratio = np.sum(np.array(all_corrs) > 0) / len(all_corrs) * 100
        
        # 新增总计指标
        total_pos_corrs = np.array(all_corrs)[np.array(all_corrs) > 0]
        total_neg_corrs = np.array(all_corrs)[np.array(all_corrs) < 0]
        total_pos_count = len(total_pos_corrs)
        total_neg_count = len(total_neg_corrs)
        total_avg_corr_pos = np.mean(total_pos_corrs) if total_pos_count > 0 else 0.0
        total_avg_corr_neg = np.mean(total_neg_corrs) if total_neg_count > 0 else 0.0
    else:
        total_avg_cv = total_median_cv = total_consistent_ratio = total_avg_corr = total_avg_abs_corr = total_pos_ratio = "N/A"
        total_pos_count = total_neg_count = total_avg_corr_pos = total_avg_corr_neg = "N/A"
    
    report = f"""
# 先验知识统计验证报告（DOTA-split_ss）
## 1. 数据集信息
- 标注路径：{ANNOTATION_PATH}
- 验证类别：{', '.join(RIGID_CLASSES)}
- 统计阈值：
  - 单图同类最小样本数：{MIN_SAMPLES}
  - 尺寸一致性阈值（CV）：{CV_THRESHOLD}
  - 最小尺度：{SCALE_MIN}，最大CV：{CV_MAX}
  - 统计范围：所有有效样本（移除CV>0.05过滤限制）

## 2. 核心统计结果
| 类别 | 有效图片数 | Avg CV | Median CV | CV<0.15比例(%) | Avg Corr（带符号） | Avg |Corr| | PosRatio(%) | 正相关数量 | 负相关数量 | 正相关平均Corr | 负相关平均Corr |
|------|------------|--------|-----------|---------------|-------------------|-----------|-------------|------------|------------|----------------|----------------|
"""
    # 分类别输出（新增正负相关统计项）
    for c in RIGID_CLASSES:
        cvs = np.array(stats[c]['cvs'])
        corrs = np.array(stats[c]['corrs'])
        
        n_imgs = len(cvs)
        if n_imgs == 0:
            report += f"| {c} | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |\n"
            continue
        
        # 原有指标
        avg_cv = np.mean(cvs)
        median_cv = np.median(cvs)
        consistent_ratio = np.sum(cvs < CV_THRESHOLD) / n_imgs * 100
        avg_corr = np.mean(corrs)
        avg_abs_corr = np.mean(np.abs(corrs))
        pos_ratio = np.sum(corrs > 0) / len(corrs) * 100
        
        # 新增指标
        pos_corrs = corrs[corrs > 0]
        neg_corrs = corrs[corrs < 0]
        pos_count = len(pos_corrs)
        neg_count = len(neg_corrs)
        avg_corr_pos = np.mean(pos_corrs) if pos_count > 0 else 0.0
        avg_corr_neg = np.mean(neg_corrs) if neg_count > 0 else 0.0
        
        report += f"| {c} | {n_imgs} | {avg_cv:.3f} | {median_cv:.3f} | {consistent_ratio:.1f} | {avg_corr:.3f} | {avg_abs_corr:.3f} | {pos_ratio:.1f} | {pos_count} | {neg_count} | {avg_corr_pos:.3f} | {avg_corr_neg:.3f} |\n"
    
    # 输出总计行（新增正负相关统计项）
    report += f"| **Total** | {total_imgs} | {total_avg_cv if total_avg_cv == 'N/A' else f'{total_avg_cv:.3f}'} | {total_median_cv if total_median_cv == 'N/A' else f'{total_median_cv:.3f}'} | {total_consistent_ratio if total_consistent_ratio == 'N/A' else f'{total_consistent_ratio:.1f}'} | {total_avg_corr if total_avg_corr == 'N/A' else f'{total_avg_corr:.3f}'} | {total_avg_abs_corr if total_avg_abs_corr == 'N/A' else f'{total_avg_abs_corr:.3f}'} | {total_pos_ratio if total_pos_ratio == 'N/A' else f'{total_pos_ratio:.1f}'} | {total_pos_count} | {total_neg_count} | {total_avg_corr_pos if total_avg_corr_pos == 'N/A' else f'{total_avg_corr_pos:.3f}'} | {total_avg_corr_neg if total_avg_corr_neg == 'N/A' else f'{total_avg_corr_neg:.3f}'} |\n"

    report += """
## 3. 结论与建议
### 3.1 尺寸一致性先验
- 强一致性类别：Avg CV < 0.15 且 Median CV < 0.15 且 CV<0.15比例>80% → 适合用全图均值约束Loss；
- 弱一致性类别：建议用局部邻域一致性约束，或放宽CV阈值；

### 3.2 透视线性先验（近大远小规律）
- 强符合规律：Avg |Corr| > 0.6 且 PosRatio(%) > 70% → 适合用线性回归约束Loss；
- 中等符合规律：0.3 < Avg |Corr| ≤ 0.6 且 PosRatio(%) > 50% → 可尝试弱线性约束；
- 弱符合规律：Avg |Corr| ≤ 0.3 → 建议仅保留尺寸约束；

### 3.3 整体建议
1. 优先对强一致性+强透视性（Avg |Corr|>0.6）的类别加入双约束Loss；
2. 对仅满足其一的类别加入单约束Loss；
3. 对均不满足的类别，建议先验证局部邻域规律，再设计Loss。
    """
    
    with open(os.path.join(out_dir, 'analysis_report.md'), 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"[提示] Markdown报告已保存至: {os.path.join(out_dir, 'analysis_report.md')}")

def verify_prior():
    # 1. 检查路径
    if not os.path.exists(ANNOTATION_PATH):
        print(f"错误: 标注路径不存在 -> {ANNOTATION_PATH}")
        return
    
    # 2. 加载标注文件
    txt_files = glob.glob(os.path.join(ANNOTATION_PATH, '*.txt'))
    if len(txt_files) == 0:
        print(f"错误: 未找到标注文件 -> {ANNOTATION_PATH}")
        return
    print(f"-> 发现 {len(txt_files)} 个标注文件，正在解析...")
    
    # 3. 初始化统计容器（移除p_values，因不再使用）
    os.makedirs(OUT_DIR, exist_ok=True)
    stats = {c: {'cvs': [], 'corrs': []} for c in RIGID_CLASSES}
    vis_samples = []  # 可视化样本
    
    # 4. 逐文件解析+统计
    for txt_file in tqdm(txt_files, desc='解析标注并统计'):
        filename = os.path.basename(txt_file)
        img_data = {c: [] for c in RIGID_CLASSES}
        
        # 解析单文件
        with open(txt_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line in lines:
            cat, data = parse_dota_line(line)
            if cat and data:
                img_data[cat].append(data)
        
        # 单图片统计
        img_stats = calculate_stats_per_image(img_data)
        
        # 汇总统计结果
        for cls_name, cls_stats in img_stats.items():
            stats[cls_name]['cvs'].append(cls_stats['cv'])
            stats[cls_name]['corrs'].append(cls_stats['corr'])
            
            # 收集可视化样本
            if len(vis_samples) < 24 and (cls_stats['cv'] < 0.1 or abs(cls_stats['corr']) > 0.7):
                vis_samples.append({
                    'file': filename,
                    'class': cls_name,
                    'ys': cls_stats['ys'],
                    'scales': cls_stats['scales'],
                    'cv': cls_stats['cv'],
                    'corr': cls_stats['corr']
                })
    
    # 5. 生成报告
    print("\n" + "="*180)
    print("核心统计结果（基于全部有效样本）：")
    print("="*180)
    text_report_path = os.path.join(OUT_DIR, 'analysis_report.txt')
    generate_text_report(stats, text_report_path)
    
    # 6. 生成可视化
    print("\n-> 正在生成可视化图表...")
    generate_visualizations(stats, vis_samples, OUT_DIR)
    
    # 7. 生成Markdown报告
    generate_markdown_report(stats, OUT_DIR)
    
    print("\n-> 所有验证完成！结果保存至：", OUT_DIR)
    print("-> 关键文件：")
    print(f"   - 文本报告：{text_report_path}")
    print(f"   - Markdown报告：{os.path.join(OUT_DIR, 'analysis_report.md')}")
    print(f"   - 分类别相关系数图：{os.path.join(OUT_DIR, '2_corr_dist_per_class.png')}")
    print(f"   - 全类别相关系数图：{os.path.join(OUT_DIR, '2_corr_dist_all_classes.png')}")
    print(f"   - 类别Avg |Corr|对比图：{os.path.join(OUT_DIR, '4_cls_abs_corr_bar.png')}")

if __name__ == '__main__':
    verify_prior()