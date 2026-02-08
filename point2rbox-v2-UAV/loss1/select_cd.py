import os
import glob
import math
import numpy as np
import cv2
import torch
from tqdm import tqdm
import multiprocessing

# ================= 配置区域 =================
# CODrone 数据集根路径
DATASET_ROOT = '/mnt/data/xiekaikai/split_ss_codrone'
# 需要计算的子集名称
TARGET_SPLITS = ['trainval', 'test']

# 搜索的 K 值列表
K_LIST = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 
          1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 
          2.2, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

# CODrone 类别定义 (12类)
CLASSES = ('car', 'truck', 'bus', 'traffic-light',
           'traffic-sign', 'bridge', 'people', 'bicycle',
           'motor', 'tricycle', 'boat', 'ship')

# 采样文件数量 (-1 表示跑全量)
SAMPLE_NUM = -1 

# 强制使用 CPU 进行指标计算
DEVICE = 'cpu' 
# ===========================================

# 类别映射表
CLS_MAP = {c: i for i, c in enumerate(CLASSES)}

def parse_codrone_file(file_path):
    """
    解析 CODrone 格式 txt 文件
    """
    bboxes = []
    labels = []
    
    if not os.path.exists(file_path):
        return None, None

    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 9: continue
        
        try:
            poly = np.array([float(x) for x in parts[:8]]).reshape(4, 2).astype(np.float32)
            cls_name = parts[8]
        except ValueError:
            continue
            
        if cls_name not in CLS_MAP:
            continue
            
        rect = cv2.minAreaRect(poly)
        (cx, cy), (w, h), angle = rect
        
        if w < h:
            w, h = h, w
            angle += 90
        
        theta = np.deg2rad(angle)
        
        bboxes.append([cx, cy, w, h, theta])
        labels.append(CLS_MAP[cls_name])
        
    if not bboxes:
        return None, None
        
    # 【修复核心】返回 numpy 数组，而不是 torch.tensor
    # 这避免了多进程共享内存句柄耗尽的问题
    return np.array(bboxes, dtype=np.float32), np.array(labels, dtype=np.int64)

@torch.no_grad()
def compute_naoa_metrics(k_radius, dataset_samples):
    """
    使用指定的 k_radius 在数据集上运行 NAOALoss V4 计算流程
    """
    total_chaos_sum = 0.0
    total_valid_samples = 0
    total_isolated_count = 0
    total_objects = 0
    total_neighbor_sum = 0.0
    
    # 这里的 dataset_samples 里存的是 numpy array
    for bboxes_np, labels_np in tqdm(dataset_samples, desc=f"计算 K={k_radius:<4}", leave=False, dynamic_ncols=True):
        if bboxes_np is None or len(bboxes_np) < 2:
            continue
            
        # 【修复核心】在主进程计算前，将 numpy 转回 tensor
        bboxes = torch.from_numpy(bboxes_np).to(DEVICE)
        labels = torch.from_numpy(labels_np).to(DEVICE)
        
        N = bboxes.shape[0]
        total_objects += N
        
        # ================= Step 1: 几何解耦 =================
        centers = bboxes[:, :2]
        wh = bboxes[:, 2:4]
        scales = (wh[:, 0] * wh[:, 1]).sqrt().clamp(min=16.0, max=800.0)
        thetas = bboxes[:, 4]
        
        # ================= Step 2: 矢量化 (4-Theta) =================
        vecs = torch.stack([torch.cos(4 * thetas), torch.sin(4 * thetas)], dim=1)
        
        # ================= Step 3: 构建亲和矩阵 =================
        dist_sq = torch.cdist(centers, centers, p=2).pow(2)
        sigmas = scales * k_radius 
        sigma_mat = sigmas.view(N, 1)
        
        # Gaussian Kernel
        W_geo = torch.exp(-dist_sq / (2 * sigma_mat.pow(2)))
        
        # 3.3 逻辑掩码 (仅同类)
        mask_cls = (labels.view(N, 1) == labels.view(1, N)).float()
        
        # 组合权重 (包含自环)
        W = W_geo * mask_cls
        
        # --- 统计指标计算 ---
        W_no_diag = W.clone()
        W_no_diag.fill_diagonal_(0)
        neighbor_strength = W_no_diag.sum(dim=1)
        
        is_isolated = neighbor_strength < 0.1
        total_isolated_count += is_isolated.sum().item()
        total_neighbor_sum += neighbor_strength.sum().item()
        
        # ================= Step 4: 归一化 =================
        W_sum = W.sum(dim=1, keepdim=True)
        W_norm = W / W_sum
        
        # ================= Step 5: 能量计算 =================
        mean_vecs = torch.mm(W_norm, vecs)
        chaos_score = 1.0 - mean_vecs.norm(dim=1)
        
        valid_mask = ~is_isolated
        if valid_mask.sum() > 0:
            total_chaos_sum += chaos_score[valid_mask].sum().item()
            total_valid_samples += valid_mask.sum().item()
            
    # 汇总全局指标
    avg_chaos = total_chaos_sum / max(1, total_valid_samples)
    isolation_rate = total_isolated_count / max(1, total_objects)
    avg_neighbors = total_neighbor_sum / max(1, total_objects)
    
    return avg_chaos, isolation_rate, avg_neighbors

def main():
    print(f"🚀 正在加载 CODrone 数据集 (包含: {', '.join(TARGET_SPLITS)}) ...")
    
    all_files = []
    for split in TARGET_SPLITS:
        split_path = os.path.join(DATASET_ROOT, split, 'labelTxt')
        print(f"  - 扫描 {split} 集: {split_path} ...")
        
        if not os.path.exists(split_path):
            print(f"    [警告] 路径不存在: {split_path}")
            continue
            
        files = glob.glob(os.path.join(split_path, '*.txt'))
        print(f"    -> 找到 {len(files)} 个文件")
        all_files.extend(files)
    
    print(f"总计找到 {len(all_files)} 个标注文件。")
    
    if len(all_files) == 0:
        print("错误: 未找到任何 .txt 文件，请检查路径。")
        return
        
    use_files = all_files
    if SAMPLE_NUM != -1 and SAMPLE_NUM < len(all_files):
        print(f"警告: 代码设置为全量计算，忽略 SAMPLE_NUM={SAMPLE_NUM}")
        
    print(f"使用全部 {len(use_files)} 个文件进行全量计算...")
    print(f"⚠️  提示：数据加载已优化为 Numpy 模式，避免内存错误。")
        
    # 预加载数据到内存
    dataset_samples = []
    print("正在预处理标注数据 (这可能需要一分钟)...")
    
    # 使用多进程加速文件读取
    pool = multiprocessing.Pool(processes=min(16, multiprocessing.cpu_count()))
    for b, l in tqdm(pool.imap(parse_codrone_file, use_files, chunksize=100), total=len(use_files)):
        if b is not None:
            dataset_samples.append((b, l))
    pool.close()
    pool.join()
    
    print(f"预处理完成，有效样本数: {len(dataset_samples)}")
    
    print(f"\n{'='*80}")
    print(f"开始 K_RADIUS 参数搜索 (K_LIST: {K_LIST})")
    print(f"衡量标准: 寻找孤立率(Iso)较低，且平均混乱度(Chaos)也较低的平衡点")
    print(f"{'='*80}")
    print(f"{'K-Radius':<10} | {'Avg Chaos':<12} | {'Isolation%':<12} | {'Avg Neighbors':<15}")
    print("-" * 80)
    
    results = []
    
    for k in K_LIST:
        avg_chaos, iso_rate, avg_neigh = compute_naoa_metrics(k, dataset_samples)
        
        # 实时打印结果
        print(f"{k:<10.1f} | {avg_chaos:<12.4f} | {iso_rate*100:<11.2f}% | {avg_neigh:<15.2f}")
        
        results.append({
            'k': k,
            'chaos': avg_chaos,
            'iso': iso_rate
        })
        
    print("-" * 80)
    
    # === 自动推荐逻辑 ===
    candidates = [r for r in results if r['iso'] < 0.15]
    
    if not candidates:
        print("\n[分析] 数据集非常稀疏，即使 K 很大孤立率依然很高。")
        best = min(results, key=lambda x: x['iso']) 
        print(f"[推荐] 建议使用较大的 K = {best['k']} (孤立率 {best['iso']*100:.1f}%)")
    else:
        best = min(candidates, key=lambda x: x['chaos'])
        print(f"\n[推荐] 最优 K_RADIUS = {best['k']}")
        print(f"  理由: 在满足覆盖率(孤立率 < 15%)的前提下，")
        print(f"        该参数能保持最低的内部混乱度 ({best['chaos']:.4f})，")
        print(f"        说明邻域内的物体既丰富又整齐。")

if __name__ == "__main__":
    main()