import os
import glob
import math
import numpy as np
import cv2
import torch
from tqdm import tqdm
import multiprocessing as mp

# ================= 配置区域 =================
# DroneVehicle 数据集根路径
DATASET_ROOT = '/mnt/data/xiekaikai/DroneVehicle'
# 需要计算的子集名称
TARGET_SPLITS = ['train', 'val', 'test']

# 搜索的 K 值列表 (设置得比较密，以便寻找拐点)
K_LIST = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 
          1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 
          2.2, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

# DroneVehicle 类别定义 (5类)
CLASSES = ('car', 'bus', 'truck', 'van', 'freight_car')

# 采样文件数量 (-1 表示跑全量)
SAMPLE_NUM = -1 

# 设备
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 多进程配置
NUM_WORKERS = mp.cpu_count() - 2  # 留2个核心给系统，避免卡死
# ===========================================

def parse_drone_file(file_path):
    """
    解析 DroneVehicle 格式 txt 文件
    样例: 393 443 419 441 419 489 396 489 0
    解析逻辑: 前8个是坐标，第9个是类别索引 (假设)
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
            # 提取 8 点坐标
            poly = np.array([float(x) for x in parts[:8]]).reshape(4, 2).astype(np.float32)
            # 提取类别索引 (假设最后一个数字是类别ID)
            cls_id = int(parts[8]) 
        except ValueError:
            continue
            
        # 过滤非法类别索引
        if cls_id < 0 or cls_id >= len(CLASSES):
            continue
            
        # 转换为旋转矩形 (cx, cy, w, h, angle_deg)
        rect = cv2.minAreaRect(poly)
        (cx, cy), (w, h), angle = rect
        
        # 统一角度定义 (OpenCV 4.5+ 范围 [-90, 0))
        if w < h:
            w, h = h, w
            angle += 90
        
        # 转换为弧度
        theta = np.deg2rad(angle)
        
        bboxes.append([cx, cy, w, h, theta])
        labels.append(cls_id)
        
    if not bboxes:
        return None, None
        
    return torch.tensor(bboxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)

def compute_naoa_metrics(k_radius, dataset_samples):
    """
    使用指定的 k_radius 在数据集上运行 NAOALoss V4 计算流程
    返回: 平均混乱度, 孤立率, 平均邻居数
    """
    total_chaos_sum = 0.0
    total_valid_samples = 0
    total_isolated_count = 0
    total_objects = 0
    total_neighbor_sum = 0.0
    
    for bboxes, labels in dataset_samples:
        if bboxes is None or len(bboxes) < 2:
            continue
            
        # 搬运到 GPU 计算加速
        bboxes = bboxes.to(DEVICE)
        labels = labels.to(DEVICE)
        
        N = bboxes.shape[0]
        total_objects += N
        
        # ================= Step 1: 几何解耦 =================
        centers = bboxes[:, :2]
        wh = bboxes[:, 2:4]
        # 计算尺度并截断
        scales = (wh[:, 0] * wh[:, 1]).sqrt().clamp(min=16.0, max=800.0)
        thetas = bboxes[:, 4]
        
        # ================= Step 2: 矢量化 (4-Theta) =================
        vecs = torch.stack([torch.cos(4 * thetas), torch.sin(4 * thetas)], dim=1)
        
        # ================= Step 3: 构建亲和矩阵 =================
        # 3.1 空间距离权重
        dist_sq = torch.cdist(centers, centers, p=2).pow(2)
        sigmas = scales * k_radius # <--- 只有这里用到了 k_radius
        sigma_mat = sigmas.view(N, 1)
        
        # Gaussian Kernel
        W_geo = torch.exp(-dist_sq / (2 * sigma_mat.pow(2)))
        
        # 3.3 逻辑掩码 (仅同类)
        mask_cls = (labels.view(N, 1) == labels.view(1, N)).float()
        
        # 组合权重 (包含自环)
        W = W_geo * mask_cls
        
        # --- 统计指标计算 ---
        
        # 1. 计算有效邻居数 (减去自环的 1.0)
        W_no_diag = W.clone()
        W_no_diag.fill_diagonal_(0)
        neighbor_strength = W_no_diag.sum(dim=1)
        
        # 定义孤立点: 邻居权重之和小于 0.1
        is_isolated = neighbor_strength < 0.1
        total_isolated_count += is_isolated.sum().item()
        total_neighbor_sum += neighbor_strength.sum().item()
        
        # ================= Step 4: 归一化 =================
        W_sum = W.sum(dim=1, keepdim=True)
        W_norm = W / W_sum
        
        # ================= Step 5: 能量计算 =================
        mean_vecs = torch.mm(W_norm, vecs)
        chaos_score = 1.0 - mean_vecs.norm(dim=1)
        
        # 只统计非孤立点的混乱度
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
    print(f"正在加载 DroneVehicle 数据集 (包含: {', '.join(TARGET_SPLITS)}) ...")
    
    # 1. 收集所有标注文件路径
    all_files = []
    for split in TARGET_SPLITS:
        split_path = os.path.join(DATASET_ROOT, split, 'annfiles')
        print(f"  - 扫描 {split} 集: {split_path} ...")
        
        if not os.path.exists(split_path):
            print(f"    [警告] 路径不存在: {split_path}")
            continue
            
        files = glob.glob(os.path.join(split_path, '*.txt'))
        print(f"    -> 找到 {len(files)} 个文件")
        all_files.extend(files)
    
    if len(all_files) == 0:
        print("错误: 未找到任何 .txt 文件，请检查路径。")
        return
    
    # 2. 采样逻辑（和DOTA代码保持一致）
    use_files = all_files
    if SAMPLE_NUM != -1 and SAMPLE_NUM < len(all_files):
        print(f"总文件数 {len(all_files)}，随机采样 {SAMPLE_NUM} 个进行测试...")
        use_files = np.random.choice(all_files, SAMPLE_NUM, replace=False)
    else:
        print(f"使用全部 {len(use_files)} 个文件进行全量计算...")
    
    # 3. 多进程并行解析标注文件（核心加速点）
    print(f"正在预处理标注数据 (使用 {NUM_WORKERS} 个进程)...")
    dataset_samples = []
    # 使用进程池+进度条
    with mp.Pool(processes=NUM_WORKERS) as pool:
        # 并行执行解析
        results = list(tqdm(pool.imap(parse_drone_file, use_files), total=len(use_files)))
    
    # 过滤空数据
    dataset_samples = [res for res in results if res[0] is not None]
    print(f"预处理完成，有效样本数: {len(dataset_samples)}")
    
    # 4. K值遍历计算（和DOTA代码逻辑一致）
    print(f"\n{'='*80}")
    print(f"开始 K_RADIUS 参数搜索 (K_LIST: {K_LIST})")
    print(f"衡量标准: 寻找孤立率(Iso)较低，且平均混乱度(Chaos)也较低的平衡点")
    print(f"{'='*80}")
    print(f"{'K-Radius':<10} | {'Avg Chaos':<12} | {'Isolation%':<12} | {'Avg Neighbors':<15}")
    print("-" * 80)
    
    results = []
    
    for k in K_LIST:
        avg_chaos, iso_rate, avg_neigh = compute_naoa_metrics(k, dataset_samples)
        
        print(f"{k:<10.1f} | {avg_chaos:<12.4f} | {iso_rate*100:<11.2f}% | {avg_neigh:<15.2f}")
        
        results.append({
            'k': k,
            'chaos': avg_chaos,
            'iso': iso_rate
        })
        
    print("-" * 80)
    
    # === 自动推荐逻辑（和DOTA代码保持一致）===
    # 1. 筛选出孤立率 < 15% 的候选者 (保证大部分物体都有邻居)
    candidates = [r for r in results if r['iso'] < 0.15]
    
    if not candidates:
        print("\n[分析] 数据集非常稀疏，即使 K 很大孤立率依然很高。")
        best = min(results, key=lambda x: x['iso']) # 选孤立率最低的
        print(f"[推荐] 建议使用较大的 K = {best['k']} (孤立率 {best['iso']*100:.1f}%)")
    else:
        # 2. 在候选者中，选混乱度最低的 (肘部点)
        best = min(candidates, key=lambda x: x['chaos'])
        print(f"\n[推荐] 最优 K_RADIUS = {best['k']}")
        print(f"  理由: 在满足覆盖率(孤立率 < 15%)的前提下，")
        print(f"        该参数能保持最低的内部混乱度 ({best['chaos']:.4f})，")
        print(f"        说明邻域内的物体既丰富又整齐。")

if __name__ == "__main__":
    # 修复多进程在Windows/Linux下的兼容性问题
    mp.set_start_method('fork', force=True)
    main()