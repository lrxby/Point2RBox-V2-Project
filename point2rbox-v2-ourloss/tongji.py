import os
import glob
import numpy as np
import math
import cv2
from tqdm import tqdm

def calculate_dota_stats(data_root):
    # 路径拼接
    label_dir = os.path.join(data_root, 'labelTxt')
    
    # 获取所有txt文件
    txt_files = glob.glob(os.path.join(label_dir, '*.txt'))
    print(f"找到 {len(txt_files)} 个标注文件，开始统计...")

    total_objects = 0
    
    # 用于累加数据
    sum_cx = 0.0
    sum_cy = 0.0
    sum_w = 0.0
    sum_h = 0.0
    sum_angle = 0.0
    
    # 另外统计长短边，防止 w, h 因为角度问题弄反
    sum_long_side = 0.0
    sum_short_side = 0.0

    # 遍历文件
    for txt_file in tqdm(txt_files):
        with open(txt_file, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            # DOTA 格式: x1 y1 x2 y2 x3 y3 x4 y4 classname difficulty
            parts = line.strip().split()
            
            # 跳过空行或格式错误的行
            if len(parts) < 10:
                continue
                
            try:
                # 提取8个坐标点
                coords = list(map(float, parts[:8]))
                points = np.array(coords).reshape(4, 2).astype(np.float32)
                
                # 使用 OpenCV 计算最小外接矩形 ((cx, cy), (w, h), theta)
                rect = cv2.minAreaRect(points)
                (cx, cy), (w, h), angle = rect
                
                # 过滤掉异常的小框 (例如噪点)
                if w < 1 or h < 1:
                    continue

                sum_cx += cx
                sum_cy += cy
                sum_w += w
                sum_h += h
                sum_angle += angle
                
                # 统计长边和短边（更具有物理意义）
                sum_long_side += max(w, h)
                sum_short_side += min(w, h)
                
                total_objects += 1
                
            except Exception as e:
                # 忽略解析错误的行
                continue

    if total_objects == 0:
        print("未找到任何有效目标！请检查路径是否正确。")
        return

    # 计算平均值
    avg_cx = sum_cx / total_objects
    avg_cy = sum_cy / total_objects
    avg_w = sum_w / total_objects
    avg_h = sum_h / total_objects
    avg_angle = sum_angle / total_objects
    
    avg_long = sum_long_side / total_objects
    avg_short = sum_short_side / total_objects

    print("\n" + "="*40)
    print(f"【统计结果】 共处理 {total_objects} 个目标")
    print("-" * 40)
    print(f"平均中心 X (cx): {avg_cx:.2f}")
    print(f"平均中心 Y (cy): {avg_cy:.2f}")
    print("-" * 40)
    print(f"OpenCV 原生平均 W: {avg_w:.2f}")
    print(f"OpenCV 原生平均 H: {avg_h:.2f}")
    print("-" * 40)
    print(f"✅ 平均长边 (Long Side): {avg_long:.2f}")
    print(f"✅ 平均短边 (Short Side): {avg_short:.2f}")
    print("-" * 40)
    print(f"平均角度 (Angle): {avg_angle:.2f} 度")
    print("="*40)
    
    print("\n【建议配置】")
    print(f"在 Config 中设置 prior_box_size=[{avg_short:.1f}, {avg_long:.1f}] (推荐用短边, 长边)")
    print("或者 prior_box_size=[32.0, 32.0] (如果你希望从更小的基准开始)")

if __name__ == '__main__':
    # 修改这里的路径为你实际的路径
    DOTA_PATH = '/mnt/data/xiekaikai/split_ss_dota/trainval'
    
    calculate_dota_stats(DOTA_PATH)
