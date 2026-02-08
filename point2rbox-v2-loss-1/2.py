import os
import random
import cv2
import numpy as np
import sys

# ================= 配置区域 =================
# 图片文件夹路径
IMG_DIR = '/mnt/data/xiekaikai/split_ss_dota/trainval/images'

# 标注文件夹路径
LABEL_DIR = '/mnt/data/xiekaikai/split_ss_dota/trainval/annfiles'

# [结果保存路径]
OUTPUT_DIR = 'work_dirs/vis_ground_truth_clean' 
# ===========================================

# 定义不同类别的颜色 (BGR格式)
CLASS_COLORS = {
    'plane': (0, 255, 0),            # 绿色
    'baseball-diamond': (0, 0, 255), # 红色
    'bridge': (255, 0, 0),           # 蓝色
    'ground-track-field': (0, 255, 255), # 黄色
    'small-vehicle': (255, 0, 255),  # 紫色
    'large-vehicle': (255, 165, 0),  # 青色
    'ship': (0, 165, 255),           # 橙色
    'tennis-court': (128, 0, 128),   # 深紫
    'basketball-court': (128, 128, 0), # 暗青
    'storage-tank': (0, 128, 128),   # 橄榄色
    'soccer-ball-field': (0, 0, 128), # 深红
    'roundabout': (128, 0, 0),       # 深蓝
    'harbor': (0, 128, 0),           # 深绿
    'swimming-pool': (255, 192, 203),# 粉色
    'helicopter': (255, 255, 255)    # 白色
}

def draw_dota_gt(img_path, label_path, save_path):
    """读取图片和DOTA标注，画框并保存 (每类只标一次文字)"""
    if not os.path.exists(img_path):
        # 尝试寻找不同后缀的图片
        base_path = os.path.splitext(img_path)[0]
        found = False
        for ext in ['.png', '.jpg', '.bmp', '.tif']:
            if os.path.exists(base_path + ext):
                img_path = base_path + ext
                found = True
                break
        if not found:
            print(f"图片不存在: {img_path}")
            return

    if not os.path.exists(label_path):
        print(f"标注文件不存在: {label_path}")
        return

    # 读取图片
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print(f"无法读取图片: {img_path}")
        return

    with open(label_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 用于记录当前图片中已标注文字的类别
    labeled_classes_in_this_img = set()
    has_obj = False

    # 为了更好的显示效果，可以先对对象按面积排序？
    # 这里简单起见，按读取顺序，但为了防止第一个是很小的截断框，
    # 我们可以稍微增加一个判断：优先给没有difficulty=1的框标字。
    
    # 解析所有数据
    objects = []
    for line in lines:
        line = line.strip()
        if not line or 'imagesource' in line or 'gsd' in line:
            continue
        parts = line.split()
        if len(parts) < 9:
            continue
        
        try:
            poly = list(map(float, parts[:8]))
            classname = parts[8]
            difficulty = int(parts[9]) if len(parts) >= 10 else 0
            objects.append({
                'poly': poly,
                'class': classname,
                'diff': difficulty
            })
        except:
            continue

    # 开始绘制
    for obj in objects:
        classname = obj['class']
        difficulty = obj['diff']
        poly = obj['poly']

        # 构造多边形
        pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
        color = CLASS_COLORS.get(classname, (200, 200, 200))
        
        # 1. 画框 (所有目标都画)
        # 困难样本画细一点，或者用虚线(cv2不支持直接虚线，这里简化为调整粗细)
        thickness = 1 if difficulty == 1 else 2
        cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)
        has_obj = True

        # 2. 画文字 (每类只画一次，且优先画非困难样本)
        # 逻辑：如果这个类还没标过字，或者之前标的是diff=1但现在遇到了diff=0的（覆盖之前的不好做，这里简化为遇到第一个有效框就标）
        
        should_label = False
        if classname not in labeled_classes_in_this_img:
            # 如果还没标过，就标
            should_label = True
            
            # 小优化：如果当前是困难样本，暂缓标记，除非它是该类唯一的样本？
            # 简化逻辑：直接标记第一个遇到的
            labeled_classes_in_this_img.add(classname)

        if should_label:
            text = classname
            if difficulty == 1:
                text += "(diff)"

            font_scale = 0.6
            font_thickness = 1
            (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            
            # 标签位置
            x, y = int(poly[0]), int(poly[1])
            
            # 防止文字画出图片上边界
            text_y = y - 5
            if text_y < text_height:
                text_y = y + text_height + 5

            # 画文字背景框
            cv2.rectangle(img, (x, text_y - text_height), (x + text_width, text_y + baseline), color, -1)
            # 画文字 (白色)
            cv2.putText(img, text, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)

    # 保存图片
    if has_obj:
        if not os.path.exists(save_path):
            _, ext = os.path.splitext(save_path)
            if not ext: 
                save_path += '.png'
                ext = '.png'
            cv2.imencode(ext, img)[1].tofile(save_path)

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"创建输出目录: {OUTPUT_DIR}")

    # 1. 获取所有图片列表
    valid_exts = ['.png', '.jpg', '.jpeg', '.bmp', '.tif']
    all_imgs = [f for f in os.listdir(IMG_DIR) if os.path.splitext(f)[-1].lower() in valid_exts]
    
    total_imgs = len(all_imgs)
    print(f"在 {IMG_DIR} 中找到 {total_imgs} 张图片。")

    if total_imgs == 0:
        return

    # 2. 随机抽取 50 张
    num_sample = 50
    selected_imgs = random.sample(all_imgs, min(num_sample, total_imgs))
    
    print(f"开始处理随机抽取的 {len(selected_imgs)} 张图片...")

    count = 0
    for img_name in selected_imgs:
        img_path = os.path.join(IMG_DIR, img_name)
        
        # 匹配标注文件
        basename = os.path.splitext(img_name)[0]
        label_name = basename + '.txt'
        label_path = os.path.join(LABEL_DIR, label_name)
        
        save_path = os.path.join(OUTPUT_DIR, f"vis_{basename}.png")
        
        draw_dota_gt(img_path, label_path, save_path)
        count += 1
        
        if count % 10 == 0:
            print(f"已处理 {count} 张...")

    print(f"\n全部完成！结果保存在: {os.path.abspath(OUTPUT_DIR)}")
    print("提示：每张图中，同一种颜色的框代表同一类目标，但为了视野清晰，只有其中一个框标注了文字名称。")

if __name__ == '__main__':
    main()