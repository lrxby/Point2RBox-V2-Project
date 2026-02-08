import os
import random
import cv2
import torch
import numpy as np
import mmcv
from pathlib import Path

# [修改点 1] 改为从 mmdet.apis 导入通用推理函数
from mmdet.apis import init_detector, inference_detector
# [修改点 2] 导入 mmrotate 模块以确保旋转框相关的 Model/Metric 被注册到注册表中
import mmrotate 
from mmrotate.registry import VISUALIZERS
from mmengine.structures import InstanceData

# ================= 配置区域 =================
# 1. 图片文件夹路径
IMG_DIR = '/mnt/data/xiekaikai/DroneVehicle/val/images'

# 2. 模型权重路径
CHECKPOINT_FILE = '/mnt/data/liurunxiang/workplace/point2rbox-v2-UAV-loss1/work_dirs/dv/1/std0.1/epoch_12.pth'

# 3. [重要] 配置文件路径
CONFIG_FILE = '/mnt/data/liurunxiang/workplace/point2rbox-v2-UAV-loss1/configs/point2rbox_v2/point2rbox_v2-1x-dronevehicle.py' 

# 4. 结果保存路径
OUT_DIR = 'work_dirs/vis_results_weight0.1_check'
# ===========================================

def main():
    # 0. 准备工作
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"正在加载模型...\nConfig: {CONFIG_FILE}\nCheckpoint: {CHECKPOINT_FILE}")
    
    # 1. 初始化模型
    try:
        # init_detector 会自动解析配置文件中的 model.type (Point2RBoxV2)
        # 只要导入了 mmrotate，注册表里就有这个类
        model = init_detector(CONFIG_FILE, CHECKPOINT_FILE, device=device)
    except Exception as e:
        print(f"模型加载失败: {e}")
        print("请检查 CONFIG_FILE 路径是否正确，是否对应了正确的配置文件。")
        return

    # 初始化可视化工具
    # 配置文件里通常定义了 visualizer (RotLocalVisualizer)，这里构建它
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.dataset_meta = model.dataset_meta

    # 2. 获取图片列表
    supported_ext = ['.png', '.jpg', '.jpeg', '.bmp', '.tif']
    all_imgs = [f for f in os.listdir(IMG_DIR) if os.path.splitext(f)[-1].lower() in supported_ext]
    
    if len(all_imgs) == 0:
        print(f"错误: 在 {IMG_DIR} 下未找到图片。")
        return

    # 3. 随机抽取 20 张
    num_samples = min(20, len(all_imgs))
    selected_imgs = random.sample(all_imgs, num_samples)
    print(f"已随机抽取 {num_samples} 张图片进行推理和验证...\n")

    print(f"{'图片名称':<30} | {'检测框数量':<10} | {'角度均值(rad)':<15} | {'角度方差':<15} | {'判定结果'}")
    print("-" * 100)

    for img_name in selected_imgs:
        img_path = os.path.join(IMG_DIR, img_name)
        out_path = os.path.join(OUT_DIR, img_name)

        # 4. 推理
        img = mmcv.imread(img_path)
        result = inference_detector(model, img)

        # 5. 统计角度数据 (验证模式坍塌的核心步骤)
        pred_instances = result.pred_instances
        
        # 过滤低置信度的框，只统计模型确信的结果 (score > 0.3)
        valid_mask = pred_instances.scores > 0.3
        valid_bboxes = pred_instances.bboxes[valid_mask]
        
        status = "无目标"
        mean_angle = 0.0
        var_angle = 0.0

        if len(valid_bboxes) > 0:
            # Point2RBox/RotatedBoxes 输出格式通常是 (x, y, w, h, theta)
            # theta 是最后一维 (索引 4)
            angles = valid_bboxes[:, 4].cpu().numpy()
            
            mean_angle = np.mean(angles)
            var_angle = np.var(angles)
            
            # 判断逻辑：如果方差极小 (< 0.01)，说明所有框角度几乎一样
            if var_angle < 0.01:
                status = "🔴 疑似坍塌 (角度固定)"
            else:
                status = "🟢 分布正常"
        else:
            # 如果没有高分框，尝试不过滤看一眼
            if len(pred_instances.bboxes) > 0:
                angles = pred_instances.bboxes[:, 4].cpu().numpy()
                mean_angle = np.mean(angles)
                var_angle = np.var(angles)
                status = "⚠️ 仅低分框 (疑似坍塌)" if var_angle < 0.01 else "⚠️ 仅低分框"

        print(f"{img_name:<30} | {len(valid_bboxes):<10} | {mean_angle:.4f}          | {var_angle:.6f}       | {status}")

        # 6. 可视化并保存
        visualizer.add_datasample(
            name='result',
            image=img,
            data_sample=result,
            draw_gt=False,
            wait_time=0,
            out_file=out_path,
            pred_score_thr=0.3 # 只画置信度大于 0.3 的
        )

    print("-" * 100)
    print(f"\n结果已保存至: {os.path.abspath(OUT_DIR)}")
    print("请查看保存的图片。如果所有框的方向看起来都一样（例如全部水平或全部垂直），且上方统计的方差接近0，则证实发生了模式坍塌。")

if __name__ == '__main__':
    main()
