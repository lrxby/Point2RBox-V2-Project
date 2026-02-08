import os
import glob
import torch
import numpy as np
import mmcv
import copy
from tqdm import tqdm
from mmengine import Config

# 尝试导入推理函数
try:
    from mmrotate.apis.inference import init_detector, inference_detector
except ImportError:
    from mmdet.apis import init_detector, inference_detector

from mmrotate.registry import VISUALIZERS
from mmrotate.utils import register_all_modules

def verify_model_predictions(config_path, checkpoint_path, img_dir, out_dir, score_thr=0.05):
    """
    运行推理，统计尺寸，并【强制修正尺寸】进行可视化，以验证中心点回归是否正常。
    """
    # 1. 注册所有模块
    register_all_modules()

    # 2. 准备输出目录
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # 3. 初始化模型
    print(f"正在加载模型: {config_path}")
    try:
        model = init_detector(config_path, checkpoint_path, device='cuda:0')
    except Exception as e:
        print(f"模型加载失败: {e}")
        return

    # 4. 初始化可视化器
    if hasattr(model.cfg, 'visualizer'):
        visualizer = VISUALIZERS.build(model.cfg.visualizer)
    else:
        from mmrotate.visualization import RotLocalVisualizer
        visualizer = RotLocalVisualizer(name='visualizer')
        
    visualizer.dataset_meta = model.dataset_meta

    # 5. 获取图片列表
    extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
    img_paths = []
    for ext in extensions:
        img_paths.extend(glob.glob(os.path.join(img_dir, ext)))
    img_paths.sort()
    
    if len(img_paths) == 0:
        print(f"错误: 在 {img_dir} 下未找到图片。")
        return

    print(f"找到 {len(img_paths)} 张图片，开始推理...")
    
    # 统计数据
    total_w = []
    total_h = []
    max_imgs = 50 
    
    for i, img_path in enumerate(tqdm(img_paths[:max_imgs])):
        img_name = os.path.basename(img_path)
        
        try:
            # --- A. 推理 ---
            result = inference_detector(model, img_path)
            pred_instances = result.pred_instances
            
            # --- B. 统计原始尺寸 (用于诊断) ---
            mask = pred_instances.scores > score_thr
            valid_bboxes = pred_instances.bboxes[mask]
            
            if len(valid_bboxes) > 0:
                ws = valid_bboxes[:, 2]
                hs = valid_bboxes[:, 3]
                total_w.extend(ws.detach().cpu().numpy().tolist())
                total_h.extend(hs.detach().cpu().numpy().tolist())

                # 打印前几张图的异常数值，让你直观看到“爆炸”
                if i < 3:
                    print(f"\n[诊断信息] 图片: {img_name}")
                    print(f"检测到 {len(valid_bboxes)} 个目标")
                    print(f"示例框尺寸 (W, H): ({ws[0]:.1f}, {hs[0]:.1f})")
                    if ws[0] > 2000:
                        print("-> 警告：尺寸异常巨大！")

            # --- C. 强制修正尺寸 (为了可视化) ---
            # 既然原始框太大画不出来，我们把它们强制缩小到 50x50
            # 这样如果中心点是对的，你就能在物体位置看到一个个小方框
            
            # 克隆结果以免修改原数据
            vis_result = result.clone()
            
            # 获取所有框 (不过滤阈值的也要改，防止报错，虽然画的时候会过滤)
            if len(vis_result.pred_instances) > 0:
                # bboxes: [cx, cy, w, h, theta]
                # 强制将 W 和 H 设为 50 像素
                vis_result.pred_instances.bboxes[:, 2] = 50.0 
                vis_result.pred_instances.bboxes[:, 3] = 50.0
            
            # --- D. 绘制 ---
            img = mmcv.imread(img_path)
            img = mmcv.imconvert(img, 'bgr', 'rgb')
            
            visualizer.add_datasample(
                name=img_name,
                image=img,
                data_sample=vis_result, # <--- 传入修改过尺寸的结果
                draw_gt=False,
                draw_pred=True,
                show=False,
                out_file=os.path.join(out_dir, img_name),
                pred_score_thr=score_thr
            )
            
        except Exception as e:
            print(f"处理图片 {img_name} 失败: {e}")
            continue

    # --- 输出报告 ---
    print("\n" + "="*50)
    print("【诊断报告】")
    if len(total_w) > 0:
        global_avg_w = np.mean(total_w)
        global_avg_h = np.mean(total_h)
        print(f"平均宽度: {global_avg_w:.1f}")
        print(f"平均高度: {global_avg_h:.1f}")
        
        if global_avg_w > 5000:
            print("\n[结论]: 确认发生【尺寸爆炸 (Size Explosion)】。")
            print("模型预测的框巨大无比，导致 IoU=0 (mAP=0) 且无法绘制。")
            print("脚本已强制将框缩小为 50x50 并保存。请查看输出图片：")
            print("如果图片上的小框确实标在物体中心，说明 Point 监督是有效的，仅需修复尺寸发散问题。")
    print("="*50)
    print(f"可视化结果目录: {out_dir}")

if __name__ == '__main__':
    # ================= 配置区域 =================
    # 1. 配置文件路径
    config_file = '/mnt/data/liurunxiang/workplace/point2rbox-v2-ourloss/configs/point2rbox_v2/point2rbox_v2-1x-dota.py' 
    
    # 2. 训练好的权重文件路径
    checkpoint_file = 'work_dirs/Lcls/epoch_1.pth'
    
    # 3. 数据集图片路径
    image_dir = '/mnt/data/xiekaikai/split_ss_dota/trainval/images'
    
    # 4. 结果保存路径
    output_dir = '/mnt/data/liurunxiang/workplace/point2rbox-v2-ourloss/work_dirs/visual-Lcls5'
    
    # 5. 置信度阈值
    score_threshold = 0.05
    # ===========================================

    verify_model_predictions(config_file, checkpoint_file, image_dir, output_dir, score_threshold)