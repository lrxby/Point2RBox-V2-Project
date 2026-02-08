import os
import glob
import torch
import numpy as np
import mmcv
from tqdm import tqdm
from mmengine import Config

# 尝试导入推理接口
try:
    from mmrotate.apis.inference import init_detector, inference_detector
except ImportError:
    from mmdet.apis import init_detector, inference_detector

from mmrotate.registry import VISUALIZERS
from mmrotate.utils import register_all_modules

def verify_model_predictions_advanced(config_path, checkpoint_path, img_dir, out_dir, score_thr=0.05):
    """
    高级诊断脚本：详细统计预测框的几何属性（尺寸、比例、角度）。
    """
    register_all_modules()

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    print(f"正在加载模型: {config_path}")
    try:
        model = init_detector(config_path, checkpoint_path, device='cuda:0')
    except Exception as e:
        print(f"模型加载失败: {e}")
        return

    # 初始化可视化器
    if hasattr(model.cfg, 'visualizer'):
        visualizer = VISUALIZERS.build(model.cfg.visualizer)
    else:
        from mmrotate.visualization import RotLocalVisualizer
        visualizer = RotLocalVisualizer(name='visualizer')
    visualizer.dataset_meta = model.dataset_meta

    # 获取图片
    extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
    img_paths = []
    for ext in extensions:
        img_paths.extend(glob.glob(os.path.join(img_dir, ext)))
    img_paths.sort()
    
    if len(img_paths) == 0:
        print(f"错误: 在 {img_dir} 下未找到图片。")
        return

    print(f"找到 {len(img_paths)} 张图片，开始推理前 50 张...")
    
    # === 数据收集容器 ===
    stats = {
        'widths': [],
        'heights': [],
        'angles': [], # 度数
        'ratios': [], # 长宽比 max(w,h)/min(w,h)
        'scores': []
    }

    max_imgs = 50
    count_valid_imgs = 0

    for i, img_path in enumerate(tqdm(img_paths[:max_imgs])):
        img_name = os.path.basename(img_path)
        
        try:
            result = inference_detector(model, img_path)
        except Exception as e:
            print(f"推理失败: {e}")
            continue
        
        pred_instances = result.pred_instances
        mask = pred_instances.scores > score_thr
        valid_bboxes = pred_instances.bboxes[mask] # [cx, cy, w, h, theta]
        valid_scores = pred_instances.scores[mask]
        
        if len(valid_bboxes) > 0:
            count_valid_imgs += 1
            
            # 提取数据 (转为numpy)
            np_bboxes = valid_bboxes.detach().cpu().numpy()
            np_scores = valid_scores.detach().cpu().numpy()
            
            ws = np_bboxes[:, 2]
            hs = np_bboxes[:, 3]
            thetas = np_bboxes[:, 4]
            
            stats['widths'].extend(ws.tolist())
            stats['heights'].extend(hs.tolist())
            stats['scores'].extend(np_scores.tolist())
            
            # 转换角度为度数 (假设是弧度)
            stats['angles'].extend((thetas * 180 / np.pi).tolist())
            
            # 计算长宽比 (不管哪个边长，取 长边/短边)
            # 防止除以0
            safe_ws = np.maximum(ws, 1e-6)
            safe_hs = np.maximum(hs, 1e-6)
            ratios = np.maximum(safe_ws / safe_hs, safe_hs / safe_ws)
            stats['ratios'].extend(ratios.tolist())

        # 可视化绘制
        try:
            img = mmcv.imread(img_path)
            img = mmcv.imconvert(img, 'bgr', 'rgb')
            visualizer.add_datasample(
                name=img_name,
                image=img,
                data_sample=result,
                draw_gt=False,
                draw_pred=True,
                show=False,
                out_file=os.path.join(out_dir, img_name),
                pred_score_thr=score_thr
            )
        except Exception as e:
            pass

    # === 详细统计报告 ===
    print("\n" + "="*60)
    print("【深度诊断报告】")
    num_boxes = len(stats['widths'])
    print(f"总检测框数量: {num_boxes} (来自 {count_valid_imgs} 张图片)")

    if num_boxes > 0:
        # Helper function
        def get_stats(data):
            return {
                'mean': np.mean(data),
                'min': np.min(data),
                'max': np.max(data),
                'median': np.median(data),
                'std': np.std(data)
            }

        w_s = get_stats(stats['widths'])
        h_s = get_stats(stats['heights'])
        a_s = get_stats(stats['angles'])
        r_s = get_stats(stats['ratios'])

        print("-" * 30)
        print(f"1. 尺寸统计 (Pixels):")
        print(f"   宽度 (W): 均值={w_s['mean']:.1f}, 中位数={w_s['median']:.1f}, 范围=[{w_s['min']:.1f}, {w_s['max']:.1f}]")
        print(f"   高度 (H): 均值={h_s['mean']:.1f}, 中位数={h_s['median']:.1f}, 范围=[{h_s['min']:.1f}, {h_s['max']:.1f}]")
        
        print("-" * 30)
        print(f"2. 形状分析 (长宽比 Ratio = Max(W,H)/Min(W,H)):")
        print(f"   均值 Ratio: {r_s['mean']:.1f}")
        print(f"   最大 Ratio: {r_s['max']:.1f}")
        print(f"   -> 如果 Ratio > 10，说明是“细长条”")
        print(f"   -> 如果 Ratio > 100，说明是“极度畸变”")

        print("-" * 30)
        print(f"3. 角度统计 (Degrees):")
        print(f"   均值: {a_s['mean']:.1f}°, 标准差: {a_s['std']:.1f}°")
        print(f"   -> 如果标准差接近 0，说明模型发生了“角度坍塌”，只会输出一个固定角度。")

        print("-" * 30)
        print("【最终结论推断】")
        if w_s['mean'] > 2000 or h_s['mean'] > 2000:
            print("🔴 [尺寸爆炸] 模型正在预测全图大小的框。")
            print("   原因：缺乏 loss_area 或 loss_overlap，模型通过最大化面积来覆盖物体。")
        elif r_s['mean'] > 20:
            print("🟠 [条纹伪影] 模型预测出了极其细长的条纹。")
            print("   原因：Box-Sensitive Loss 只有推力没有拉力，模型找到了“扫描线”作弊解法。")
        else:
            print("🟢 尺寸分布看起来相对正常，请检查 IoU 匹配问题。")

    else:
        print("未检测到任何框。")
    print("="*60)

if __name__ == '__main__':
    # ================= 配置区域 =================
    config_file = '/mnt/data/liurunxiang/workplace/point2rbox-v2-ourloss/configs/point2rbox_v2/point2rbox_v2-1x-dota.py' 
    checkpoint_file = 'work_dirs/1/Lcls1/epoch_1.pth' # 换成你最新的权重
    image_dir = '/mnt/data/xiekaikai/split_ss_dota/trainval/images'
    output_dir = '/mnt/data/liurunxiang/workplace/point2rbox-v2-ourloss/work_dirs/1/visual-Lcls1'
    score_threshold = 0.1
    # ===========================================

    verify_model_predictions_advanced(config_file, checkpoint_file, image_dir, output_dir, score_threshold)