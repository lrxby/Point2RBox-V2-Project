import os
import glob
import torch
import numpy as np
import mmcv
import cv2
import tempfile
from tqdm import tqdm
from mmengine import Config

# 尝试导入推理接口
try:
    from mmrotate.apis.inference import init_detector, inference_detector
except ImportError:
    from mmdet.apis import init_detector, inference_detector

from mmrotate.registry import VISUALIZERS
from mmrotate.utils import register_all_modules

# ========== 自定义调色板（至少15种颜色，对应DOTA 15个类别） ==========
def get_dota_palette(num_classes):
    """生成足够长度的DOTA数据集调色板"""
    # 基础颜色（15种，覆盖DOTA所有类别）
    base_palette = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
        (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
        (128, 0, 128), (0, 128, 128), (64, 0, 0), (0, 64, 0), (0, 0, 64)
    ]
    # 如果类别数超过15，循环扩展调色板
    palette = []
    for i in range(num_classes):
        palette.append(base_palette[i % len(base_palette)])
    return palette

def verify_model_predictions_advanced(config_path, checkpoint_path, img_dir, out_dir, score_thr=0.05):
    """
    高级诊断脚本：详细统计预测框的几何属性（尺寸、比例、角度），确保框和标签同时显示。
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

    # ========== 核心：DOTA固定15类，避免配置路径错误 ==========
    num_classes = 15
    # DOTA官方类别列表
    dota_classes = (
        'plane', 'baseball-diamond', 'bridge', 'ground-track-field',
        'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
        'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout',
        'harbor', 'swimming-pool', 'helicopter'
    )
    class_names = dota_classes

    # 可选：尝试从配置文件自动读取（备用方案）
    try:
        cfg = Config.fromfile(config_path)
        # 尝试常见的类别数配置路径
        if hasattr(cfg.model, 'bbox_head'):
            auto_num_classes = cfg.model.bbox_head.num_classes
        elif hasattr(cfg.model, 'head'):
            auto_num_classes = cfg.model.head.num_classes
        elif hasattr(cfg.model, 'roi_head'):
            auto_num_classes = cfg.model.roi_head.bbox_head.num_classes
        else:
            auto_num_classes = num_classes
        
        if auto_num_classes != num_classes:
            print(f"警告：配置文件中类别数({auto_num_classes})与DOTA标准类别数(15)不一致！")
            print(f"将使用DOTA标准15类进行可视化")
    except Exception as e:
        print(f"从配置文件读取类别数失败: {e}，使用DOTA标准15类")

    # ========== 核心修复：可视化器初始化（移除顶层palette参数） ==========
    visualizer_cfg = dict(
        type='RotLocalVisualizer',
        name='visualizer',
        vis_backends=[{'type': 'LocalVisBackend', 'save_dir': out_dir}],
        line_width=2,
        # 关键修复：移除顶层palette，仅在dataset_meta中指定
        dataset_meta=dict(
            classes=class_names,
            palette=get_dota_palette(num_classes)  # palette仅放在dataset_meta中
        )
    )
    visualizer = VISUALIZERS.build(visualizer_cfg)
    # 强制覆盖model的dataset_meta（确保类别和调色板一致）
    visualizer.dataset_meta = {
        'classes': class_names,
        'palette': get_dota_palette(num_classes)
    }

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
        'angles': [],
        'ratios': [],
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
        valid_bboxes = pred_instances.bboxes[mask]
        
        if len(valid_bboxes) > 0:
            count_valid_imgs += 1
            np_bboxes = valid_bboxes.detach().cpu().numpy()
            np_scores = pred_instances.scores[mask].detach().cpu().numpy()
            ws = np_bboxes[:, 2]
            hs = np_bboxes[:, 3]
            thetas = np_bboxes[:, 4]
            stats['widths'].extend(ws.tolist())
            stats['heights'].extend(hs.tolist())
            stats['scores'].extend(np_scores.tolist())
            stats['angles'].extend((thetas * 180 / np.pi).tolist())
            safe_ws = np.maximum(ws, 1e-6)
            safe_hs = np.maximum(hs, 1e-6)
            ratios = np.maximum(safe_ws / safe_hs, safe_hs / safe_ws)
            stats['ratios'].extend(ratios.tolist())

        # ========== 框和标签同时显示逻辑 ==========
        try:
            # 1. 创建临时文件
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                tmp_path = tmp_file.name
            
            # 2. 可视化器绘制检测框
            img = mmcv.imread(img_path)
            visualizer.add_datasample(
                name=img_name,
                image=mmcv.imconvert(img, 'bgr', 'rgb'),
                data_sample=result,
                draw_gt=False,
                draw_pred=True,
                show=False,
                out_file=tmp_path,
                pred_score_thr=score_thr
            )
            
            # 3. 读取带框图片并删除临时文件
            img_with_boxes = cv2.imread(tmp_path)
            os.remove(tmp_path)
            
            # 4. 提取标注信息
            pred_instances = result.pred_instances
            mask = pred_instances.scores > score_thr
            if mask.sum() == 0:
                cv2.imwrite(os.path.join(out_dir, img_name), img_with_boxes)
                continue
            
            valid_bboxes = pred_instances.bboxes[mask].detach().cpu().numpy()
            valid_scores = pred_instances.scores[mask].detach().cpu().numpy()
            valid_labels = pred_instances.labels[mask].detach().cpu().numpy()
            
            # 5. 叠加标签到图片
            for bbox, score, label in zip(valid_bboxes, valid_scores, valid_labels):
                cx, cy, w, h, theta = bbox
                x_offset = -w/2 * np.cos(theta) - h/2 * np.sin(theta)
                y_offset = -w/2 * np.sin(theta) + h/2 * np.cos(theta)
                label_x = int(cx + x_offset)
                label_y = int(cy + y_offset)
                # 防止标签超出图片边界
                label_x = max(10, min(label_x, img_with_boxes.shape[1]-100))
                label_y = max(20, min(label_y, img_with_boxes.shape[0]-20))
                label_text = f"{class_names[label]} {score:.2f}"
                
                # 绘制标签背景和文字
                (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(
                    img_with_boxes, 
                    (label_x-2, label_y-text_h-4), 
                    (label_x+text_w+2, label_y+2), 
                    (0, 0, 0), 
                    -1
                )
                cv2.putText(
                    img_with_boxes,
                    label_text,
                    (label_x, label_y-2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1
                )
            
            # 6. 保存最终图片
            cv2.imwrite(os.path.join(out_dir, img_name), img_with_boxes)
            
        except Exception as e:
            print(f"可视化失败 {img_name}: {e}")

    # === 详细统计报告 ===
    print("\n" + "="*60)
    print("【深度诊断报告】")
    num_boxes = len(stats['widths'])
    print(f"总检测框数量: {num_boxes} (来自 {count_valid_imgs} 张图片)")

    if num_boxes > 0:
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
    config_file = '/mnt/data/liurunxiang/workplace/point2rbox-v2-our/configs/point2rbox_v2/point2rbox_v2-1x-dota.py' 
    checkpoint_file = 'work_dirs/dt/1/e2e/epoch_1.pth'
    image_dir = '/mnt/data/xiekaikai/split_ss_dota/trainval/images'
    output_dir = '/mnt/data/liurunxiang/workplace/point2rbox-v2-our/work_dirs/dt/1/visual1'
    score_threshold = 0.5
    # ===========================================

    # 屏蔽torch.meshgrid无关警告
    import warnings
    warnings.filterwarnings("ignore", message="torch.meshgrid: in an upcoming release")

    verify_model_predictions_advanced(config_file, checkpoint_file, image_dir, output_dir, score_threshold)