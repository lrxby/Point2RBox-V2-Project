import torch
# 引入这个文件里的类
from mmrotate.models.roi_heads.roi_extractors.rotate_single_level_roi_extractor import RotatedSingleRoIExtractor

def test_the_wheel():
    print("=== 开始测试 RotatedSingleRoIExtractor ===")

    # 1. 准备配置 (Config)
    # 模拟配置文件里的写法
    roi_layer_cfg = dict(
        type='RoIAlignRotated', # 我们要用的底层算子
        out_size=7,             # 输出 7x7
        sample_num=2,
        clockwise=True
    )
    
    # 2. 初始化轮子
    extractor = RotatedSingleRoIExtractor(
        roi_layer=roi_layer_cfg,
        out_channels=4,         # 假设特征图有 4 个通道
        featmap_strides=[8]     # 假设特征图是 8 倍下采样 (P3)
    )
    print("1. 初始化成功！")
    print(f"   底层的算子类型: {type(extractor.roi_layers[0])}") 
    # 这里应该会打印出 mmcv.ops.roi_align_rotated.RoIAlignRotated

    # 3. 准备假数据
    # 特征图: Batch=1, Channel=4, H=100, W=100
    feats = [torch.randn(1, 4, 100, 100).cuda()]
    
    # 旋转框: [batch_ind, cx, cy, w, h, theta]
    # 在 50,50 位置，大小 20x10，旋转 0.5 弧度
    rois = torch.tensor([[0, 50.0, 50.0, 20.0, 10.0, 0.5]]).cuda()
    
    print("2. 准备数据...")
    print(f"   输入特征图形状: {feats[0].shape}")
    print(f"   输入 ROIs: {rois}")

    # 4. 前向传播 (抠图)
    # 注意：这个类不仅仅是抠图，它还负责把 ROI 映射到正确的层级
    # 因为我们只给了一层特征 (stride=8)，它应该直接去这层抠
    roi_feats = extractor(feats, rois)
    
    print("3. 执行抠图...")
    print(f"   输出特征形状: {roi_feats.shape}") # 应该是 [1, 4, 7, 7]
    
    if roi_feats.shape == (1, 4, 7, 7):
        print(">>> 测试完美通过！这就是我们要找的轮子。")
    else:
        print(">>> 测试存疑，形状不对。")

if __name__ == '__main__':
    if torch.cuda.is_available():
        test_the_wheel()
    else:
        print("错误：需要 CUDA 环境")
