import torch
# 这就是老师说的“现成的轮子”，直接从库里拿
from mmcv.ops import RoIAlignRotated

def check_roi_align():
    print("正在检查环境中的 RoIAlignRotated...")
    
    # 1. 准备数据 (Batch=1, Channel=1, 10x10)
    feat = torch.zeros(1, 1, 10, 10).cuda()
    feat[:, :, 4:7, 4:7] = 100  # 在中心放一个 3x3 的高亮区域
    
    # 2. 准备一个旋转框 (BatchID=0, x=5, y=5, w=3, h=3, theta=0)
    # 这个框正好框住那个高亮区域
    # 格式: [batch_ind, x, y, w, h, theta]
    rois = torch.tensor([[0, 5.0, 5.0, 3.0, 3.0, 0.0]]).cuda()
    
    # 3. 初始化轮子
    # out_size=(3, 3) 意思是抠出来的图也是 3x3
    roi_align = RoIAlignRotated(out_size=(3, 3), spatial_scale=1.0)
    
    # 4. 运行
    res = roi_align(feat, rois)
    
    print(f"输入特征图值 (中心): \n{feat[0,0,4:7,4:7]}")
    print(f"ROI Align 抠出来的结果: \n{res[0,0]}")
    
    if res.mean() > 90:
        print("\n>>> 成功！RotatedRoIAlign 工作正常，可以准确抠出特征！")
    else:
        print("\n>>> 警告：抠图结果不对，可能要注意坐标系或参数。")

if __name__ == "__main__":
    try:
        check_roi_align()
    except Exception as e:
        print(f"运行失败: {e}")
        print("可能原因：您的 mmcv 版本可能不支持该算子，或者 import 路径不对。")
