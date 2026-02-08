# mmrotate/models/utils/differentiable_roi_align.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class DifferentiableRoIAlignRotated(nn.Module):
    """
    纯 PyTorch 实现的可导旋转 RoI Align (Differentiable for ROIs)。
    完全对齐 mmcv.ops.RoIAlignRotated (aligned=True) 的采样逻辑。
    
    【优化版】：增加了 Chunking 机制，防止在 ROI 数量过多时 grid_sample 反向传播爆显存。
    """
    def __init__(self, output_size, spatial_scale, sampling_ratio=0, clockwise=True):
        """
        Args:
            output_size (tuple): (h, w)
            spatial_scale (float): scale the input boxes by this number
            sampling_ratio (int): 默认为0。暂不支持 >0 的过采样。
            clockwise (bool): 角度方向。MMRotate 默认为 True (顺时针)。
        """
        super(DifferentiableRoIAlignRotated, self).__init__()
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            self.output_size = output_size
            
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio
        self.clockwise = clockwise

    def forward(self, features, rois):
        """
        Args:
            features: (N, C, H, W)
            rois: (K, 6) [batch_ind, x, y, w, h, theta]
        Returns:
            output: (K, C, output_h, output_w)
        """
        num_rois = rois.shape[0]
        batch_size, channels, height, width = features.shape
        out_h, out_w = self.output_size

        # 1. 解析 ROIs
        batch_inds = rois[:, 0].long()
        # 将 ROI 坐标映射到特征图尺度
        rois_feat = rois[:, 1:] * self.spatial_scale
        cx, cy, w, h, theta = rois_feat[:, 0], rois_feat[:, 1], rois_feat[:, 2], rois_feat[:, 3], rois_feat[:, 4]

        # 2. 构建采样网格 (Affine Transformation)
        # 使用 Bin Center 逻辑
        _y = (torch.arange(out_h, dtype=rois.dtype, device=rois.device) + 0.5) / out_h - 0.5
        _x = (torch.arange(out_w, dtype=rois.dtype, device=rois.device) + 0.5) / out_w - 0.5
        
        # (1, H, W) -> (K, H*W)
        _y, _x = torch.meshgrid(_y, _x, indexing='ij')
        grid_x = _x.reshape(-1).unsqueeze(0).expand(num_rois, -1)
        grid_y = _y.reshape(-1).unsqueeze(0).expand(num_rois, -1)

        # 2.2 旋转矩阵构建
        if self.clockwise:
            cos_t = torch.cos(theta)
            sin_t = torch.sin(theta)
        else:
            cos_t = torch.cos(-theta)
            sin_t = torch.sin(-theta)

        # 2.3 应用仿射变换
        gx = grid_x * w.unsqueeze(1) 
        gy = grid_y * h.unsqueeze(1)

        x_sample = gx * cos_t.unsqueeze(1) - gy * sin_t.unsqueeze(1) + cx.unsqueeze(1)
        y_sample = gx * sin_t.unsqueeze(1) + gy * cos_t.unsqueeze(1) + cy.unsqueeze(1)

        # 3. 归一化到 [-1, 1] (align_corners=False)
        x_grid = 2.0 * x_sample / width - 1.0
        y_grid = 2.0 * y_sample / height - 1.0
        
        # 堆叠为 (K, out_h, out_w, 2)
        grid = torch.stack([x_grid, y_grid], dim=2).view(num_rois, out_h, out_w, 2)

        # 4. 执行采样 (Batch Loop + Chunking)
        output = torch.zeros(num_rois, channels, out_h, out_w, 
                             dtype=features.dtype, device=features.device)
        
        unique_batches = batch_inds.unique()
        
        # 【修改点】：针对 A5000 (24GB) 调大了 Chunk Size
        # 32 -> 256
        # 每次反向传播约占用 2GB 显存，非常安全且高效
        ROI_CHUNK_SIZE = 256 
        
        for b_idx in unique_batches:
            # 找到属于当前图片的所有 ROI 的全局索引
            mask = (batch_inds == b_idx)
            # 获取这些 ROI 在 output 中的位置索引
            batch_roi_indices = torch.nonzero(mask).squeeze(1)
            
            batch_grid = grid[mask] # (M, H, W, 2)
            batch_feat = features[b_idx:b_idx+1] # (1, C, H, W)
            
            M = batch_grid.shape[0]
            
            # 对该图片内的 ROI 进行分块处理
            for start in range(0, M, ROI_CHUNK_SIZE):
                end = min(start + ROI_CHUNK_SIZE, M)
                
                # 切片：取出一小部分 Grid
                sub_grid = batch_grid[start:end] # (sub_M, H, W, 2)
                sub_M = sub_grid.shape[0]
                
                # 扩展特征图：只扩展 sub_M 次，而非 M 次
                sub_feat_expanded = batch_feat.expand(sub_M, -1, -1, -1)
                
                # 采样
                sub_out = F.grid_sample(
                    sub_feat_expanded, 
                    sub_grid, 
                    mode='bilinear', 
                    padding_mode='zeros', 
                    align_corners=False
                )
                
                # 填回 Output
                # 对应的全局索引是 batch_roi_indices[start:end]
                output[batch_roi_indices[start:end]] = sub_out

        return output