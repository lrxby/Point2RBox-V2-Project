import torch
import torch.nn as nn
import torch.nn.functional as F

class DifferentiableRoIAlignRotated(nn.Module):
    """
    纯 PyTorch 实现的可导旋转 RoI Align。
    支持梯度回传到 ROIs (x, y, w, h, theta)。
    
    [Fix] 增加了分块处理 (Chunking) 机制，防止显存爆炸。
    针对 A5000 (24GB) 优化，CHUNK_SIZE 设为 512。
    """
    def __init__(self, output_size, spatial_scale, sampling_ratio=0, clockwise=True):
        super(DifferentiableRoIAlignRotated, self).__init__()
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
        # 如果没有 ROI，直接返回空 tensor，且要带上梯度信息防止报错
        if num_rois == 0:
            return features.sum() * 0.0 + features.new_zeros(0, features.size(1), self.output_size[0], self.output_size[1])

        batch_size, channels, height, width = features.shape
        out_h, out_w = self.output_size

        # 1. 解析 ROIs
        batch_inds = rois[:, 0].long()
        # 将 ROI 坐标映射到特征图尺度
        rois_feat = rois[:, 1:] * self.spatial_scale
        
        cx, cy, w, h, theta = rois_feat[:, 0], rois_feat[:, 1], rois_feat[:, 2], rois_feat[:, 3], rois_feat[:, 4]

        # 2. 构建采样网格 (Affine Transformation)
        # 预先生成基础网格
        _y, _x = torch.meshgrid(
            torch.linspace(-0.5, 0.5, out_h, dtype=rois.dtype, device=rois.device),
            torch.linspace(-0.5, 0.5, out_w, dtype=rois.dtype, device=rois.device),
            indexing='ij'
        )
        # (1, H*W)
        base_grid_x = _x.reshape(-1).unsqueeze(0) 
        base_grid_y = _y.reshape(-1).unsqueeze(0)

        # 预计算所有 ROI 的变换参数
        if self.clockwise:
            cos_t = torch.cos(theta)
            sin_t = torch.sin(theta)
        else:
            cos_t = torch.cos(-theta)
            sin_t = torch.sin(-theta)

        # 3. 执行采样 (使用分块策略防止 OOM)
        output = torch.zeros(num_rois, channels, out_h, out_w, 
                             dtype=features.dtype, device=features.device)
        
        unique_batches = rois[:, 0].unique().long()
        
        # [A5000 优化] 分块大小 512
        # 显存占用约为 4GB，非常安全且高效
        CHUNK_SIZE = 512

        for b_idx in unique_batches:
            # 找到属于当前图片的所有 ROI 索引
            mask = (batch_inds == b_idx)
            batch_rois_indices = torch.nonzero(mask, as_tuple=True)[0]
            
            # 取出当前图片特征 (1, C, H, W)
            batch_feat = features[b_idx].unsqueeze(0)
            
            # 分块循环处理
            num_batch_rois = batch_rois_indices.shape[0]
            
            for i in range(0, num_batch_rois, CHUNK_SIZE):
                end = min(i + CHUNK_SIZE, num_batch_rois)
                
                # 当前块的 ROI 索引 (相对于全局 rois)
                sub_indices = batch_rois_indices[i:end]
                sub_batch_size = sub_indices.shape[0]
                
                # --- 在块内即时计算 Grid，节省显存 ---
                # 取出这一小批 ROI 的参数
                sub_w = w[sub_indices].unsqueeze(1)
                sub_h = h[sub_indices].unsqueeze(1)
                sub_cx = cx[sub_indices].unsqueeze(1)
                sub_cy = cy[sub_indices].unsqueeze(1)
                sub_cos = cos_t[sub_indices].unsqueeze(1)
                sub_sin = sin_t[sub_indices].unsqueeze(1)

                # 扩展基础网格 (M, H*W)
                sub_grid_x = base_grid_x.expand(sub_batch_size, -1)
                sub_grid_y = base_grid_y.expand(sub_batch_size, -1)

                # 仿射变换
                gx = sub_grid_x * sub_w
                gy = sub_grid_y * sub_h

                x_sample = gx * sub_cos - gy * sub_sin + sub_cx
                y_sample = gx * sub_sin + gy * sub_cos + sub_cy

                # 归一化到 [-1, 1]
                x_grid = 2.0 * x_sample / max(width - 1, 1) - 1.0
                y_grid = 2.0 * y_sample / max(height - 1, 1) - 1.0
                
                # 堆叠并 reshape 为 (M, H_out, W_out, 2)
                sub_grid = torch.stack([x_grid, y_grid], dim=2)
                sub_grid = sub_grid.view(sub_batch_size, out_h, out_w, 2)
                
                # 扩展特征图 (只扩展 chunk 次，显存占用 ~4GB)
                sub_feat_expanded = batch_feat.expand(sub_batch_size, -1, -1, -1)
                
                # 采样
                sub_out = F.grid_sample(sub_feat_expanded, sub_grid, 
                                        mode='bilinear', padding_mode='zeros', align_corners=False)
                
                # 填回输出
                output[sub_indices] = sub_out

        return output