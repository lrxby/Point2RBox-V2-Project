# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmrotate.registry import MODELS

@MODELS.register_module()
class NAOALoss(nn.Module):
    """NAOA角度一致性损失 (Neighbor-Aware Orientation Alignment Loss)
    
    原理：利用“曼哈顿世界假设”和局部熵最小化原理。
    假设空间上相邻的物体（尤其是密集排列的车辆、船舶）通常具有相同或正交的朝向。
    通过计算局部邻域内的向量一致性来监督角度，无需GT框。

    Args:
        loss_weight (float): 损失总权重。默认为 1.0
        k_radius (float): 高斯核半径缩放系数。默认为 2.0
        score_alpha (float): 置信度权重幂次。默认为 1.0
        target_classes (list[int] | None): 指定参与计算的类别索引列表。
        reduction (str): 损失归一化方式。默认为 'mean'
    """

    def __init__(self,
                 loss_weight=1.0,
                 k_radius=2.0,
                 score_alpha=1.0,
                 target_classes=None,
                 reduction='mean'):
        super(NAOALoss, self).__init__()
        self.loss_weight = loss_weight
        self.k_radius = k_radius
        self.score_alpha = score_alpha
        self.target_classes = target_classes
        self.reduction = reduction

    def forward(self,
                pos_bbox_preds,
                pos_scores,
                pos_labels,
                batch_idxs,
                **kwargs):
        """
        Args:
            pos_bbox_preds (Tensor): (N, 5) [x, y, w, h, theta]
            pos_scores (Tensor): (N, )
            pos_labels (Tensor): (N, )
            batch_idxs (Tensor): (N, ) 指示每个样本属于哪张图片
        """
        N = pos_bbox_preds.shape[0]
        # 样本数量不足时跳过计算
        if N < 2: 
            return pos_bbox_preds.sum() * 0.0

        # ================= Step 1: 几何解耦 =================
        centers = pos_bbox_preds[:, :2].detach()
        wh = pos_bbox_preds[:, 2:4].detach()
        # 限制 scale 范围防止数值不稳定
        scales = (wh[:, 0] * wh[:, 1]).sqrt().clamp(min=16.0, max=800.0)
        thetas = pos_bbox_preds[:, 4]

        # ================= Step 2: 矢量化 (4-Theta) =================
        # 使用 4*theta 实现 90度旋转不变性 (即十字路口的车互相认为一致)
        vecs = torch.stack([torch.cos(4 * thetas), torch.sin(4 * thetas)], dim=1)

        # ================= Step 3: 构建亲和矩阵 =================
        # 3.1 空间距离权重 (Gaussian Kernel)
        dist_sq = torch.cdist(centers, centers, p=2).pow(2)
        sigmas = scales * self.k_radius
        sigma_mat = sigmas.view(N, 1)
        # 距离越近，权重越大
        W_geo = torch.exp(-dist_sq / (2 * sigma_mat.pow(2))).detach()

        # 3.2 置信度加权 (让高置信度样本主导)
        scores_detached = pos_scores.detach().pow(self.score_alpha)
        W_conf = scores_detached.view(1, N)

        # 3.3 逻辑掩码 (同类、同图才算邻居，允许自环)
        mask_cls = (pos_labels.view(N, 1) == pos_labels.view(1, N)).float()
        mask_batch = (batch_idxs.view(N, 1) == batch_idxs.view(1, N)).float()
        
        W = W_geo * W_conf * mask_cls * mask_batch

        # ================= Step 4: 归一化 =================
        # 包含自环，分母恒大于0
        W_sum = W.sum(dim=1, keepdim=True)
        W_norm = W / W_sum

        # ================= Step 5: 能量/混乱度计算 =================
        # 计算加权平均向量
        mean_vecs = torch.mm(W_norm, vecs)
        
        # 模长越大说明越一致。Loss = 1 - 模长
        # 孤立点: chaos=0 (因为只有自环，模长为1) -> Loss=0 (自动屏蔽)
        # 混乱邻域: chaos -> 1
        chaos_score = 1.0 - mean_vecs.norm(dim=1)

        # ================= Step 6: 类别筛选与最终 Loss =================
        if self.target_classes is not None:
            class_mask = torch.zeros_like(pos_labels, dtype=torch.bool)
            for t_cls in self.target_classes:
                class_mask = class_mask | (pos_labels == t_cls)
            class_mask = class_mask.float()
        else:
            class_mask = torch.ones_like(pos_labels, dtype=torch.float)

        final_loss_per_item = chaos_score * class_mask

        if self.reduction == 'mean':
            num_valid = class_mask.sum()
            if num_valid > 0:
                return self.loss_weight * final_loss_per_item.sum() / num_valid
            else:
                return final_loss_per_item.sum() # 0.0
        else:
            return self.loss_weight * final_loss_per_item.sum()

@MODELS.register_module()
class PerspectiveAwareSizeConsistencyLoss(nn.Module):
    """透视感知尺寸一致性损失 (Image-wise V2 - Strict Mode)。

    原理：利用“透视投影”原理。
    在同一张航拍图中，同一类物体（如小型车辆）的物理尺寸是固定的。
    图像上的尺寸变化主要由相机距离（透视）决定。
    本 Loss 试图拟合平面方程：log(Scale) = wx * x + wy * y + b_class
    从而约束物体尺寸不至于毫无规律地发散。

    Args:
        loss_weight (float): 损失权重。
        ridge_lambda (float): 岭回归正则项。
        norm_type (str): 'z-score' 或 'image-norm'。推荐 'image-norm'。
        target_classes (list[int]): 建议只对刚性物体开启。
    """

    def __init__(self,
                 loss_weight=1.0,
                 ridge_lambda=1e-4,
                 beta=1.0,
                 norm_type='z-score',
                 target_classes=None):
        super(PerspectiveAwareSizeConsistencyLoss, self).__init__()
        self.loss_weight = loss_weight
        self.ridge_lambda = ridge_lambda
        self.beta = beta
        self.norm_type = norm_type
        self.target_classes = target_classes
        self.smooth_l1 = nn.SmoothL1Loss(reduction='none', beta=beta)

    def _solve_single_image(self, pred_bboxes, scores, labels, img_shape=None):
        """针对单张图片的求解逻辑 (加权岭回归)。"""
        # 1. 准备数据
        if pred_bboxes.shape[0] == 0:
            return pred_bboxes.new_tensor(0.0)

        x_c = pred_bboxes[:, 0]
        y_c = pred_bboxes[:, 1]
        w = pred_bboxes[:, 2].clamp(min=1e-2)
        h = pred_bboxes[:, 3].clamp(min=1e-2)

        # 目标向量 Y：对数空间下的 log(sqrt(area))
        s_log = 0.5 * torch.log(w * h)
        Y = s_log.unsqueeze(1)

        # 权重 W (基于预测置信度)
        weights = scores.clamp(min=1e-6)
        sqrt_w = torch.sqrt(weights).unsqueeze(1)

        # 2. 检查约束: 样本数必须 >= 未知数数量 (2个斜率 + K个截距)
        unique_labels, labels_inv = torch.unique(labels, return_inverse=True)
        K = len(unique_labels)
        N = len(pred_bboxes)

        if N < K + 3:
            return pred_bboxes.new_tensor(0.0)

        # 3. 归一化坐标 (Image-wise)
        if self.norm_type == 'z-score':
            x_mean, x_std = x_c.mean().detach(), x_c.std().detach().clamp(min=1e-6)
            y_mean, y_std = y_c.mean().detach(), y_c.std().detach().clamp(min=1e-6)
            x_norm = (x_c - x_mean) / x_std
            y_norm = (y_c - y_mean) / y_std
            
        elif self.norm_type == 'image-norm':
            if img_shape is None:
                # Fallback if no meta info
                x_norm, y_norm = x_c, y_c
            else:
                H_img, W_img = img_shape[:2]
                x_norm = (x_c - W_img / 2.0) / (W_img / 2.0)
                y_norm = (y_c - H_img / 2.0) / (H_img / 2.0)
        else:
            x_norm, y_norm = x_c, y_c

        # 4. 构建设计矩阵 A [x, y, one-hot_class]
        A = pred_bboxes.new_zeros((N, 2 + K))
        A[:, 0] = x_norm
        A[:, 1] = y_norm
        A.scatter_(1, 2 + labels_inv.unsqueeze(1), 1.0)

        # 5. 加权岭回归求解 (Closed-form solution)
        # theta = (A^T * W * A + lambda * I)^-1 * A^T * W * Y
        A_w = A * sqrt_w
        Y_w = Y * sqrt_w

        M = torch.matmul(A_w.t(), A_w)
        I_reg = torch.eye(2 + K, device=pred_bboxes.device) * self.ridge_lambda
        M_reg = M + I_reg

        try:
            RHS = torch.matmul(A_w.t(), Y_w)
            theta = torch.linalg.solve(M_reg, RHS)
        except RuntimeError:
            # 矩阵不可逆等极端情况
            return pred_bboxes.new_tensor(0.0)

        # 6. 计算拟合误差作为 Loss
        Y_hat = torch.matmul(A, theta).squeeze(1)
        diff = s_log - Y_hat # 实际尺寸 - 理论尺寸
        
        loss_element = self.smooth_l1(diff, torch.zeros_like(diff))
        loss_weighted = torch.sum(loss_element * weights) / (torch.sum(weights) + 1e-6)

        return loss_weighted

    def forward(self, pred_bboxes, scores, labels, batch_idxs=None, img_metas=None):
        """
        Args:
            pred_bboxes: (N, 5)
            scores: (N, )
            labels: (N, )
            batch_idxs: (N, ) 必须提供，用于在每张图片内部独立计算
            img_metas: list[dict]
        """
        # 0. 过滤目标类别
        if self.target_classes is not None:
            mask = torch.zeros_like(labels, dtype=torch.bool)
            for cls_id in self.target_classes:
                mask |= (labels == cls_id)
            
            if mask.sum() == 0:
                return pred_bboxes.new_tensor(0.0)

            pred_bboxes = pred_bboxes[mask]
            scores = scores[mask]
            labels = labels[mask]
            if batch_idxs is not None:
                batch_idxs = batch_idxs[mask]

        if pred_bboxes.shape[-1] < 4:
            return pred_bboxes.new_tensor(0.0)
        
        if batch_idxs is None:
            raise ValueError("PerspectiveAwareSizeConsistencyLoss requires 'batch_idxs'.")

        # 3. 逐图计算 Loss (Image-wise Loop)
        # 透视关系只存在于单张 2D 图像内部，不能跨图拟合
        unique_batches = torch.unique(batch_idxs)
        total_loss = pred_bboxes.new_tensor(0.0)
        valid_batches = 0.0

        for b_idx in unique_batches:
            mask = (batch_idxs == b_idx)
            b_pred_bboxes = pred_bboxes[mask]
            b_scores = scores[mask]
            b_labels = labels[mask]
            
            b_img_shape = None
            if img_metas is not None and len(img_metas) > int(b_idx.item()):
                b_img_shape = img_metas[int(b_idx.item())]['img_shape']
            
            loss_per_img = self._solve_single_image(b_pred_bboxes, b_scores, b_labels, img_shape=b_img_shape)
            
            if loss_per_img > 0:
                total_loss += loss_per_img
                valid_batches += 1.0

        if valid_batches > 0:
            return self.loss_weight * (total_loss / valid_batches)
        else:
            return pred_bboxes.new_tensor(0.0)