# # Copyright (c) OpenMMLab. All rights reserved.
# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import cv2
# import numpy as np
# from mmdet.models.losses.utils import weighted_loss

# from mmrotate.registry import MODELS
# from mmrotate.models.losses.gaussian_dist_loss import postprocess


# @weighted_loss
# def gwd_sigma_loss(pred, target, fun='log1p', tau=1.0, alpha=1.0, normalize=True):
#     """Gaussian Wasserstein distance loss.
#     Modified from gwd_loss. 
#     gwd_sigma_loss only involves sigma in Gaussian, with mu ignored.

#     Args:
#         pred (torch.Tensor): Predicted bboxes.
#         target (torch.Tensor): Corresponding gt bboxes.
#         fun (str): The function applied to distance. Defaults to 'log1p'.
#         tau (float): Defaults to 1.0.
#         alpha (float): Defaults to 1.0.
#         normalize (bool): Whether to normalize the distance. Defaults to True.

#     Returns:
#         loss (torch.Tensor)

#     """
#     Sigma_p = pred
#     Sigma_t = target

#     whr_distance = Sigma_p.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
#     whr_distance = whr_distance + Sigma_t.diagonal(
#         dim1=-2, dim2=-1).sum(dim=-1)

#     _t_tr = (Sigma_p.bmm(Sigma_t)).diagonal(dim1=-2, dim2=-1).sum(dim=-1)
#     _t_det_sqrt = (Sigma_p.det() * Sigma_t.det()).clamp(1e-7).sqrt()
#     whr_distance = whr_distance + (-2) * (
#         (_t_tr + 2 * _t_det_sqrt).clamp(1e-7).sqrt())

#     distance = (alpha * alpha * whr_distance).clamp(1e-7).sqrt()

#     if normalize:
#         scale = 2 * (
#             _t_det_sqrt.clamp(1e-7).sqrt().clamp(1e-7).sqrt()).clamp(1e-7)
#         distance = distance / scale

#     return postprocess(distance, fun=fun, tau=tau)


# def bhattacharyya_coefficient(pred, target):
#     """Calculate bhattacharyya coefficient between 2-D Gaussian distributions.

#     Args:
#         pred (Tuple): tuple of (xy, sigma).
#             xy (torch.Tensor): center point of 2-D Gaussian distribution
#                 with shape (N, 2).
#             sigma (torch.Tensor): covariance matrix of 2-D Gaussian distribution
#                 with shape (N, 2, 2).
#         target (Tuple): tuple of (xy, sigma).

#     Returns:
#         coef (Tensor): bhattacharyya coefficient with shape (N,).
#     """
#     xy_p, Sigma_p = pred
#     xy_t, Sigma_t = target

#     _shape = xy_p.shape

#     xy_p = xy_p.reshape(-1, 2)
#     xy_t = xy_t.reshape(-1, 2)
#     Sigma_p = Sigma_p.reshape(-1, 2, 2)
#     Sigma_t = Sigma_t.reshape(-1, 2, 2)

#     Sigma_M = (Sigma_p + Sigma_t) / 2
#     dxy = (xy_p - xy_t).unsqueeze(-1)
#     t0 = torch.exp(-0.125 * dxy.permute(0, 2, 1).bmm(torch.linalg.solve(Sigma_M, dxy)))
#     t1 = (Sigma_p.det() * Sigma_t.det()).clamp(1e-7).sqrt()
#     t2 = Sigma_M.det()

#     coef = t0 * (t1 / t2).clamp(1e-7).sqrt()[..., None, None]
#     coef = coef.reshape(_shape[:-1])
#     return coef


# @weighted_loss
# def gaussian_overlap_loss(pred, target, alpha=0.01, beta=0.6065):
#     """Calculate Gaussian overlap loss based on bhattacharyya coefficient.

#     Args:
#         pred (Tuple): tuple of (xy, sigma).
#             xy (torch.Tensor): center point of 2-D Gaussian distribution
#                 with shape (N, 2).
#             sigma (torch.Tensor): covariance matrix of 2-D Gaussian distribution
#                 with shape (N, 2, 2).

#     Returns:
#         loss (Tensor): overlap loss with shape (N, N).
#     """
#     mu, sigma = pred
#     B = mu.shape[0]
#     mu0 = mu[None].expand(B, B, 2)
#     sigma0 = sigma[None].expand(B, B, 2, 2)
#     mu1 = mu[:, None].expand(B, B, 2)
#     sigma1 = sigma[:, None].expand(B, B, 2, 2)
#     loss = bhattacharyya_coefficient((mu0, sigma0), (mu1, sigma1))
#     loss[torch.eye(B, dtype=bool)] = 0
#     loss = F.leaky_relu(loss - beta, negative_slope=alpha) + beta * alpha
#     loss = loss.sum(-1)
#     return loss


# @MODELS.register_module()
# class GaussianOverlapLoss(nn.Module):
#     """Gaussian Overlap Loss.

#     Args:
#         reduction (str, optional): The method used to reduce the loss into
#             a scalar. Defaults to 'mean'. Options are "none", "mean" and
#             "sum".
#         loss_weight (float, optional): Weight of loss. Defaults to 1.0.

#     Returns:
#         loss (torch.Tensor)
#     """

#     def __init__(self,
#                  reduction='mean',
#                  loss_weight=1.0,
#                  lamb=1e-4):
#         super(GaussianOverlapLoss, self).__init__()
#         self.reduction = reduction
#         self.loss_weight = loss_weight
#         self.lamb = lamb

#     def forward(self,
#                 pred,
#                 weight=None,
#                 avg_factor=None,
#                 reduction_override=None):
#         """Forward function.

#         Args:
#             pred (Tuple): tuple of (xy, sigma).
#                 xy (torch.Tensor): center point of 2-D Gaussian distribution
#                     with shape (N, 2).
#                 sigma (torch.Tensor): covariance matrix of 2-D Gaussian distribution
#                     with shape (N, 2, 2).
#             weight (torch.Tensor, optional): The weight of loss for each
#                 prediction. Defaults to None.
#             avg_factor (int, optional): Average factor that is used to average
#                 the loss. Defaults to None.
#             reduction_override (str, optional): The reduction method used to
#                 override the original reduction method of the loss.
#                 Options are "none", "mean" and "sum".

#         Returns:
#             torch.Tensor: The calculated loss
#         """
#         assert reduction_override in (None, 'none', 'mean', 'sum')
#         reduction = (
#             reduction_override if reduction_override else self.reduction)
#         assert len(pred[0]) == len(pred[1])

#         sigma = pred[1]
#         L = torch.linalg.eigh(sigma)[0].clamp(1e-7).sqrt()
#         loss_lamb = F.l1_loss(L, torch.zeros_like(L), reduction='none')
#         loss_lamb = self.lamb * loss_lamb.log1p().mean()
        
#         return self.loss_weight * (loss_lamb + gaussian_overlap_loss(
#             pred,
#             None,
#             weight,
#             reduction=reduction,
#             avg_factor=avg_factor))


# def plot_gaussian_voronoi_watershed(*images):
#     """Plot figures for debug."""
#     import matplotlib.pyplot as plt
#     plt.figure(dpi=300, figsize=(len(images) * 4, 4))
#     plt.tight_layout()
#     fileid = np.random.randint(0, 20)
#     for i in range(len(images)):
#         img = images[i]
#         img = (img - img.min()) / (img.max() - img.min())
#         if img.dim() == 3:
#             img = img.permute(1, 2, 0)
#         img = img.detach().cpu().numpy()
#         plt.subplot(1, len(images), i + 1)
#         if i == 3:
#             plt.imshow(img)
#             x = np.linspace(0, 1024, 1024)
#             y = np.linspace(0, 1024, 1024)
#             X, Y = np.meshgrid(x, y)
#             plt.contourf(X, Y, img, levels=8, cmap=plt.get_cmap('magma'))
#         else:
#             plt.imshow(img)
#         plt.xticks([])
#         plt.yticks([])
#     plt.savefig(f'debug/Gaussian-Voronoi-{fileid}.png')
#     plt.close()


# def gaussian_2d(xy, mu, sigma, normalize=False):
#     dxy = (xy - mu).unsqueeze(-1)
#     t0 = torch.exp(-0.5 * dxy.permute(0, 2, 1).bmm(torch.linalg.solve(sigma, dxy)))
#     if normalize:
#         t0 = t0 / (2 * np.pi * sigma.det().clamp(1e-7).sqrt())
#     return t0


# def gaussian_voronoi_watershed_loss(mu, sigma,
#                                     label, image, 
#                                     pos_thres, neg_thres, 
#                                     down_sample=2, topk=0.95, 
#                                     default_sigma=4096,
#                                     voronoi='gaussian-orientation',
#                                     alpha=0.1,
#                                     debug=False):
#     J = len(sigma)
#     if J == 0:
#         return sigma.sum()
    
#     D = down_sample
#     H, W = image.shape[-2:]
#     h, w = H // D, W // D
#     x = torch.linspace(0, h, h, device=mu.device)
#     y = torch.linspace(0, w, w, device=mu.device)
#     xy = torch.stack(torch.meshgrid(x, y, indexing='xy'), -1)
#     vor = mu.new_zeros(J, h, w)
#     # Get distribution for each instance
#     mm = (mu.detach() / D).round()
#     if voronoi == 'standard':
#         sg = sigma.new_tensor((default_sigma, 0, 0, default_sigma)).reshape(2, 2)
#         sg = sg / D ** 2
#         for j, m in enumerate(mm):
#             vor[j] = gaussian_2d(xy.view(-1, 2), m[None], sg[None]).view(h, w)
#     elif voronoi == 'gaussian-orientation':
#         L, V = torch.linalg.eigh(sigma)
#         L = L.detach().clone()
#         L = L / (L[:, 0:1] * L[:, 1:2]).sqrt() * default_sigma
#         sg = V.matmul(torch.diag_embed(L)).matmul(V.permute(0, 2, 1)).detach()
#         sg = sg / D ** 2
#         for j, (m, s) in enumerate(zip(mm, sg)):
#             vor[j] = gaussian_2d(xy.view(-1, 2), m[None], s[None]).view(h, w)
#     elif voronoi == 'gaussian-full':
#         sg = sigma.detach() / D ** 2
#         for j, (m, s) in enumerate(zip(mm, sg)):
#             vor[j] = gaussian_2d(xy.view(-1, 2), m[None], s[None]).view(h, w)
#     # val: max prob, vor: belong to which instance, cls: belong to which class
#     val, vor = torch.max(vor, 0)
#     if D > 1:
#         vor = vor[:, None, :, None].expand(-1, D, -1, D).reshape(H, W)
#         val = F.interpolate(
#             val[None, None], (H, W), mode='bilinear', align_corners=True)[0, 0]
#     cls = label[vor]
#     kernel = val.new_ones((1, 1, 3, 3))
#     kernel[0, 0, 1, 1] = -8
#     ridges = torch.conv2d(vor[None].float(), kernel, padding=1)[0] != 0
#     vor += 1
#     pos_thres = val.new_tensor(pos_thres)
#     neg_thres = val.new_tensor(neg_thres)
#     vor[val < pos_thres[cls]] = 0
#     vor[val < neg_thres[cls]] = J + 1
#     vor[ridges] = J + 1

#     cls_bg = torch.where(vor == J + 1, 15, cls)
#     cls_bg = torch.where(vor == 0, -1, cls_bg)

#     # PyTorch does not support watershed, use cv2
#     img_uint8 = (image - image.min()) / (image.max() - image.min()) * 255
#     img_uint8 = img_uint8.permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)
#     img_uint8 = cv2.medianBlur(img_uint8, 3)
#     markers = vor.detach().cpu().numpy().astype(np.int32)
#     markers = vor.new_tensor(cv2.watershed(img_uint8, markers))

#     if debug:
#         plot_gaussian_voronoi_watershed(image, cls_bg, markers)

#     L, V = torch.linalg.eigh(sigma)
#     L_target = []
#     for j in range(J):
#         xy = (markers == j + 1).nonzero()[:, (1, 0)].float()
#         if len(xy) == 0:
#             L_target.append(L[j].detach())
#             continue
#         xy = xy - mu[j]
#         xy = V[j].T.matmul(xy[:, :, None])[:, :, 0]
#         max_x = torch.max(torch.abs(xy[:, 0]))
#         max_y = torch.max(torch.abs(xy[:, 1]))
#         L_target.append(torch.stack((max_x, max_y)) ** 2)
#     L_target = torch.stack(L_target)
#     L = torch.diag_embed(L)
#     L_target = torch.diag_embed(L_target)
#     loss = gwd_sigma_loss(L, L_target.detach(), reduction='none')
#     loss = torch.topk(loss, int(np.ceil(len(loss) * topk)), largest=False)[0].mean()
#     return loss, (vor, markers)


# @MODELS.register_module()
# class VoronoiWatershedLoss(nn.Module):
#     """Gaussian Overlap Loss.

#     Args:
#         reduction (str, optional): The method used to reduce the loss into
#             a scalar. Defaults to 'mean'. Options are "none", "mean" and
#             "sum".
#         loss_weight (float, optional): Weight of loss. Defaults to 1.0.

#     Returns:
#         loss (torch.Tensor)
#     """

#     def __init__(self,
#                  down_sample=2,
#                  reduction='mean',
#                  loss_weight=1.0,
#                  topk=0.95,
#                  alpha=0.1,
#                  debug=False):
#         super(VoronoiWatershedLoss, self).__init__()
#         self.down_sample = down_sample
#         self.reduction = reduction
#         self.loss_weight = loss_weight
#         self.topk = topk
#         self.alpha = alpha
#         self.debug = debug

#     def forward(self, pred, label, image, pos_thres, neg_thres, voronoi='orientation'):
#         """Forward function.

#         Args:
#             pred (Tuple): Tuple of (xy, sigma).
#                 xy (torch.Tensor): Center point of 2-D Gaussian distribution
#                     with shape (N, 2).
#                 sigma (torch.Tensor): Covariance matrix of 2-D Gaussian distribution
#                     with shape (N, 2, 2).
#             image (torch.Tensor): The image for watershed with shape (3, H, W).
#             standard_voronoi (bool, optional): Use standard or Gaussian voronoi.

#         Returns:
#             torch.Tensor: The calculated loss
#         """
#         loss, self.vis = gaussian_voronoi_watershed_loss(*pred, 
#                                                          label,
#                                                          image, 
#                                                          pos_thres, 
#                                                          neg_thres, 
#                                                          self.down_sample, 
#                                                          topk=self.topk,
#                                                          voronoi=voronoi,
#                                                          alpha=self.alpha,
#                                                          debug=self.debug)
#         return self.loss_weight * loss


# def rbbox2roi(bbox_list):
#     """Convert a list of bboxes to roi format.

#     Args:
#         bbox_list (list[Tensor]): a list of bboxes corresponding to a batch
#             of images.

#     Returns:
#         Tensor: shape (n, 6), [batch_ind, cx, cy, w, h, a]
#     """
#     rois_list = []
#     for img_id, bboxes in enumerate(bbox_list):
#         if bboxes.size(0) > 0:
#             img_inds = bboxes.new_full((bboxes.size(0), 1), img_id)
#             rois = torch.cat([img_inds, bboxes[:, :5]], dim=-1)
#         else:
#             rois = bboxes.new_zeros((0, 6))
#         rois_list.append(rois)
#     rois = torch.cat(rois_list, 0)
#     return rois


# def plot_edge_map(feat, edgex, edgey):
#     """Plot figures for debug."""
#     import matplotlib.pyplot as plt
#     plt.figure(dpi=300, figsize=(4, 4))
#     plt.tight_layout()
#     fileid = np.random.randint(0, 20)
#     for i in range(len(feat)):
#         img0 = feat[i, :3]
#         img0 = (img0 - img0.min()) / (img0.max() - img0.min())
#         img1 = edgex[i, :3]
#         img1 = (img1 - img1.min()) / (img1.max() - img1.min())
#         img2 = edgey[i, :3]
#         img2 = (img2 - img2.min()) / (img2.max() - img2.min())
#         img3 = img1 + img2
#         img3 = (img3 - img3.min()) / (img3.max() - img3.min())
#         img = torch.cat((torch.cat((img0, img2), -1), 
#                          torch.cat((img1, img3), -1)), -2
#                          ).permute(1, 2, 0).detach().cpu().numpy()
#         N = int(np.ceil(np.sqrt(len(feat))))
#         plt.subplot(N, N, i + 1)
#         plt.imshow(img)
#         plt.xticks([])
#         plt.yticks([])
#     plt.savefig(f'debug/Edge-Map-{fileid}.png')
#     plt.close()


# @MODELS.register_module()
# class EdgeLoss(nn.Module):
#     """Edge Loss.

#     Args:
#         reduction (str, optional): The method used to reduce the loss into
#             a scalar. Defaults to 'mean'. Options are "none", "mean" and
#             "sum".
#         loss_weight (float, optional): Weight of loss. Defaults to 1.0.

#     Returns:
#         loss (torch.Tensor)
#     """

#     def __init__(self,
#                  resolution=24,
#                  max_scale=1.6,
#                  sigma=6,
#                  reduction='mean',
#                  loss_weight=1.0,
#                  debug=False):
#         super(EdgeLoss, self).__init__()
#         self.resolution = resolution
#         self.max_scale = max_scale
#         self.sigma = sigma
#         self.reduction = reduction
#         self.loss_weight = loss_weight
#         self.center_idx = self.resolution / self.max_scale
#         self.debug = debug

#         self.roi_extractor = MODELS.build(dict(
#             type='RotatedSingleRoIExtractor',
#                 roi_layer=dict(
#                     type='RoIAlignRotated',
#                     out_size=(2 * self.resolution + 1),
#                     sample_num=2,
#                     clockwise=True),
#             out_channels=1,
#             featmap_strides=[1],
#             finest_scale=1024))

#         edge_idx = torch.arange(0, self.resolution + 1)
#         edge_distribution = torch.exp(-((edge_idx - self.center_idx) ** 2) / (2 * self.sigma ** 2))
#         edge_distribution[0] = edge_distribution[-1] = 0
#         self.register_buffer('edge_idx', edge_idx)
#         self.register_buffer('edge_distribution', edge_distribution)

#     def forward(self, pred, edge):
#         """Forward function.

#         Args:
#             pred (Tuple): Batched predicted rboxes
#             edge (torch.Tensor): The edge map with shape (B, 1, H, W).

#         Returns:
#             torch.Tensor: The calculated loss
#         """
#         G = self.resolution
#         C = self.center_idx
#         roi = rbbox2roi(pred)
#         roi[:, 3:5] *= self.max_scale
#         feat = self.roi_extractor([edge], roi)
#         if len(feat) == 0:
#             return pred[0].new_tensor(0)
#         featx = feat.sum(1).abs().sum(1)
#         featy = feat.sum(1).abs().sum(2)
#         featx2 = torch.flip(featx[:, :G + 1], (-1,)) + featx[:, G:]
#         featy2 = torch.flip(featy[:, :G + 1], (-1,)) + featy[:, G:]  # (N, 25)
#         ex = ((featx2 * self.edge_distribution).softmax(1) * self.edge_idx).sum(1) / C
#         ey = ((featy2 * self.edge_distribution).softmax(1) * self.edge_idx).sum(1) / C
#         exy = torch.stack((ex, ey), -1)
#         rbbox_concat = torch.cat(pred, 0)
        
#         if self.debug:
#             edgex = featx[:, None, None, :].expand(-1, 1, 2 * self.resolution + 1, -1)
#             edgey = featy[:, None, :, None].expand(-1, 1, -1, 2 * self.resolution + 1)
#             plot_edge_map(feat, edgex, edgey)

#         return self.loss_weight * F.smooth_l1_loss(rbbox_concat[:, 2:4], 
#                                                    (rbbox_concat[:, 2:4] * exy).detach(),
#                                                    beta=8)


# @MODELS.register_module()
# class Point2RBoxV2ConsistencyLoss(nn.Module):
#     """Consistency Loss.

#     Args:
#         reduction (str, optional): The method used to reduce the loss into
#             a scalar. Defaults to 'mean'. Options are "none", "mean" and
#             "sum".
#         loss_weight (float, optional): Weight of loss. Defaults to 1.0.

#     Returns:
#         loss (torch.Tensor)
#     """

#     def __init__(self,
#                  reduction='mean',
#                  loss_weight=1.0):
#         super(Point2RBoxV2ConsistencyLoss, self).__init__()
#         self.reduction = reduction
#         self.loss_weight = loss_weight

#     def forward(self, ori_pred, trs_pred, square_mask, aug_type, aug_val):
#         """Forward function.

#         Args:
#             ori_pred (Tuple): (Sigma, theta)
#             trs_pred (Tuple): (Sigma, theta)
#             square_mask: When True, the angle is ignored
#             aug_type: 'rot', 'flp', 'sca'
#             aug_val: Rotation or scale value

#         Returns:
#             torch.Tensor: The calculated loss
#         """
#         ori_gaus, ori_angle = ori_pred
#         trs_gaus, trs_angle = trs_pred

#         if aug_type == 'rot':
#             rot = ori_gaus.new_tensor(aug_val)
#             cos_r = torch.cos(rot)
#             sin_r = torch.sin(rot)
#             R = torch.stack((cos_r, -sin_r, sin_r, cos_r), dim=-1).reshape(-1, 2, 2)
#             ori_gaus = R.matmul(ori_gaus).matmul(R.permute(0, 2, 1))
#             d_ang = trs_angle - ori_angle - aug_val
#         elif aug_type == 'flp':
#             ori_gaus = ori_gaus * ori_gaus.new_tensor((1, -1, -1, 1)).reshape(2, 2)
#             d_ang = trs_angle + ori_angle
#         else:
#             sca = ori_gaus.new_tensor(aug_val)
#             ori_gaus = ori_gaus * sca
#             d_ang = trs_angle - ori_angle
        
#         loss_ssg = gwd_sigma_loss(ori_gaus.bmm(ori_gaus), trs_gaus.bmm(trs_gaus))
#         d_ang = (d_ang + math.pi / 2) % math.pi - math.pi / 2
#         loss_ssa = F.smooth_l1_loss(d_ang, torch.zeros_like(d_ang), reduction='none', beta=0.1)
#         loss_ssa = loss_ssa[~square_mask].sum() / max(1, (~square_mask).sum())

#         return self.loss_weight * (loss_ssg + loss_ssa)


# # @MODELS.register_module()
# # class PerspectiveAwareSizeConsistencyLoss(nn.Module):
# #     """透视感知尺寸一致性损失。

# #     实现了对数空间下透视投影的矩阵形式。
# #     它使用可微分岭回归联合拟合全局透视平面（共享斜率）和特定类别的截距。

# #     公式：log(s) = wx * x + wy * y + b_class

# #     参数：
# #         loss_weight (float): 损失权重。默认为 1.0。
# #         ridge_lambda (float): 岭回归的正则化强度。默认为 1e-4。
# #         beta (float): SmoothL1Loss 的 beta 参数。默认为 1.0。
# #         norm_type (str): 坐标归一化类型。默认为 'z-score'。
# #         target_classes (list[int] | None): 应用损失的类别索引列表。
# #             如果为 None，则应用于所有类别。
# #     """

# #     def __init__(self,
# #                  loss_weight=1.0,
# #                  ridge_lambda=1e-4,
# #                  beta=1.0,
# #                  norm_type='z-score',
# #                  target_classes=None):
# #         super(PerspectiveAwareSizeConsistencyLoss, self).__init__()
# #         self.loss_weight = loss_weight
# #         self.ridge_lambda = ridge_lambda
# #         self.beta = beta
# #         self.norm_type = norm_type
# #         self.target_classes = target_classes
# #         self.smooth_l1 = nn.SmoothL1Loss(reduction='none', beta=beta)

# #     def forward(self, pred_bboxes, scores, labels):
# #         """
# #         参数：
# #             pred_bboxes (Tensor): 形状 (N, 4) 或 (N, 5)，[x, y, w, h, ...]
# #             scores (Tensor): 形状 (N, )，置信度分数，用作权重。
# #             labels (Tensor): 形状 (N, )，类别索引。
# #         """
# #         # 0. 过滤目标类别
# #         if self.target_classes is not None:
# #             mask = torch.zeros_like(labels, dtype=torch.bool)
# #             for cls_id in self.target_classes:
# #                 mask |= (labels == cls_id)
# #             if mask.sum() == 0:
# #                 return pred_bboxes.new_tensor(0.0)

# #             pred_bboxes = pred_bboxes[mask]
# #             scores = scores[mask]
# #             labels = labels[mask]

# #         # 1. 准备数据
# #         if pred_bboxes.shape[-1] >= 4:
# #             x_c = pred_bboxes[:, 0]
# #             y_c = pred_bboxes[:, 1]
# #             w = pred_bboxes[:, 2].clamp(min=1e-2)
# #             h = pred_bboxes[:, 3].clamp(min=1e-2)
# #         else:
# #             return pred_bboxes.new_tensor(0.0)

# #         # 目标向量 Y：对数空间下的 log(sqrt(area))
# #         s_log = 0.5 * torch.log(w * h)
# #         Y = s_log.unsqueeze(1)

# #         # 权重 W：开方以转换为普通最小二乘法 (OLS) 形式
# #         # 加权逻辑：高置信度的物体对平面拟合有更强的影响
# #         weights = scores.clamp(min=1e-6)
# #         sqrt_w = torch.sqrt(weights).unsqueeze(1)

# #         # 2. 检查约束
# #         # 我们需要 N >= K + 3 个样本来稳定求解 (2个斜率 + K个截距)
# #         unique_labels, labels_inv = torch.unique(labels, return_inverse=True)
# #         K = len(unique_labels)
# #         N = len(pred_bboxes)

# #         if N < K + 3:
# #             return pred_bboxes.new_tensor(0.0)

# #         # 3. 归一化坐标
# #         if self.norm_type == 'z-score':
# #             # 截断统计量的梯度，防止均值/标准差导致的不稳定梯度
# #             x_mean, x_std = x_c.mean().detach(), x_c.std().detach().clamp(min=1e-6)
# #             y_mean, y_std = y_c.mean().detach(), y_c.std().detach().clamp(min=1e-6)
# #             x_norm = (x_c - x_mean) / x_std
# #             y_norm = (y_c - y_mean) / y_std
# #         else:
# #             x_norm = x_c
# #             y_norm = y_c

# #         # 4. 构建设计矩阵 A
# #         # 结构：[x_norm, y_norm, one_hot_intercepts...]
# #         A = pred_bboxes.new_zeros((N, 2 + K))
# #         A[:, 0] = x_norm
# #         A[:, 1] = y_norm
# #         # 填充特定类别截距的 One-hot 编码
# #         A.scatter_(1, 2 + labels_inv.unsqueeze(1), 1.0)

# #         # 5. 将权重应用于矩阵形式
# #         # 等价于求解：argmin || sqrt(W) * (A * theta - Y) ||^2
# #         A_w = A * sqrt_w
# #         Y_w = Y * sqrt_w

# #         # 6. 可微分求解器 (岭回归)
# #         # 公式：theta = (A^T A + lambda I)^-1 A^T Y
# #         M = torch.matmul(A_w.t(), A_w)
# #         I_reg = torch.eye(2 + K, device=pred_bboxes.device) * self.ridge_lambda
# #         M_reg = M + I_reg

# #         try:
# #             RHS = torch.matmul(A_w.t(), Y_w)
# #             # torch.linalg.solve 允许梯度通过 M_reg 和 RHS 传回输入
# #             theta = torch.linalg.solve(M_reg, RHS)
# #         except RuntimeError:
# #             # 奇异矩阵的回退处理
# #             return pred_bboxes.new_tensor(0.0)

# #         # 7. 计算损失 (端到端，不截断梯度)
# #         # 基于拟合平面计算预期尺寸
# #         Y_hat = torch.matmul(A, theta).squeeze(1)

# #         # 对数空间下的残差 (代表线性空间下的比例误差)
# #         diff = s_log - Y_hat

# #         # 加权 Smooth L1 Loss
# #         loss_element = self.smooth_l1(diff, torch.zeros_like(diff))
# #         loss_weighted = torch.sum(loss_element * weights) / (torch.sum(weights) + 1e-6)

# #         return self.loss_weight * loss_weighted

# @MODELS.register_module()
# class PerspectiveAwareSizeConsistencyLoss(nn.Module):
#     """透视感知尺寸一致性损失 (Image-wise V2 - Strict Mode)。

#     修正版：
#     1. 强制要求传入 batch_idxs，严禁将整个 Batch 混在一起拟合。
#     2. 在 image-norm 模式下强制要求传入 img_metas。
    
#     公式：log(s) = wx * x + wy * y + b_class
#     """

#     def __init__(self,
#                  loss_weight=1.0,
#                  ridge_lambda=1e-4,
#                  beta=1.0,
#                  norm_type='z-score',
#                  target_classes=None):
#         super(PerspectiveAwareSizeConsistencyLoss, self).__init__()
#         self.loss_weight = loss_weight
#         self.ridge_lambda = ridge_lambda
#         self.beta = beta
#         self.norm_type = norm_type
#         self.target_classes = target_classes
#         self.smooth_l1 = nn.SmoothL1Loss(reduction='none', beta=beta)

#     def _solve_single_image(self, pred_bboxes, scores, labels, img_shape=None):
#         """
#         针对单张图片的求解逻辑。
#         """
#         # 1. 准备数据
#         if pred_bboxes.shape[0] == 0:
#             return pred_bboxes.new_tensor(0.0)

#         x_c = pred_bboxes[:, 0]
#         y_c = pred_bboxes[:, 1]
#         w = pred_bboxes[:, 2].clamp(min=1e-2)
#         h = pred_bboxes[:, 3].clamp(min=1e-2)

#         # 目标向量 Y：对数空间下的 log(sqrt(area))
#         s_log = 0.5 * torch.log(w * h)
#         Y = s_log.unsqueeze(1)

#         # 权重 W
#         weights = scores.clamp(min=1e-6)
#         sqrt_w = torch.sqrt(weights).unsqueeze(1)

#         # 2. 检查约束
#         unique_labels, labels_inv = torch.unique(labels, return_inverse=True)
#         K = len(unique_labels)
#         N = len(pred_bboxes)

#         # 样本数必须足以求解 (2个斜率 + K个截距)
#         if N < K + 3:
#             return pred_bboxes.new_tensor(0.0)

#         # 3. 归一化坐标 (Image-wise)
#         if self.norm_type == 'z-score':
#             # Z-Score: 以当前图片内物体的重心为原点
#             x_mean, x_std = x_c.mean().detach(), x_c.std().detach().clamp(min=1e-6)
#             y_mean, y_std = y_c.mean().detach(), y_c.std().detach().clamp(min=1e-6)
#             x_norm = (x_c - x_mean) / x_std
#             y_norm = (y_c - y_mean) / y_std
            
#         elif self.norm_type == 'image-norm':
#             # [严格检查] 如果是 image-norm，必须有 img_shape
#             if img_shape is None:
#                 raise ValueError(
#                     "PerspectiveAwareSizeConsistencyLoss: 'norm_type' is set to 'image-norm', "
#                     "but 'img_shape' was not provided. Please ensure 'img_metas' is passed correctly."
#                 )
            
#             H_img, W_img = img_shape[:2]
#             # 公式：x' = (x - W/2) / (W/2) = 2x/W - 1
#             x_norm = (x_c - W_img / 2.0) / (W_img / 2.0)
#             y_norm = (y_c - H_img / 2.0) / (H_img / 2.0)
#         else:
#             # None
#             x_norm = x_c
#             y_norm = y_c

#         # 4. 构建设计矩阵 A
#         A = pred_bboxes.new_zeros((N, 2 + K))
#         A[:, 0] = x_norm
#         A[:, 1] = y_norm
#         # 填充特定类别截距的 One-hot 编码
#         A.scatter_(1, 2 + labels_inv.unsqueeze(1), 1.0)

#         # 5. 加权岭回归求解
#         A_w = A * sqrt_w
#         Y_w = Y * sqrt_w

#         M = torch.matmul(A_w.t(), A_w)
#         I_reg = torch.eye(2 + K, device=pred_bboxes.device) * self.ridge_lambda
#         M_reg = M + I_reg

#         try:
#             RHS = torch.matmul(A_w.t(), Y_w)
#             theta = torch.linalg.solve(M_reg, RHS)
#         except RuntimeError:
#             return pred_bboxes.new_tensor(0.0)

#         # 6. 计算 Loss
#         Y_hat = torch.matmul(A, theta).squeeze(1)
#         diff = s_log - Y_hat
        
#         loss_element = self.smooth_l1(diff, torch.zeros_like(diff))
#         loss_weighted = torch.sum(loss_element * weights) / (torch.sum(weights) + 1e-6)

#         return loss_weighted

#     def forward(self, pred_bboxes, scores, labels, batch_idxs=None, img_metas=None):
#         """
#         参数：
#             pred_bboxes (Tensor): 形状 (N, 5)
#             scores (Tensor): 形状 (N, )
#             labels (Tensor): 形状 (N, )
#             batch_idxs (Tensor): 形状 (N, )，必须提供
#             img_metas (list[dict]): 图片元信息
#         """
#         # 0. 过滤目标类别
#         if self.target_classes is not None:
#             mask = torch.zeros_like(labels, dtype=torch.bool)
#             for cls_id in self.target_classes:
#                 mask |= (labels == cls_id)
            
#             if mask.sum() == 0:
#                 return pred_bboxes.new_tensor(0.0)

#             pred_bboxes = pred_bboxes[mask]
#             scores = scores[mask]
#             labels = labels[mask]
#             if batch_idxs is not None:
#                 batch_idxs = batch_idxs[mask]

#         # 1. 基础检查
#         if pred_bboxes.shape[-1] < 4:
#             return pred_bboxes.new_tensor(0.0)
        
#         # 2. [严格检查] 必须提供 batch_idxs
#         if batch_idxs is None:
#             raise ValueError(
#                 "PerspectiveAwareSizeConsistencyLoss: 'batch_idxs' is None. "
#                 "You must modify your Head code to pass 'batch_idxs' (Image ID for each bbox) "
#                 "to ensure image-wise fitting. Global fitting on the whole batch is mathematically wrong."
#             )

#         # 3. 逐图计算 Loss (Image-wise Loop)
#         unique_batches = torch.unique(batch_idxs)
#         total_loss = pred_bboxes.new_tensor(0.0)
#         valid_batches = 0.0

#         for b_idx in unique_batches:
#             # 提取属于当前图片的数据
#             mask = (batch_idxs == b_idx)
#             b_pred_bboxes = pred_bboxes[mask]
#             b_scores = scores[mask]
#             b_labels = labels[mask]
            
#             # 获取当前图片的尺寸 (用于 image-norm)
#             b_img_shape = None
#             if img_metas is not None and len(img_metas) > int(b_idx.item()):
#                 b_img_shape = img_metas[int(b_idx.item())]['img_shape']
            
#             # 如果 norm_type 是 image-norm 且 b_img_shape 没拿到，_solve_single_image 内部会报错
            
#             # 计算该图的 Loss
#             loss_per_img = self._solve_single_image(b_pred_bboxes, b_scores, b_labels, img_shape=b_img_shape)
            
#             # 如果该图有效 (loss > 0)，累加
#             if loss_per_img > 0:
#                 total_loss += loss_per_img
#                 valid_batches += 1.0

#         # 4. 求平均 Loss
#         if valid_batches > 0:
#             return self.loss_weight * (total_loss / valid_batches)
#         else:
#             return pred_bboxes.new_tensor(0.0)

# Copyright (c) OpenMMLab. All rights reserved.
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from mmdet.models.losses.utils import weighted_loss

from mmrotate.registry import MODELS
from mmrotate.models.losses.gaussian_dist_loss import postprocess


@weighted_loss
def gwd_sigma_loss(pred, target, fun='log1p', tau=1.0, alpha=1.0, normalize=True):
    """Gaussian Wasserstein distance loss.
    Modified from gwd_loss. 
    gwd_sigma_loss only involves sigma in Gaussian, with mu ignored.

    Args:
        pred (torch.Tensor): Predicted bboxes.
        target (torch.Tensor): Corresponding gt bboxes.
        fun (str): The function applied to distance. Defaults to 'log1p'.
        tau (float): Defaults to 1.0.
        alpha (float): Defaults to 1.0.
        normalize (bool): Whether to normalize the distance. Defaults to True.

    Returns:
        loss (torch.Tensor)

    """
    Sigma_p = pred
    Sigma_t = target

    whr_distance = Sigma_p.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    whr_distance = whr_distance + Sigma_t.diagonal(
        dim1=-2, dim2=-1).sum(dim=-1)

    _t_tr = (Sigma_p.bmm(Sigma_t)).diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    _t_det_sqrt = (Sigma_p.det() * Sigma_t.det()).clamp(1e-7).sqrt()
    whr_distance = whr_distance + (-2) * (
        (_t_tr + 2 * _t_det_sqrt).clamp(1e-7).sqrt())

    distance = (alpha * alpha * whr_distance).clamp(1e-7).sqrt()

    if normalize:
        scale = 2 * (
            _t_det_sqrt.clamp(1e-7).sqrt().clamp(1e-7).sqrt()).clamp(1e-7)
        distance = distance / scale

    return postprocess(distance, fun=fun, tau=tau)


def bhattacharyya_coefficient(pred, target):
    """Calculate bhattacharyya coefficient between 2-D Gaussian distributions.

    Args:
        pred (Tuple): tuple of (xy, sigma).
            xy (torch.Tensor): center point of 2-D Gaussian distribution
                with shape (N, 2).
            sigma (torch.Tensor): covariance matrix of 2-D Gaussian distribution
                with shape (N, 2, 2).
        target (Tuple): tuple of (xy, sigma).

    Returns:
        coef (Tensor): bhattacharyya coefficient with shape (N,).
    """
    xy_p, Sigma_p = pred
    xy_t, Sigma_t = target

    _shape = xy_p.shape

    xy_p = xy_p.reshape(-1, 2)
    xy_t = xy_t.reshape(-1, 2)
    Sigma_p = Sigma_p.reshape(-1, 2, 2)
    Sigma_t = Sigma_t.reshape(-1, 2, 2)

    Sigma_M = (Sigma_p + Sigma_t) / 2
    dxy = (xy_p - xy_t).unsqueeze(-1)
    t0 = torch.exp(-0.125 * dxy.permute(0, 2, 1).bmm(torch.linalg.solve(Sigma_M, dxy)))
    t1 = (Sigma_p.det() * Sigma_t.det()).clamp(1e-7).sqrt()
    t2 = Sigma_M.det()

    coef = t0 * (t1 / t2).clamp(1e-7).sqrt()[..., None, None]
    coef = coef.reshape(_shape[:-1])
    return coef


@weighted_loss
def gaussian_overlap_loss(pred, target, alpha=0.01, beta=0.6065):
    """Calculate Gaussian overlap loss based on bhattacharyya coefficient.

    Args:
        pred (Tuple): tuple of (xy, sigma).
            xy (torch.Tensor): center point of 2-D Gaussian distribution
                with shape (N, 2).
            sigma (torch.Tensor): covariance matrix of 2-D Gaussian distribution
                with shape (N, 2, 2).

    Returns:
        loss (Tensor): overlap loss with shape (N, N).
    """
    mu, sigma = pred
    B = mu.shape[0]
    mu0 = mu[None].expand(B, B, 2)
    sigma0 = sigma[None].expand(B, B, 2, 2)
    mu1 = mu[:, None].expand(B, B, 2)
    sigma1 = sigma[:, None].expand(B, B, 2, 2)
    loss = bhattacharyya_coefficient((mu0, sigma0), (mu1, sigma1))
    loss[torch.eye(B, dtype=bool)] = 0
    loss = F.leaky_relu(loss - beta, negative_slope=alpha) + beta * alpha
    loss = loss.sum(-1)
    return loss


@MODELS.register_module()
class GaussianOverlapLoss(nn.Module):
    """Gaussian Overlap Loss.

    Args:
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'. Options are "none", "mean" and
            "sum".
        loss_weight (float, optional): Weight of loss. Defaults to 1.0.

    Returns:
        loss (torch.Tensor)
    """

    def __init__(self,
                 reduction='mean',
                 loss_weight=1.0,
                 lamb=1e-4):
        super(GaussianOverlapLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.lamb = lamb

    def forward(self,
                pred,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (Tuple): tuple of (xy, sigma).
                xy (torch.Tensor): center point of 2-D Gaussian distribution
                    with shape (N, 2).
                sigma (torch.Tensor): covariance matrix of 2-D Gaussian distribution
                    with shape (N, 2, 2).
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum".

        Returns:
            torch.Tensor: The calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        assert len(pred[0]) == len(pred[1])

        sigma = pred[1]
        L = torch.linalg.eigh(sigma)[0].clamp(1e-7).sqrt()
        loss_lamb = F.l1_loss(L, torch.zeros_like(L), reduction='none')
        loss_lamb = self.lamb * loss_lamb.log1p().mean()
        
        return self.loss_weight * (loss_lamb + gaussian_overlap_loss(
            pred,
            None,
            weight,
            reduction=reduction,
            avg_factor=avg_factor))


def plot_gaussian_voronoi_watershed(*images):
    """Plot figures for debug."""
    import matplotlib.pyplot as plt
    plt.figure(dpi=300, figsize=(len(images) * 4, 4))
    plt.tight_layout()
    fileid = np.random.randint(0, 20)
    for i in range(len(images)):
        img = images[i]
        img = (img - img.min()) / (img.max() - img.min())
        if img.dim() == 3:
            img = img.permute(1, 2, 0)
        img = img.detach().cpu().numpy()
        plt.subplot(1, len(images), i + 1)
        if i == 3:
            plt.imshow(img)
            x = np.linspace(0, 1024, 1024)
            y = np.linspace(0, 1024, 1024)
            X, Y = np.meshgrid(x, y)
            plt.contourf(X, Y, img, levels=8, cmap=plt.get_cmap('magma'))
        else:
            plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
    plt.savefig(f'debug/Gaussian-Voronoi-{fileid}.png')
    plt.close()


def gaussian_2d(xy, mu, sigma, normalize=False):
    dxy = (xy - mu).unsqueeze(-1)
    t0 = torch.exp(-0.5 * dxy.permute(0, 2, 1).bmm(torch.linalg.solve(sigma, dxy)))
    if normalize:
        t0 = t0 / (2 * np.pi * sigma.det().clamp(1e-7).sqrt())
    return t0


def gaussian_voronoi_watershed_loss(mu, sigma,
                                    label, image, 
                                    pos_thres, neg_thres, 
                                    down_sample=2, topk=0.95, 
                                    default_sigma=4096,
                                    voronoi='gaussian-orientation',
                                    alpha=0.1,
                                    debug=False):
    J = len(sigma)
    if J == 0:
        return sigma.sum()
    
    D = down_sample
    H, W = image.shape[-2:]
    h, w = H // D, W // D
    x = torch.linspace(0, h, h, device=mu.device)
    y = torch.linspace(0, w, w, device=mu.device)
    xy = torch.stack(torch.meshgrid(x, y, indexing='xy'), -1)
    vor = mu.new_zeros(J, h, w)
    # Get distribution for each instance
    mm = (mu.detach() / D).round()
    if voronoi == 'standard':
        sg = sigma.new_tensor((default_sigma, 0, 0, default_sigma)).reshape(2, 2)
        sg = sg / D ** 2
        for j, m in enumerate(mm):
            vor[j] = gaussian_2d(xy.view(-1, 2), m[None], sg[None]).view(h, w)
    elif voronoi == 'gaussian-orientation':
        L, V = torch.linalg.eigh(sigma)
        L = L.detach().clone()
        L = L / (L[:, 0:1] * L[:, 1:2]).sqrt() * default_sigma
        sg = V.matmul(torch.diag_embed(L)).matmul(V.permute(0, 2, 1)).detach()
        sg = sg / D ** 2
        for j, (m, s) in enumerate(zip(mm, sg)):
            vor[j] = gaussian_2d(xy.view(-1, 2), m[None], s[None]).view(h, w)
    elif voronoi == 'gaussian-full':
        sg = sigma.detach() / D ** 2
        for j, (m, s) in enumerate(zip(mm, sg)):
            vor[j] = gaussian_2d(xy.view(-1, 2), m[None], s[None]).view(h, w)
    # val: max prob, vor: belong to which instance, cls: belong to which class
    val, vor = torch.max(vor, 0)
    if D > 1:
        vor = vor[:, None, :, None].expand(-1, D, -1, D).reshape(H, W)
        val = F.interpolate(
            val[None, None], (H, W), mode='bilinear', align_corners=True)[0, 0]
    cls = label[vor]
    kernel = val.new_ones((1, 1, 3, 3))
    kernel[0, 0, 1, 1] = -8
    ridges = torch.conv2d(vor[None].float(), kernel, padding=1)[0] != 0
    vor += 1
    pos_thres = val.new_tensor(pos_thres)
    neg_thres = val.new_tensor(neg_thres)
    vor[val < pos_thres[cls]] = 0
    vor[val < neg_thres[cls]] = J + 1
    vor[ridges] = J + 1

    cls_bg = torch.where(vor == J + 1, 15, cls)
    cls_bg = torch.where(vor == 0, -1, cls_bg)

    # PyTorch does not support watershed, use cv2
    img_uint8 = (image - image.min()) / (image.max() - image.min()) * 255
    img_uint8 = img_uint8.permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)
    img_uint8 = cv2.medianBlur(img_uint8, 3)
    markers = vor.detach().cpu().numpy().astype(np.int32)
    markers = vor.new_tensor(cv2.watershed(img_uint8, markers))

    if debug:
        plot_gaussian_voronoi_watershed(image, cls_bg, markers)

    L, V = torch.linalg.eigh(sigma)
    L_target = []
    for j in range(J):
        xy = (markers == j + 1).nonzero()[:, (1, 0)].float()
        if len(xy) == 0:
            L_target.append(L[j].detach())
            continue
        xy = xy - mu[j]
        xy = V[j].T.matmul(xy[:, :, None])[:, :, 0]
        max_x = torch.max(torch.abs(xy[:, 0]))
        max_y = torch.max(torch.abs(xy[:, 1]))
        L_target.append(torch.stack((max_x, max_y)) ** 2)
    L_target = torch.stack(L_target)
    L = torch.diag_embed(L)
    L_target = torch.diag_embed(L_target)
    loss = gwd_sigma_loss(L, L_target.detach(), reduction='none')
    loss = torch.topk(loss, int(np.ceil(len(loss) * topk)), largest=False)[0].mean()
    return loss, (vor, markers)


@MODELS.register_module()
class VoronoiWatershedLoss(nn.Module):
    """Gaussian Overlap Loss.

    Args:
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'. Options are "none", "mean" and
            "sum".
        loss_weight (float, optional): Weight of loss. Defaults to 1.0.

    Returns:
        loss (torch.Tensor)
    """

    def __init__(self,
                 down_sample=2,
                 reduction='mean',
                 loss_weight=1.0,
                 topk=0.95,
                 alpha=0.1,
                 debug=False):
        super(VoronoiWatershedLoss, self).__init__()
        self.down_sample = down_sample
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.topk = topk
        self.alpha = alpha
        self.debug = debug

    def forward(self, pred, label, image, pos_thres, neg_thres, voronoi='orientation'):
        """Forward function.

        Args:
            pred (Tuple): Tuple of (xy, sigma).
                xy (torch.Tensor): Center point of 2-D Gaussian distribution
                    with shape (N, 2).
                sigma (torch.Tensor): Covariance matrix of 2-D Gaussian distribution
                    with shape (N, 2, 2).
            image (torch.Tensor): The image for watershed with shape (3, H, W).
            standard_voronoi (bool, optional): Use standard or Gaussian voronoi.

        Returns:
            torch.Tensor: The calculated loss
        """
        loss, self.vis = gaussian_voronoi_watershed_loss(*pred, 
                                                         label,
                                                         image, 
                                                         pos_thres, 
                                                         neg_thres, 
                                                         self.down_sample, 
                                                         topk=self.topk,
                                                         voronoi=voronoi,
                                                         alpha=self.alpha,
                                                         debug=self.debug)
        return self.loss_weight * loss


def rbbox2roi(bbox_list):
    """Convert a list of bboxes to roi format.

    Args:
        bbox_list (list[Tensor]): a list of bboxes corresponding to a batch
            of images.

    Returns:
        Tensor: shape (n, 6), [batch_ind, cx, cy, w, h, a]
    """
    rois_list = []
    for img_id, bboxes in enumerate(bbox_list):
        if bboxes.size(0) > 0:
            img_inds = bboxes.new_full((bboxes.size(0), 1), img_id)
            rois = torch.cat([img_inds, bboxes[:, :5]], dim=-1)
        else:
            rois = bboxes.new_zeros((0, 6))
        rois_list.append(rois)
    rois = torch.cat(rois_list, 0)
    return rois


def plot_edge_map(feat, edgex, edgey):
    """Plot figures for debug."""
    import matplotlib.pyplot as plt
    plt.figure(dpi=300, figsize=(4, 4))
    plt.tight_layout()
    fileid = np.random.randint(0, 20)
    for i in range(len(feat)):
        img0 = feat[i, :3]
        img0 = (img0 - img0.min()) / (img0.max() - img0.min())
        img1 = edgex[i, :3]
        img1 = (img1 - img1.min()) / (img1.max() - img1.min())
        img2 = edgey[i, :3]
        img2 = (img2 - img2.min()) / (img2.max() - img2.min())
        img3 = img1 + img2
        img3 = (img3 - img3.min()) / (img3.max() - img3.min())
        img = torch.cat((torch.cat((img0, img2), -1), 
                         torch.cat((img1, img3), -1)), -2
                         ).permute(1, 2, 0).detach().cpu().numpy()
        N = int(np.ceil(np.sqrt(len(feat))))
        plt.subplot(N, N, i + 1)
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
    plt.savefig(f'debug/Edge-Map-{fileid}.png')
    plt.close()


@MODELS.register_module()
class EdgeLoss(nn.Module):
    """Edge Loss.

    Args:
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'. Options are "none", "mean" and
            "sum".
        loss_weight (float, optional): Weight of loss. Defaults to 1.0.

    Returns:
        loss (torch.Tensor)
    """

    def __init__(self,
                 resolution=24,
                 max_scale=1.6,
                 sigma=6,
                 reduction='mean',
                 loss_weight=1.0,
                 debug=False):
        super(EdgeLoss, self).__init__()
        self.resolution = resolution
        self.max_scale = max_scale
        self.sigma = sigma
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.center_idx = self.resolution / self.max_scale
        self.debug = debug

        self.roi_extractor = MODELS.build(dict(
            type='RotatedSingleRoIExtractor',
                roi_layer=dict(
                    type='RoIAlignRotated',
                    out_size=(2 * self.resolution + 1),
                    sample_num=2,
                    clockwise=True),
            out_channels=1,
            featmap_strides=[1],
            finest_scale=1024))

        edge_idx = torch.arange(0, self.resolution + 1)
        edge_distribution = torch.exp(-((edge_idx - self.center_idx) ** 2) / (2 * self.sigma ** 2))
        edge_distribution[0] = edge_distribution[-1] = 0
        self.register_buffer('edge_idx', edge_idx)
        self.register_buffer('edge_distribution', edge_distribution)

    def forward(self, pred, edge):
        """Forward function.

        Args:
            pred (Tuple): Batched predicted rboxes
            edge (torch.Tensor): The edge map with shape (B, 1, H, W).

        Returns:
            torch.Tensor: The calculated loss
        """
        G = self.resolution
        C = self.center_idx
        roi = rbbox2roi(pred)
        roi[:, 3:5] *= self.max_scale
        feat = self.roi_extractor([edge], roi)
        if len(feat) == 0:
            return pred[0].new_tensor(0)
        featx = feat.sum(1).abs().sum(1)
        featy = feat.sum(1).abs().sum(2)
        featx2 = torch.flip(featx[:, :G + 1], (-1,)) + featx[:, G:]
        featy2 = torch.flip(featy[:, :G + 1], (-1,)) + featy[:, G:]  # (N, 25)
        ex = ((featx2 * self.edge_distribution).softmax(1) * self.edge_idx).sum(1) / C
        ey = ((featy2 * self.edge_distribution).softmax(1) * self.edge_idx).sum(1) / C
        exy = torch.stack((ex, ey), -1)
        rbbox_concat = torch.cat(pred, 0)
        
        if self.debug:
            edgex = featx[:, None, None, :].expand(-1, 1, 2 * self.resolution + 1, -1)
            edgey = featy[:, None, :, None].expand(-1, 1, -1, 2 * self.resolution + 1)
            plot_edge_map(feat, edgex, edgey)

        return self.loss_weight * F.smooth_l1_loss(rbbox_concat[:, 2:4], 
                                                   (rbbox_concat[:, 2:4] * exy).detach(),
                                                   beta=8)


@MODELS.register_module()
class Point2RBoxV2ConsistencyLoss(nn.Module):
    """Consistency Loss.

    Args:
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'. Options are "none", "mean" and
            "sum".
        loss_weight (float, optional): Weight of loss. Defaults to 1.0.

    Returns:
        loss (torch.Tensor)
    """

    def __init__(self,
                 reduction='mean',
                 loss_weight=1.0):
        super(Point2RBoxV2ConsistencyLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, ori_pred, trs_pred, square_mask, aug_type, aug_val):
        """Forward function.

        Args:
            ori_pred (Tuple): (Sigma, theta)
            trs_pred (Tuple): (Sigma, theta)
            square_mask: When True, the angle is ignored
            aug_type: 'rot', 'flp', 'sca'
            aug_val: Rotation or scale value

        Returns:
            torch.Tensor: The calculated loss
        """
        ori_gaus, ori_angle = ori_pred
        trs_gaus, trs_angle = trs_pred

        if aug_type == 'rot':
            rot = ori_gaus.new_tensor(aug_val)
            cos_r = torch.cos(rot)
            sin_r = torch.sin(rot)
            R = torch.stack((cos_r, -sin_r, sin_r, cos_r), dim=-1).reshape(-1, 2, 2)
            ori_gaus = R.matmul(ori_gaus).matmul(R.permute(0, 2, 1))
            d_ang = trs_angle - ori_angle - aug_val
        elif aug_type == 'flp':
            ori_gaus = ori_gaus * ori_gaus.new_tensor((1, -1, -1, 1)).reshape(2, 2)
            d_ang = trs_angle + ori_angle
        else:
            sca = ori_gaus.new_tensor(aug_val)
            ori_gaus = ori_gaus * sca
            d_ang = trs_angle - ori_angle
        
        loss_ssg = gwd_sigma_loss(ori_gaus.bmm(ori_gaus), trs_gaus.bmm(trs_gaus))
        d_ang = (d_ang + math.pi / 2) % math.pi - math.pi / 2
        loss_ssa = F.smooth_l1_loss(d_ang, torch.zeros_like(d_ang), reduction='none', beta=0.1)
        loss_ssa = loss_ssa[~square_mask].sum() / max(1, (~square_mask).sum())

        return self.loss_weight * (loss_ssg + loss_ssa)


@MODELS.register_module()
class PerspectiveAwareSizeConsistencyLoss(nn.Module):
    """透视感知尺寸一致性损失 (Image-wise V2 - Strict Mode)。

    修正版：
    1. 强制要求传入 batch_idxs，严禁将整个 Batch 混在一起拟合。
    2. 在 image-norm 模式下强制要求传入 img_metas。
    
    公式：log(s) = wx * x + wy * y + b_class
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
        """
        针对单张图片的求解逻辑。
        """
        # 1. 准备数据
        if pred_bboxes.shape[0] == 0:
            return pred_bboxes.new_tensor(0.0)

        x_c = pred_bboxes[:, 0]
        y_c = pred_bboxes[:, 1]
        w = pred_bboxes[:, 2].clamp(min=1e-2)
        h = pred_bboxes[:, 3].clamp(min=1e-2)

        # 目标向量 Y：对数空间下的 log(sqrt(area))
        s_log = 0.5 * torch.log(w * h)
        Y = s_log.unsqueeze(1).float()  # [关键] 强转 FP32

        # 权重 W
        weights = scores.clamp(min=1e-6)
        sqrt_w = torch.sqrt(weights).unsqueeze(1).float() # [关键] 强转 FP32

        # 2. 检查约束
        unique_labels, labels_inv = torch.unique(labels, return_inverse=True)
        K = len(unique_labels)
        N = len(pred_bboxes)

        # 样本数必须足以求解 (2个斜率 + K个截距)
        if N < K + 3:
            return pred_bboxes.new_tensor(0.0)

        # 3. 归一化坐标 (Image-wise)
        if self.norm_type == 'z-score':
            # Z-Score: 以当前图片内物体的重心为原点
            x_mean, x_std = x_c.mean().detach(), x_c.std().detach().clamp(min=1e-6)
            y_mean, y_std = y_c.mean().detach(), y_c.std().detach().clamp(min=1e-6)
            x_norm = (x_c - x_mean) / x_std
            y_norm = (y_c - y_mean) / y_std
            
        elif self.norm_type == 'image-norm':
            # [严格检查] 如果是 image-norm，必须有 img_shape
            if img_shape is None:
                raise ValueError(
                    "PerspectiveAwareSizeConsistencyLoss: 'norm_type' is set to 'image-norm', "
                    "but 'img_shape' was not provided. Please ensure 'img_metas' is passed correctly."
                )
            
            H_img, W_img = img_shape[:2]
            # 公式：x' = (x - W/2) / (W/2) = 2x/W - 1
            x_norm = (x_c - W_img / 2.0) / (W_img / 2.0)
            y_norm = (y_c - H_img / 2.0) / (H_img / 2.0)
        else:
            # None
            x_norm = x_c
            y_norm = y_c

        # 4. 构建设计矩阵 A
        A = pred_bboxes.new_zeros((N, 2 + K)).float()  # [关键] 强转 FP32
        A[:, 0] = x_norm
        A[:, 1] = y_norm
        # 填充特定类别截距的 One-hot 编码
        A.scatter_(1, 2 + labels_inv.unsqueeze(1), 1.0)

        # 5. 加权岭回归求解
        A_w = A * sqrt_w
        Y_w = Y * sqrt_w

        M = torch.matmul(A_w.t(), A_w)
        I_reg = torch.eye(2 + K, device=pred_bboxes.device).float() * self.ridge_lambda
        M_reg = M + I_reg

        try:
            RHS = torch.matmul(A_w.t(), Y_w)
            theta = torch.linalg.solve(M_reg, RHS)
        except RuntimeError:
            return pred_bboxes.new_tensor(0.0)

        # 6. 计算 Loss
        Y_hat = torch.matmul(A, theta).squeeze(1)
        diff = s_log - Y_hat
        
        # Loss 计算可以是 FP32，也可以是混合精度，但通常这里保持 FP32 精度更好
        loss_element = self.smooth_l1(diff, torch.zeros_like(diff))
        
        # 将权重转回和 loss_element 一致的类型 (防止 sqrt_w 是 FP32 而 loss 是 FP16 导致的类型不匹配)
        weights = weights.to(loss_element.dtype)
        
        loss_weighted = torch.sum(loss_element * weights) / (torch.sum(weights) + 1e-6)

        return loss_weighted

    def forward(self, pred_bboxes, scores, labels, batch_idxs=None, img_metas=None):
        """
        参数：
            pred_bboxes (Tensor): 形状 (N, 5)
            scores (Tensor): 形状 (N, )
            labels (Tensor): 形状 (N, )
            batch_idxs (Tensor): 形状 (N, )，必须提供
            img_metas (list[dict]): 图片元信息
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

        # 1. 基础检查
        if pred_bboxes.shape[-1] < 4:
            return pred_bboxes.new_tensor(0.0)
        
        # 2. [严格检查] 必须提供 batch_idxs
        if batch_idxs is None:
            raise ValueError(
                "PerspectiveAwareSizeConsistencyLoss: 'batch_idxs' is None. "
                "You must modify your Head code to pass 'batch_idxs' (Image ID for each bbox) "
                "to ensure image-wise fitting. Global fitting on the whole batch is mathematically wrong."
            )

        # 3. 逐图计算 Loss (Image-wise Loop)
        unique_batches = torch.unique(batch_idxs)
        total_loss = pred_bboxes.new_tensor(0.0)
        valid_batches = 0.0

        for b_idx in unique_batches:
            # 提取属于当前图片的数据
            mask = (batch_idxs == b_idx)
            b_pred_bboxes = pred_bboxes[mask]
            b_scores = scores[mask]
            b_labels = labels[mask]
            
            # 获取当前图片的尺寸 (用于 image-norm)
            b_img_shape = None
            if img_metas is not None and len(img_metas) > int(b_idx.item()):
                b_img_shape = img_metas[int(b_idx.item())]['img_shape']
            
            # 如果 norm_type 是 image-norm 且 b_img_shape 没拿到，_solve_single_image 内部会报错
            
            # 计算该图的 Loss
            loss_per_img = self._solve_single_image(b_pred_bboxes, b_scores, b_labels, img_shape=b_img_shape)
            
            # 如果该图有效 (loss > 0)，累加
            if loss_per_img > 0:
                total_loss += loss_per_img
                valid_batches += 1.0

        # 4. 求平均 Loss
        if valid_batches > 0:
            return self.loss_weight * (total_loss / valid_batches)
        else:
            return pred_bboxes.new_tensor(0.0)