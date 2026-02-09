# # Copyright (c) OpenMMLab. All rights reserved.
# import os, copy, math
# from typing import Dict, List, Optional, Tuple

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from mmcv.cnn import Scale, ConvModule
# from mmdet.models.dense_heads import AnchorFreeHead
# from mmdet.models.utils import (filter_scores_and_topk, multi_apply,
#                                 select_single_mlvl, unpack_gt_instances)
# from mmdet.structures import SampleList
# from mmdet.structures.bbox import cat_boxes
# from mmdet.utils import (ConfigType, InstanceList, OptConfigType,
#                          OptInstanceList, reduce_mean)
# from mmengine import ConfigDict
# from mmengine.structures import InstanceData
# from torch import Tensor

# from mmrotate.registry import MODELS, TASK_UTILS
# from mmrotate.structures import RotatedBoxes, rbox2qbox, hbox2rbox, rbox2hbox
# from mmrotate.models.losses.gaussian_dist_loss import xy_wh_r_2_xy_sigma, gwd_loss
# # 【修改点】：导入自定义的可导 ROI Align
# from mmrotate.models.utils.differentiable_roi_align import DifferentiableRoIAlignRotated

# INF = 1e8


# @MODELS.register_module()
# class Point2RBoxV2Head(AnchorFreeHead):
#     """Anchor-free head used in `FCOS <https://arxiv.org/abs/1904.01355>`_.

#     Compared with FCOS head, Rotated FCOS head add a angle branch to
#     support rotated object detection.
#     """  # noqa: E501

#     def __init__(self,
#                  num_classes: int,
#                  in_channels: int,
#                  strides: list = [8],
#                  regress_ranges: list = [(-1, 1e8)],
#                  center_sampling: bool = True,
#                  center_sample_radius: float = 0.75,
#                  angle_version: str = 'le90',
#                  edge_loss_start_epoch: int = 6,
#                  joint_angle_start_epoch: int = 1,
#                  pseudo_generator: bool = False,
#                  voronoi_type: str = 'gaussian-orientation',
#                  voronoi_thres: dict = dict(default=[0.994, 0.005]),
#                  square_cls: list = [],
#                  edge_loss_cls: list = [],
#                  post_process: dict = {},
#                  bbox_coder: ConfigType = dict(type='DistanceAnglePointCoder'),
#                  angle_coder: ConfigType = dict(
#                     type='PSCCoder',
#                     angle_version='le90',
#                     dual_freq=False,
#                     num_step=3,
#                     thr_mod=0),
#                  loss_cls: ConfigType = dict(
#                      type='mmdet.CrossEntropyLoss', # 推荐改为 CrossEntropyLoss
#                      use_sigmoid=False,
#                      loss_weight=1.0),
#                  loss_bbox: ConfigType = dict(
#                      type='GWDLoss', loss_weight=5.0),
#                  loss_overlap: ConfigType = dict(
#                      type='GaussianOverlapLoss', loss_weight=10.0),
#                  loss_voronoi: ConfigType = dict(
#                      type='VoronoiWatershedLoss', loss_weight=5.0),
#                  loss_bbox_edg: ConfigType = dict(
#                      type='EdgeLoss', loss_weight=0.3),
#                  loss_ss=dict(
#                     type='Point2RBoxV2ConsistencyLoss', loss_weight=1.0),
#                  # Added: Loss for Perspective Awareness
#                  loss_perspective: ConfigType = dict(
#                      type='PerspectiveAwareSizeConsistencyLoss', loss_weight=1.0),
#                  # Added: NAOA Loss
#                  loss_naoa: ConfigType = dict(
#                      type='NAOALoss', loss_weight=1.0),
#                  norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
#                  init_cfg=dict(
#                      type='Normal',
#                      layer='Conv2d',
#                      std=0.01,
#                      override=[
#                          dict(
#                              type='Normal',
#                              name='conv_cls',
#                              std=0.01,
#                              bias_prob=0.01),
#                          dict(
#                              type='Normal',
#                              name='conv_gate',
#                              std=0.01,
#                              bias_prob=0.01)]),
#                  **kwargs):
#         self.angle_coder = TASK_UTILS.build(angle_coder)
#         super().__init__(
#             num_classes,
#             in_channels,
#             strides=strides,
#             bbox_coder=bbox_coder,
#             loss_cls=loss_cls,
#             loss_bbox=loss_bbox,
#             norm_cfg=norm_cfg,
#             init_cfg=init_cfg,
#             **kwargs)
#         self.regress_ranges = regress_ranges
#         self.center_sampling = center_sampling
#         self.center_sample_radius = center_sample_radius
#         self.angle_version = angle_version
#         self.edge_loss_start_epoch = edge_loss_start_epoch
#         self.joint_angle_start_epoch = joint_angle_start_epoch
#         self.pseudo_generator = pseudo_generator
#         self.voronoi_type = voronoi_type
#         self.voronoi_thres = voronoi_thres
#         self.square_cls = square_cls
#         self.edge_loss_cls = edge_loss_cls
#         self.post_process = post_process
#         self.loss_ss = MODELS.build(loss_ss)
#         self.loss_overlap = MODELS.build(loss_overlap)
#         self.loss_voronoi = MODELS.build(loss_voronoi)
#         self.loss_bbox_edg = MODELS.build(loss_bbox_edg)
#         self.loss_perspective = MODELS.build(loss_perspective)
#         self.loss_naoa = MODELS.build(loss_naoa)
        
#         # 【修改点】：初始化 RoI Align 和 ROI 分类头
#         # 假设 stride[0] (通常是 8) 是我们做 ROI Align 的基础特征层
#         self.roi_align = DifferentiableRoIAlignRotated(
#             output_size=(7, 7),
#             spatial_scale=1.0 / self.strides[0], 
#             sampling_ratio=0, 
#             clockwise=True
#         )
        
#         # 定义 ROI 后分类的 MLP： Flatten -> Linear -> ReLU -> Linear
#         # 输入特征维度 = in_channels * 7 * 7
#         # 输出维度 = num_classes + 1 (增加背景类)
#         self.roi_cls_head = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(self.feat_channels * 7 * 7, 256),
#             nn.ReLU(inplace=True),
#             nn.Linear(256, self.num_classes + 1) 
#         )
            
#     def _init_layers(self):
#         """Initialize layers of the head."""
#         super()._init_layers()
#         # self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])
#         self.conv_angle = nn.Conv2d(
#             self.feat_channels, self.angle_coder.encode_size, 3, padding=1)
#         self.conv_gate = nn.Conv2d(self.feat_channels, 1, 3, padding=1)
        
#     def forward(
#             self, x: Tuple[Tensor]
#     ) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
#         """Forward features from the upstream network.
#         """
#         cls_feat = x[0]
#         reg_feat = x[0]

#         for cls_layer in self.cls_convs:
#             cls_feat = cls_layer(cls_feat)
#         # 这个 cls_score 现在是随机初始化的，只用于 shape 占位，不参与 Loss
#         cls_score = self.conv_cls(cls_feat)

#         for reg_layer in self.reg_convs:
#             reg_feat = reg_layer(reg_feat)
#         bbox_pred = self.conv_reg(reg_feat)
#         angle_pred = self.conv_angle(reg_feat)

#         # Gaussian sig_x, sig_y, p
#         sig_x = bbox_pred[:, 0].exp()
#         sig_y = bbox_pred[:, 1].exp()
#         dx = bbox_pred[:, 2].sigmoid() * 2 - 1  # (-1, 1)
#         dy = bbox_pred[:, 3].sigmoid() * 2 - 1  # (-1, 1)
#         bbox_pred = torch.stack((sig_x, sig_y, dx, dy), 1) * 8

#         return (cls_score,), (bbox_pred,), (angle_pred,)
    
#     # 【修改点】：签名修改，添加 `x` 作为第一个参数
#     def loss_by_feat(
#         self,
#         x: Tuple[Tensor],
#         cls_scores: List[Tensor],
#         bbox_preds: List[Tensor],
#         angle_preds: List[Tensor],
#         batch_gt_instances: InstanceList,
#         batch_img_metas: List[dict],
#         batch_gt_instances_ignore: OptInstanceList = None
#     ) -> Dict[str, Tensor]:
#         """Calculate the loss based on the features extracted by the detection
#         head.
#         """
#         assert len(cls_scores) == len(bbox_preds) == len(angle_preds)
#         featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
#         all_level_points = self.prior_generator.grid_priors(
#             featmap_sizes,
#             dtype=bbox_preds[0].dtype,
#             device=bbox_preds[0].device)
#         # bbox_targets here is in format t,b,l,r
#         # angle_targets is not coded here
#         labels, bbox_targets, bid_targets = self.get_targets(
#             all_level_points, batch_gt_instances)

#         num_imgs = cls_scores[0].size(0)
#         # flatten cls_scores, bbox_preds, angle_preds and centerness
#         flatten_cls_scores = [
#             cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
#             for cls_score in cls_scores
#         ]
#         flatten_bbox_preds = [
#             bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
#             for bbox_pred in bbox_preds
#         ]
#         flatten_angle_preds = [
#             angle_pred.permute(0, 2, 3, 1).reshape(-1, self.angle_coder.encode_size)
#             for angle_pred in angle_preds
#         ]
#         flatten_cls_scores = torch.cat(flatten_cls_scores)
#         flatten_bbox_preds = torch.cat(flatten_bbox_preds)
#         flatten_angle_preds = torch.cat(flatten_angle_preds)
#         flatten_labels = torch.cat(labels)
#         flatten_bbox_targets = torch.cat(bbox_targets)
#         flatten_bid_targets = torch.cat(bid_targets)
#         # repeat points to align with bbox_preds
#         flatten_points = torch.cat(
#             [points.repeat(num_imgs, 1) for points in all_level_points])

#         bg_class_ind = self.num_classes
#         # 正样本索引
#         pos_inds = ((flatten_labels >= 0)
#                     & (flatten_labels < bg_class_ind)).nonzero().reshape(-1)
        
#         # 【修改点】：采样负样本 (背景)
#         # get_targets 会把不满足条件的点标记为 bg_class_ind (即 num_classes)
#         neg_inds = (flatten_labels == bg_class_ind).nonzero().reshape(-1)
#         if len(neg_inds) > 0:
#             # 简单随机采样，保持正负样本比例 1:3 (类似 RPN 的采样策略)
#             # 如果不训练背景，模型就会产生大量误检
#             num_neg_sample = min(len(neg_inds), max(len(pos_inds), 1) * 3)
#             perm = torch.randperm(len(neg_inds), device=neg_inds.device)[:num_neg_sample]
#             neg_inds_sampled = neg_inds[perm]
#         else:
#             neg_inds_sampled = neg_inds.new_zeros(0)

#         # 合并训练样本 (Pos + Neg) 用于分类
#         train_inds = torch.cat([pos_inds, neg_inds_sampled])
        
#         num_pos = torch.tensor(
#             len(pos_inds), dtype=torch.float, device=bbox_preds[0].device)
#         num_pos = max(reduce_mean(num_pos), 1.0)
        
#         # 【修改点】：移除原本的 Point-based loss_cls (Focal Loss) 计算
#         # loss_cls = self.loss_cls(...)

#         # -----------------------------------------------------------
#         # 计算 ROI Classification Loss (Pos + Neg)
#         # -----------------------------------------------------------
#         if len(train_inds) > 0:
#             train_bbox_preds = flatten_bbox_preds[train_inds]
#             train_angle_preds = flatten_angle_preds[train_inds]
#             train_points = flatten_points[train_inds]
#             # train_labels 包含 0~14 (Object) 和 15 (Background)
#             train_labels = flatten_labels[train_inds]
            
#             # 解码角度
#             train_decoded_angle = self.angle_coder.decode(
#                 train_angle_preds, keepdim=True)
#             if self.epoch < self.joint_angle_start_epoch:
#                 train_decoded_angle = train_decoded_angle.detach()
            
#             # 对特定类别角度清零
#             square_mask = torch.zeros_like(train_labels, dtype=torch.bool)
#             for c in self.square_cls:
#                 square_mask = torch.logical_or(square_mask, train_labels == c)
#             train_decoded_angle[square_mask] = 0

#             # 解码框: (cx, cy, w, h, theta)
#             # 使用预测的 bbox_preds 来调整 Proposal，这样梯度能传回 Regression 分支
#             train_rbox_preds = torch.cat((train_points + train_bbox_preds[:, 2:], 
#                                           train_bbox_preds[:, :2] * 2,
#                                           train_decoded_angle), -1)
            
#             # 构造 ROIs: [batch_ind, x, y, w, h, theta]
#             train_batch_idxs = flatten_bid_targets[train_inds, 0:1] # targets里的batch_id是对的
#             rois = torch.cat([train_batch_idxs, train_rbox_preds], dim=1)
            
#             # ROI Align
#             # 使用 x[0] (通常是 stride 8 的特征图)
#             # 输出: [N, C, 7, 7]
#             roi_feats = self.roi_align(x[0], rois) 
            
#             # MLP Prediction -> [N, 16] (logits)
#             roi_cls_preds = self.roi_cls_head(roi_feats)
            
#             # CrossEntropy Loss (自动处理多分类，包括背景类)
#             loss_roi_cls = self.loss_cls(roi_cls_preds, train_labels)
#         else:
#             loss_roi_cls = flatten_bbox_preds.sum() * 0.0

#         # -----------------------------------------------------------
#         # 以下部分仅针对正样本 (Geometric Losses)
#         # -----------------------------------------------------------
        
#         pos_cls_scores = flatten_cls_scores[pos_inds].sigmoid()
#         pos_labels = flatten_labels[pos_inds]
#         # 即使移除了 Point Loss，我们仍提取分数用于 Perspective Loss 的权重计算
#         if len(pos_inds) > 0:
#             pos_cls_scores = torch.gather(pos_cls_scores, 1, pos_labels[:, None])[:, 0]
#         else:
#             pos_cls_scores = flatten_cls_scores.new_zeros(0)

#         pos_bbox_preds = flatten_bbox_preds[pos_inds]
#         pos_angle_preds = flatten_angle_preds[pos_inds]
#         pos_bbox_targets = flatten_bbox_targets[pos_inds]
#         pos_bid_targets = flatten_bid_targets[pos_inds]

#         self.vis = [None] * len(batch_gt_instances)  # For visual debug
        
#         if len(pos_inds) > 0:
#             pos_points = flatten_points[pos_inds]
            
#             # 重新解码正样本角度
#             pos_decoded_angle_preds = self.angle_coder.decode(
#                 pos_angle_preds, keepdim=True)
#             if self.epoch < self.joint_angle_start_epoch:
#                 pos_decoded_angle_preds = pos_decoded_angle_preds.detach()
#             square_mask = torch.zeros_like(pos_labels, dtype=torch.bool)
#             for c in self.square_cls:
#                 square_mask = torch.logical_or(square_mask, pos_labels == c)
#             pos_decoded_angle_preds[square_mask] = 0

#             # 几何 Loss 使用 GT 中心点 (pos_rbox_targets) 来稳定训练
#             pos_rbox_targets = self.bbox_coder.decode(pos_points, pos_bbox_targets)
#             pos_rbox_preds = torch.cat((pos_points + pos_bbox_preds[:, 2:], 
#                                         pos_bbox_preds[:, :2] * 2,
#                                         pos_decoded_angle_preds), -1)

#             cos_r = torch.cos(pos_decoded_angle_preds)
#             sin_r = torch.sin(pos_decoded_angle_preds)
#             R = torch.stack((cos_r, -sin_r, sin_r, cos_r), dim=-1).reshape(-1, 2, 2)
#             pos_gaus_preds = R.matmul(torch.diag_embed(pos_bbox_preds[:, :2])).matmul(R.permute(0, 2, 1))

#             # Regress copy-paste objects
#             pos_syn_mask = pos_bid_targets[:, 1] == 1
#             pos_rbox_targets[~pos_syn_mask, 2:] = pos_rbox_preds[~pos_syn_mask, 2:].detach()
#             loss_bbox = self.loss_bbox(
#                 pos_rbox_preds,
#                 pos_rbox_targets,
#                 avg_factor=num_pos)
            
#             # --- Geometric Losses Calculation ---
#             # 构造用于几何计算的 Preds
#             pos_rbox_preds_geo = torch.cat((pos_rbox_targets[:, :2], 
#                                             pos_bbox_preds[:, :2] * 2,
#                                             pos_decoded_angle_preds), -1)

#             # Instance Grouping Logic (保持原样)
#             bid_with_view = pos_bid_targets[:, 3] + 0.5 * pos_bid_targets[:, 2]
#             bid, idx = torch.unique(bid_with_view, return_inverse=True)
            
#             ins_bid_with_view = bid.new_zeros(*bid.shape).index_reduce_(
#                 0, idx, bid_with_view, 'amin', include_self=False)
#             _, bidx, bcnt = torch.unique(
#                 ins_bid_with_view.long(),
#                 return_inverse=True,
#                 return_counts=True)
#             bmsk = bcnt[bidx] == 2

#             ins_bids = pos_bid_targets.new_zeros(*bid.shape).index_reduce_(
#                     0, idx, pos_bid_targets[:, 3], 'amin', include_self=False)
#             ins_batch = pos_bid_targets.new_zeros(*bid.shape).index_reduce_(
#                     0, idx, pos_bid_targets[:, 0], 'amin', include_self=False)
#             ins_labels = pos_labels.new_zeros(*bid.shape).index_reduce_(
#                     0, idx, pos_labels, 'amin', include_self=False)
            
#             ins_gaus_preds = pos_gaus_preds.new_zeros(
#                 *bid.shape, 4).index_reduce_(
#                     0, idx, pos_gaus_preds.view(-1, 4), 'mean',
#                     include_self=False).view(-1, 2, 2)
            
#             ins_rbox_preds = pos_rbox_preds_geo.new_zeros(
#                 *bid.shape, pos_rbox_preds_geo.shape[-1]).index_reduce_(
#                     0, idx, pos_rbox_preds_geo, 'mean',
#                     include_self=False)
            
#             ins_rbox_targets = pos_rbox_targets.new_zeros(
#                 *bid.shape, pos_rbox_targets.shape[-1]).index_reduce_(
#                     0, idx, pos_rbox_targets, 'mean',
#                     include_self=False)

#             ori_mu_all = ins_rbox_targets[:, 0:2]
#             loss_bbox_ovl = ori_mu_all.new_tensor(0)
#             loss_bbox_vor = ori_mu_all.new_tensor(0)
            
#             for batch_id in range(len(batch_gt_instances)):
#                 group_mask = (ins_batch == batch_id) & (ins_bids != 0)
#                 mu = ori_mu_all[group_mask]
#                 sigma = ins_gaus_preds[group_mask]
#                 label = ins_labels[group_mask]
#                 if len(mu) >= 2:
#                     loss_bbox_ovl += self.loss_overlap((mu, sigma.bmm(sigma)))
#                 if len(mu) >= 1:
#                     pos_thres = [self.voronoi_thres['default'][0]] * self.num_classes
#                     neg_thres = [self.voronoi_thres['default'][1]] * self.num_classes
#                     if 'override' in self.voronoi_thres.keys():
#                         for item in self.voronoi_thres['override']:
#                             for cls in item[0]:
#                                 pos_thres[cls] = item[1][0]
#                                 neg_thres[cls] = item[1][1]
#                     loss_bbox_vor += self.loss_voronoi((mu, sigma.bmm(sigma)),
#                                                         label, self.images[batch_id],
#                                                         pos_thres, neg_thres,
#                                                         voronoi=self.voronoi_type)
#                     self.vis[batch_id] = self.loss_voronoi.vis
            
#             # Edge Loss
#             loss_bbox_edg = ori_mu_all.new_tensor(0)
#             if self.epoch >= self.edge_loss_start_epoch:
#                 batched_rbox = []
#                 for batch_id in range(len(batch_gt_instances)):
#                     group_mask = (ins_batch == batch_id) & (ins_bids != 0)
#                     rbox = ins_rbox_preds[group_mask]
#                     label = ins_labels[group_mask]
#                     edge_loss_mask = torch.zeros_like(label, dtype=torch.bool)
#                     for c in self.edge_loss_cls:
#                         edge_loss_mask = torch.logical_or(edge_loss_mask, label == c)
#                     batched_rbox.append(rbox[edge_loss_mask])
#                 loss_bbox_edg = self.loss_bbox_edg(batched_rbox, self.edges)
            
#             loss_bbox_ovl /= len(batch_gt_instances)
#             loss_bbox_vor /= len(batch_gt_instances)
#             loss_bbox_edg /= len(batch_gt_instances)

#             # Self-Supervision (Keep logic for stability but can be weight 0)
#             loss_ss = pos_bbox_preds.sum() * 0.0
            
#             # New Losses
#             pos_batch_idxs = pos_bid_targets[:, 0].long()
#             loss_perspective = self.loss_perspective(
#                 pred_bboxes=pos_rbox_preds_geo,
#                 scores=pos_cls_scores,
#                 labels=pos_labels,
#                 batch_idxs=pos_batch_idxs,  
#                 img_metas=batch_img_metas   
#             )
#             loss_naoa = self.loss_naoa(
#                 pos_bbox_preds=pos_rbox_preds_geo,
#                 pos_scores=pos_cls_scores,
#                 pos_labels=pos_labels,
#                 batch_idxs=pos_batch_idxs
#             )

#         else:
#             loss_bbox = pos_bbox_preds.sum()
#             loss_bbox_vor = pos_bbox_preds.sum()
#             loss_bbox_ovl = pos_bbox_preds.sum()
#             loss_bbox_edg = pos_bbox_preds.sum()
#             loss_ss = pos_bbox_preds.sum()
#             # Added: Dummy Losses
#             loss_perspective = pos_bbox_preds.sum()
#             loss_naoa = pos_bbox_preds.sum()
#             loss_roi_cls = pos_bbox_preds.sum()

#         return dict(
#             # loss_cls=loss_cls, # REMOVED
#             loss_roi_cls=loss_roi_cls, # ADDED
#             loss_bbox=loss_bbox,
#             loss_bbox_vor=loss_bbox_vor,
#             loss_bbox_ovl=loss_bbox_ovl,
#             loss_bbox_edg=loss_bbox_edg,
#             loss_ss=loss_ss,
#             loss_perspective=loss_perspective,
#             loss_naoa=loss_naoa,
#             )

#     def get_targets(
#         self, points: List[Tensor], batch_gt_instances: InstanceList
#     ) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
#         """Compute regression, classification and centerness targets for points
#         in multiple images.
#         """
#         assert len(points) == len(self.regress_ranges)
#         num_levels = len(points)
#         # expand regress ranges to align with points
#         expanded_regress_ranges = [
#             points[i].new_tensor(self.regress_ranges[i])[None].expand_as(
#                 points[i]) for i in range(num_levels)
#         ]
#         # concat all levels points and regress ranges
#         concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
#         concat_points = torch.cat(points, dim=0)

#         # the number of points per img, per lvl
#         num_points = [center.size(0) for center in points]

#         # get labels and bbox_targets of each image
#         labels_list, bbox_targets_list, bid_targets_list = multi_apply(
#             self._get_targets_single,
#             batch_gt_instances,
#             points=concat_points,
#             regress_ranges=concat_regress_ranges,
#             num_points_per_lvl=num_points)

#         # split to per img, per level
#         labels_list = [labels.split(num_points, 0) for labels in labels_list]
#         bbox_targets_list = [
#             bbox_targets.split(num_points, 0)
#             for bbox_targets in bbox_targets_list
#         ]
#         bid_targets_list = [
#             bid_targets.split(num_points, 0)
#             for bid_targets in bid_targets_list
#         ]

#         # concat per level image
#         concat_lvl_labels = []
#         concat_lvl_bbox_targets = []
#         concat_lvl_bid_targets = []
#         for i in range(num_levels):
#             concat_lvl_labels.append(
#                 torch.cat([labels[i] for labels in labels_list]))
#             bbox_targets = torch.cat(
#                 [bbox_targets[i] for bbox_targets in bbox_targets_list])
#             bid_targets = torch.cat(
#                 [bid_targets[i] for bid_targets in bid_targets_list])
#             concat_lvl_bbox_targets.append(bbox_targets)
#             concat_lvl_bid_targets.append(bid_targets)
#         return (concat_lvl_labels, concat_lvl_bbox_targets,
#                 concat_lvl_bid_targets)

#     def _get_targets_single(
#             self, gt_instances: InstanceData, points: Tensor,
#             regress_ranges: Tensor,
#             num_points_per_lvl: List[int]) -> Tuple[Tensor, Tensor, Tensor]:
#         """Compute regression and classification targets for a single image."""
#         num_points = points.size(0)
#         num_gts = len(gt_instances)
#         gt_bboxes = gt_instances.bboxes
#         gt_labels = gt_instances.labels
#         gt_bids = gt_instances.bids

#         if num_gts == 0:
#             return gt_labels.new_full((num_points,), self.num_classes), \
#                    gt_bboxes.new_zeros((num_points, 4)), \
#                    gt_bids.new_zeros((num_points, 4))

#         areas = gt_bboxes.areas
#         gt_bboxes = gt_bboxes.tensor

#         # TODO: figure out why these two are different
#         # areas = areas[None].expand(num_points, num_gts)
#         areas = areas[None].repeat(num_points, 1)
#         regress_ranges = regress_ranges[:, None, :].expand(
#             num_points, num_gts, 2)
#         points = points[:, None, :].expand(num_points, num_gts, 2)
#         gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 5)
#         gt_ctr, gt_wh, gt_angle = torch.split(gt_bboxes, [2, 2, 1], dim=2)
        
#         offset = points - gt_ctr
#         w, h = gt_wh[..., 0].clone(), gt_wh[..., 1].clone()

#         center_r = torch.clamp((w * h).sqrt() / 64, 1, 5)[..., None]
#         offset_x, offset_y = offset[..., 0], offset[..., 1]
#         left = w / 2 + offset_x
#         right = w / 2 - offset_x
#         top = h / 2 + offset_y
#         bottom = h / 2 - offset_y
#         bbox_targets = torch.stack((left, top, right, bottom), -1)

#         # condition1: inside a gt bbox
#         # inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0
#         if self.center_sampling:
#             # condition1: inside a `center bbox`
#             radius = self.center_sample_radius
#             stride = offset.new_zeros(offset.shape)

#             # project the points on current lvl back to the `original` sizes
#             lvl_begin = 0
#             for lvl_idx, num_points_lvl in enumerate(num_points_per_lvl):
#                 lvl_end = lvl_begin + num_points_lvl
#                 stride[lvl_begin:lvl_end] = self.strides[lvl_idx] * radius
#                 lvl_begin = lvl_end

#             # inside_center_bbox_mask = (abs(offset) < stride * center_r).all(dim=-1)
#             # inside_gt_bbox_mask = torch.logical_and(inside_center_bbox_mask,
#             #                                         inside_gt_bbox_mask)
#             inside_gt_bbox_mask = (abs(offset) < stride * center_r).all(dim=-1)

#         # condition2: limit the regression range for each location
#         max_regress_distance = bbox_targets.max(-1)[0]
#         inside_regress_range = (
#             (max_regress_distance >= regress_ranges[..., 0])
#             & (max_regress_distance <= regress_ranges[..., 1]))

#         # if there are still more than one objects for a location,
#         # we choose the one with minimal area
#         areas[inside_gt_bbox_mask == 0] = INF
#         areas[inside_regress_range == 0] = INF
#         min_area, min_area_inds = areas.min(dim=1)

#         labels = gt_labels[min_area_inds]
#         labels[min_area == INF] = self.num_classes  # set as BG
#         bbox_targets = bbox_targets[range(num_points), min_area_inds]
#         angle_targets = gt_angle[range(num_points), min_area_inds]
#         bid_targets = gt_bids[min_area_inds]
#         bbox_targets = torch.cat((bbox_targets, angle_targets), -1)

#         return labels, bbox_targets, bid_targets

#     def predict(self,
#                 x: Tuple[Tensor],
#                 batch_data_samples: SampleList,
#                 rescale: bool = False) -> InstanceList:
#         """Perform forward propagation of the detection head and predict
#         detection results on the features of the upstream network.
        
#         【修改点】：重写 predict 逻辑，强制使用 ROI Head 进行推理评分
#         """
#         # 1. 运行 Forward 得到所有 Dense Proposals (几何信息)
#         outs = self(x) # cls_scores, bbox_preds, angle_preds
        
#         # 2. 将特征图 x 传给 predict_by_feat
#         # 我们需要在那里做 ROI Align
#         predictions = self.predict_by_feat(
#             x,  # 传入特征图
#             *outs, 
#             batch_data_samples=unpack_gt_instances(batch_data_samples), 
#             rescale=rescale
#         )
#         return predictions
    
#     def predict_by_feat(
#             self,
#             x: Tuple[Tensor], # 【修改点】：新增参数
#             cls_scores: List[Tensor],
#             bbox_preds: List[Tensor],
#             angle_preds: List[Tensor],
#             batch_data_samples: Optional[List[dict]] = None,
#             cfg: Optional[ConfigDict] = None,
#             rescale: bool = False,
#             with_nms: bool = True) -> InstanceList:
#         """Transform a batch of output features extracted from the head into
#         bbox results.
#         """
#         assert len(cls_scores) == len(bbox_preds)
#         num_levels = len(cls_scores)

#         featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
#         mlvl_priors = self.prior_generator.grid_priors(
#             featmap_sizes,
#             dtype=cls_scores[0].dtype,
#             device=cls_scores[0].device)

#         result_list = []
#         for img_id in range(len(batch_data_samples[2])):
#             data_sample = (batch_data_samples[0][img_id], batch_data_samples[1][img_id], batch_data_samples[2][img_id])
            
#             # 取出单张图的预测
#             cls_score_list = select_single_mlvl(
#                 cls_scores, img_id, detach=True)
#             bbox_pred_list = select_single_mlvl(
#                 bbox_preds, img_id, detach=True)
#             angle_pred_list = select_single_mlvl(
#                 angle_preds, img_id, detach=True)
            
#             # 【修改点】：取出单张图的特征 (目前只用 stride 8 的层 x[0])
#             # 注意：x[0] 是 [B, C, H, W]，我们需要切片出 [1, C, H, W]
#             single_feat = x[0][img_id:img_id+1]
            
#             if self.training or self.pseudo_generator:
#                 predict_by_feat = self._predict_by_feat_single_pseudo
#             else:
#                 predict_by_feat = self._predict_by_feat_single

#             results = predict_by_feat(
#                 single_feat=single_feat, # 传入特征
#                 cls_score_list=cls_score_list,
#                 bbox_pred_list=bbox_pred_list,
#                 angle_pred_list=angle_pred_list,
#                 mlvl_priors=mlvl_priors,
#                 data_sample=data_sample,
#                 cfg=cfg,
#                 rescale=rescale,
#                 with_nms=with_nms)
#             result_list.append(results)
#         return result_list
    
#     def _predict_by_feat_single_pseudo(self,
#                                 single_feat: Tensor, # 新增
#                                 cls_score_list: List[Tensor],
#                                 bbox_pred_list: List[Tensor],
#                                 angle_pred_list: List[Tensor],
#                                 mlvl_priors: List[Tensor],
#                                 data_sample: dict,
#                                 cfg: ConfigDict,
#                                 rescale: bool = False,
#                                 with_nms: bool = True) -> InstanceData:
#         """Transform a single image's features extracted from the head into
#         bbox results.
#         """        
#         # 简单起见，如果逻辑通用，直接调用 _predict_by_feat_single
#         # 如果伪标签生成有特殊逻辑（例如不需要 ROI Align），请保持原样，
#         # 但既然你已经废弃了 ConvCls，这里也应该用 ROI Align 才能得到正确分数。
#         return self._predict_by_feat_single(
#             single_feat, cls_score_list, bbox_pred_list, angle_pred_list, 
#             mlvl_priors, data_sample, cfg, rescale, with_nms)

#     def _predict_by_feat_single(self,
#                                 single_feat: Tensor, # [1, C, H, W]
#                                 cls_score_list: List[Tensor],
#                                 bbox_pred_list: List[Tensor],
#                                 angle_pred_list: List[Tensor],
#                                 mlvl_priors: List[Tensor],
#                                 data_sample: dict,
#                                 cfg: ConfigDict,
#                                 rescale: bool = False,
#                                 with_nms: bool = True) -> InstanceData:
#         """Transform a single image's features extracted from the head into
#         bbox results.
#         """
#         cfg = self.test_cfg if cfg is None else cfg
#         cfg = copy.deepcopy(cfg)
#         img_meta = data_sample[2]
#         nms_pre = cfg.get('nms_pre', -1)

#         mlvl_bbox_preds = []
#         mlvl_valid_priors = []
#         mlvl_scores = [] # 这个 scores 将是 ROI Head 的 scores
#         mlvl_labels = [] # 这个 labels 将是 ROI Head 的 labels

#         # Loop over levels (Point2RBox 通常只有 1 个 level stride=8)
#         for level_idx, (cls_score, bbox_pred, angle_pred, priors) in \
#                 enumerate(zip(cls_score_list, bbox_pred_list, angle_pred_list,
#                               mlvl_priors)):

#             assert cls_score.size()[-2:] == bbox_pred.size()[-2:]

#             # 1. 解码所有候选框 Geometry
#             bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
#             angle_pred = angle_pred.permute(1, 2, 0).reshape(
#                 -1, self.angle_coder.encode_size)
            
#             # 【注意】：我们**不能**再使用 cls_score 进行预筛选 (filter_scores_and_topk)
#             # 因为现在的 cls_score 是随机噪声。
#             # 我们必须对所有 priors 进行处理。
#             # 为了显存和速度考虑，如果 priors 太多 (>20000)，可以考虑在这里截断，
#             # 但 Point2RBox P3 层点数还好，我们全部处理。
            
#             decoded_angle = self.angle_coder.decode(angle_pred, keepdim=True)
#             rbox_pred = torch.cat((priors + bbox_pred[:, 2:], bbox_pred[:, :2] * 2, decoded_angle), -1)
            
#             # 2. 【核心修改】使用 ROI Head 进行评分
#             # 构造 ROIs: [batch_ind(0), x, y, w, h, theta]
#             batch_inds = rbox_pred.new_zeros(rbox_pred.size(0), 1)
#             rois = torch.cat([batch_inds, rbox_pred], dim=1)
            
#             # ROI Align
#             # 注意：single_feat 是 [1, C, H, W]
#             roi_feats = self.roi_align(single_feat, rois)
            
#             # MLP Inference
#             # Output: [N, 16] (15 classes + 1 bg)
#             cls_logits = self.roi_cls_head(roi_feats)
            
#             # Softmax 归一化
#             all_scores = F.softmax(cls_logits, dim=1)
            
#             # 取出前 15 类的分数 (排除背景类 index=15)
#             object_scores = all_scores[:, :-1] # [N, 15]
            
#             # 3. 筛选 Top-K (NMS Pre)
#             # 此时 object_scores 是真实的分类置信度
#             # 拿到最大分数和对应类别
#             scores, labels = object_scores.max(dim=1)
            
#             if nms_pre > 0 and scores.shape[0] > nms_pre:
#                 _, topk_inds = scores.topk(nms_pre)
#                 rbox_pred = rbox_pred[topk_inds]
#                 scores = scores[topk_inds]
#                 labels = labels[topk_inds]
#                 # 对应的 object_scores 也需要筛选，如果后续 NMS 需要完整分数矩阵的话
#                 # 但一般 Rotated NMS 只需要 max score
#                 # object_scores = object_scores[topk_inds] 

#             mlvl_bbox_preds.append(rbox_pred)
#             mlvl_valid_priors.append(priors[topk_inds] if nms_pre > 0 else priors)
#             mlvl_scores.append(scores)
#             mlvl_labels.append(labels)

#         # 合并层级结果
#         scores = torch.cat(mlvl_scores)
#         labels = torch.cat(mlvl_labels)
#         bboxes = torch.cat(mlvl_bbox_preds)
#         priors = cat_boxes(mlvl_valid_priors)

#         # Post-processing (rescale, square class handling)
#         for id in self.post_process.keys():
#             bboxes[labels == id, 2:4] *= self.post_process[id]
#         for id in self.square_cls:
#             bboxes[labels == id, -1] = 0
        
#         results = InstanceData()
#         results.bboxes = RotatedBoxes(bboxes)
#         results.scores = scores
#         results.labels = labels
        
#         # 调用父类/Mixin 的后处理 (它会处理 rescale, nms)
#         results = self._bbox_post_process(
#             results=results,
#             cfg=cfg,
#             rescale=rescale,
#             with_nms=with_nms,
#             img_meta=img_meta)

#         return results


# Copyright (c) OpenMMLab. All rights reserved.
import os, copy, math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Scale, ConvModule
from mmdet.models.dense_heads import AnchorFreeHead
from mmdet.models.utils import (filter_scores_and_topk, multi_apply,
                                select_single_mlvl, unpack_gt_instances)
from mmdet.structures import SampleList
from mmdet.structures.bbox import cat_boxes
from mmdet.utils import (ConfigType, InstanceList, OptConfigType,
                         OptInstanceList, reduce_mean)
from mmengine import ConfigDict
from mmengine.structures import InstanceData
from torch import Tensor

from mmrotate.registry import MODELS, TASK_UTILS
from mmrotate.structures import RotatedBoxes, rbox2qbox, hbox2rbox, rbox2hbox
from mmrotate.models.losses.gaussian_dist_loss import xy_wh_r_2_xy_sigma, gwd_loss
# 导入自定义的可导 ROI Align
from mmrotate.models.utils.differentiable_roi_align import DifferentiableRoIAlignRotated

INF = 1e8


@MODELS.register_module()
class Point2RBoxV2Head(AnchorFreeHead):
    """Anchor-free head used in `FCOS <https://arxiv.org/abs/1904.01355>`_.
    Compared with FCOS head, Rotated FCOS head add a angle branch to
    support rotated object detection.
    """  # noqa: E501

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 strides: list = [8],
                 regress_ranges: list = [(-1, 1e8)],
                 center_sampling: bool = True,
                 center_sample_radius: float = 0.75,
                 angle_version: str = 'le90',
                 edge_loss_start_epoch: int = 6,
                 joint_angle_start_epoch: int = 1,
                 pseudo_generator: bool = False,
                 voronoi_type: str = 'gaussian-orientation',
                 voronoi_thres: dict = dict(default=[0.994, 0.005]),
                 square_cls: list = [],
                 edge_loss_cls: list = [],
                 post_process: dict = {},
                 bbox_coder: ConfigType = dict(type='DistanceAnglePointCoder'),
                 angle_coder: ConfigType = dict(
                    type='PSCCoder',
                    angle_version='le90',
                    dual_freq=False,
                    num_step=3,
                    thr_mod=0),
                 loss_cls: ConfigType = dict(
                     type='mmdet.CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 loss_bbox: ConfigType = dict(
                     type='GWDLoss', loss_weight=5.0),
                 loss_overlap: ConfigType = dict(
                     type='GaussianOverlapLoss', loss_weight=10.0),
                 loss_voronoi: ConfigType = dict(
                     type='VoronoiWatershedLoss', loss_weight=5.0),
                 loss_bbox_edg: ConfigType = dict(
                     type='EdgeLoss', loss_weight=0.3),
                 loss_ss=dict(
                    type='Point2RBoxV2ConsistencyLoss', loss_weight=1.0),
                 # Added: Loss for Perspective Awareness
                 loss_perspective: ConfigType = dict(
                     type='PerspectiveAwareSizeConsistencyLoss', loss_weight=1.0),
                 # Added: NAOA Loss
                 loss_naoa: ConfigType = dict(
                     type='NAOALoss', loss_weight=1.0),
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 init_cfg=dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=[
                         dict(
                             type='Normal',
                             name='conv_cls',
                             std=0.01,
                             bias_prob=0.01),
                         dict(
                             type='Normal',
                             name='conv_gate',
                             std=0.01,
                             bias_prob=0.01),
                         dict(
                             type='Xavier',
                             name='roi_cls_head',
                             distribution='uniform')]),
                 **kwargs):
        self.angle_coder = TASK_UTILS.build(angle_coder)
        super().__init__(
            num_classes,
            in_channels,
            strides=strides,
            bbox_coder=bbox_coder,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            norm_cfg=norm_cfg,
            init_cfg=init_cfg,
            **kwargs)
        self.regress_ranges = regress_ranges
        self.center_sampling = center_sampling
        self.center_sample_radius = center_sample_radius
        self.angle_version = angle_version
        self.edge_loss_start_epoch = edge_loss_start_epoch
        self.joint_angle_start_epoch = joint_angle_start_epoch
        self.pseudo_generator = pseudo_generator
        self.voronoi_type = voronoi_type
        self.voronoi_thres = voronoi_thres
        self.square_cls = square_cls
        self.edge_loss_cls = edge_loss_cls
        self.post_process = post_process
        self.loss_ss = MODELS.build(loss_ss)
        self.loss_overlap = MODELS.build(loss_overlap)
        self.loss_voronoi = MODELS.build(loss_voronoi)
        self.loss_bbox_edg = MODELS.build(loss_bbox_edg)
        self.loss_perspective = MODELS.build(loss_perspective)
        self.loss_naoa = MODELS.build(loss_naoa)
        
        self.roi_align = DifferentiableRoIAlignRotated(
            output_size=(7, 7),
            spatial_scale=1.0 / self.strides[0], 
            sampling_ratio=0, 
            clockwise=True
        )
        
        self.roi_cls_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.feat_channels * 7 * 7, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.num_classes + 1) 
        )
            
    def _init_layers(self):
        super()._init_layers()
        self.conv_angle = nn.Conv2d(
            self.feat_channels, self.angle_coder.encode_size, 3, padding=1)
        self.conv_gate = nn.Conv2d(self.feat_channels, 1, 3, padding=1)
        
    def forward(
            self, x: Tuple[Tensor]
    ) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
        cls_feat = x[0]
        reg_feat = x[0]

        for cls_layer in self.cls_convs:
            cls_feat = cls_layer(cls_feat)
        cls_score = self.conv_cls(cls_feat)

        for reg_layer in self.reg_convs:
            reg_feat = reg_layer(reg_feat)
        bbox_pred = self.conv_reg(reg_feat)
        angle_pred = self.conv_angle(reg_feat)

        sig_x = bbox_pred[:, 0].exp()
        sig_y = bbox_pred[:, 1].exp()
        dx = bbox_pred[:, 2].sigmoid() * 2 - 1
        dy = bbox_pred[:, 3].sigmoid() * 2 - 1
        bbox_pred = torch.stack((sig_x, sig_y, dx, dy), 1) * 8

        return (cls_score,), (bbox_pred,), (angle_pred,)
    
    def loss_by_feat(
        self,
        x: Tuple[Tensor],
        cls_scores: List[Tensor],
        bbox_preds: List[Tensor],
        angle_preds: List[Tensor],
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        batch_gt_instances_ignore: OptInstanceList = None
    ) -> Dict[str, Tensor]:
        
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=bbox_preds[0].dtype,
            device=bbox_preds[0].device)
        
        labels, bbox_targets, bid_targets = self.get_targets(
            all_level_points, batch_gt_instances)

        num_imgs = cls_scores[0].size(0)
        
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_angle_preds = [
            angle_pred.permute(0, 2, 3, 1).reshape(-1, self.angle_coder.encode_size)
            for angle_pred in angle_preds
        ]
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_angle_preds = torch.cat(flatten_angle_preds)
        flatten_labels = torch.cat(labels)
        flatten_bbox_targets = torch.cat(bbox_targets)
        flatten_bid_targets = torch.cat(bid_targets)
        flatten_points = torch.cat(
            [points.repeat(num_imgs, 1) for points in all_level_points])

        bg_class_ind = self.num_classes
        pos_inds = ((flatten_labels >= 0) & (flatten_labels < bg_class_ind)).nonzero().reshape(-1)
        
        neg_inds = (flatten_labels == bg_class_ind).nonzero().reshape(-1)
        if len(neg_inds) > 0:
            num_neg_sample = min(len(neg_inds), max(len(pos_inds), 1) * 3)
            perm = torch.randperm(len(neg_inds), device=neg_inds.device)[:num_neg_sample]
            neg_inds_sampled = neg_inds[perm]
        else:
            neg_inds_sampled = neg_inds.new_zeros(0)

        train_inds = torch.cat([pos_inds, neg_inds_sampled])
        num_pos = torch.tensor(
            len(pos_inds), dtype=torch.float, device=bbox_preds[0].device)
        num_pos = max(reduce_mean(num_pos), 1.0)
        
        if len(train_inds) > 0:
            train_bbox_preds = flatten_bbox_preds[train_inds]
            train_angle_preds = flatten_angle_preds[train_inds]
            train_points = flatten_points[train_inds]
            train_labels = flatten_labels[train_inds]
            
            train_decoded_angle = self.angle_coder.decode(train_angle_preds, keepdim=True)
            if self.epoch < self.joint_angle_start_epoch:
                train_decoded_angle = train_decoded_angle.detach()
            
            square_mask = torch.zeros_like(train_labels, dtype=torch.bool)
            for c in self.square_cls:
                square_mask = torch.logical_or(square_mask, train_labels == c)
            train_decoded_angle[square_mask] = 0

            train_rbox_preds = torch.cat((train_points + train_bbox_preds[:, 2:], 
                                          train_bbox_preds[:, :2] * 2,
                                          train_decoded_angle), -1)
            
            train_batch_idxs = flatten_bid_targets[train_inds, 0:1]
            rois = torch.cat([train_batch_idxs, train_rbox_preds], dim=1)
            
            roi_feats = self.roi_align(x[0], rois) 
            roi_cls_preds = self.roi_cls_head(roi_feats)
            loss_roi_cls = self.loss_cls(roi_cls_preds, train_labels)
        else:
            loss_roi_cls = flatten_bbox_preds.sum() * 0.0

        pos_cls_scores = flatten_cls_scores[pos_inds].sigmoid()
        pos_labels = flatten_labels[pos_inds]
        if len(pos_inds) > 0:
            pos_cls_scores = torch.gather(pos_cls_scores, 1, pos_labels[:, None])[:, 0]
        else:
            pos_cls_scores = flatten_cls_scores.new_zeros(0)

        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_angle_preds = flatten_angle_preds[pos_inds]
        pos_bbox_targets = flatten_bbox_targets[pos_inds]
        pos_bid_targets = flatten_bid_targets[pos_inds]

        self.vis = [None] * len(batch_gt_instances)

        if len(pos_inds) > 0:
            pos_points = flatten_points[pos_inds]
            
            pos_decoded_angle_preds = self.angle_coder.decode(pos_angle_preds, keepdim=True)
            if self.epoch < self.joint_angle_start_epoch:
                pos_decoded_angle_preds = pos_decoded_angle_preds.detach()
            square_mask = torch.zeros_like(pos_labels, dtype=torch.bool)
            for c in self.square_cls:
                square_mask = torch.logical_or(square_mask, pos_labels == c)
            pos_decoded_angle_preds[square_mask] = 0

            pos_rbox_targets = self.bbox_coder.decode(pos_points, pos_bbox_targets)
            pos_rbox_preds = torch.cat((pos_points + pos_bbox_preds[:, 2:], 
                                        pos_bbox_preds[:, :2] * 2,
                                        pos_decoded_angle_preds), -1)

            cos_r = torch.cos(pos_decoded_angle_preds)
            sin_r = torch.sin(pos_decoded_angle_preds)
            R = torch.stack((cos_r, -sin_r, sin_r, cos_r), dim=-1).reshape(-1, 2, 2)
            pos_gaus_preds = R.matmul(torch.diag_embed(pos_bbox_preds[:, :2])).matmul(R.permute(0, 2, 1))

            pos_syn_mask = pos_bid_targets[:, 1] == 1
            pos_rbox_targets[~pos_syn_mask, 2:] = pos_rbox_preds[~pos_syn_mask, 2:].detach()
            loss_bbox = self.loss_bbox(pos_rbox_preds, pos_rbox_targets, avg_factor=num_pos)
            
            pos_rbox_preds_geo = torch.cat((pos_rbox_targets[:, :2], 
                                            pos_bbox_preds[:, :2] * 2,
                                            pos_decoded_angle_preds), -1)

            bid_with_view = pos_bid_targets[:, 3] + 0.5 * pos_bid_targets[:, 2]
            bid, idx = torch.unique(bid_with_view, return_inverse=True)
            ins_bid_with_view = bid.new_zeros(*bid.shape).index_reduce_(0, idx, bid_with_view, 'amin', include_self=False)
            _, bidx, bcnt = torch.unique(ins_bid_with_view.long(), return_inverse=True, return_counts=True)
            bmsk = bcnt[bidx] == 2
            
            ins_bids = pos_bid_targets.new_zeros(*bid.shape).index_reduce_(0, idx, pos_bid_targets[:, 3], 'amin', include_self=False)
            ins_batch = pos_bid_targets.new_zeros(*bid.shape).index_reduce_(0, idx, pos_bid_targets[:, 0], 'amin', include_self=False)
            ins_labels = pos_labels.new_zeros(*bid.shape).index_reduce_(0, idx, pos_labels, 'amin', include_self=False)
            
            ins_gaus_preds = pos_gaus_preds.new_zeros(*bid.shape, 4).index_reduce_(0, idx, pos_gaus_preds.view(-1, 4), 'mean', include_self=False).view(-1, 2, 2)
            ins_rbox_preds = pos_rbox_preds_geo.new_zeros(*bid.shape, pos_rbox_preds_geo.shape[-1]).index_reduce_(0, idx, pos_rbox_preds_geo, 'mean', include_self=False)
            ins_rbox_targets = pos_rbox_targets.new_zeros(*bid.shape, pos_rbox_targets.shape[-1]).index_reduce_(0, idx, pos_rbox_targets, 'mean', include_self=False)

            ori_mu_all = ins_rbox_targets[:, 0:2]
            loss_bbox_ovl = ori_mu_all.new_tensor(0)
            loss_bbox_vor = ori_mu_all.new_tensor(0)
            
            for batch_id in range(len(batch_gt_instances)):
                group_mask = (ins_batch == batch_id) & (ins_bids != 0)
                mu = ori_mu_all[group_mask]
                sigma = ins_gaus_preds[group_mask]
                label = ins_labels[group_mask]
                if len(mu) >= 2:
                    loss_bbox_ovl += self.loss_overlap((mu, sigma.bmm(sigma)))
                if len(mu) >= 1:
                    pos_thres = [self.voronoi_thres['default'][0]] * self.num_classes
                    neg_thres = [self.voronoi_thres['default'][1]] * self.num_classes
                    if 'override' in self.voronoi_thres.keys():
                        for item in self.voronoi_thres['override']:
                            for cls in item[0]:
                                pos_thres[cls] = item[1][0]
                                neg_thres[cls] = item[1][1]
                    loss_bbox_vor += self.loss_voronoi((mu, sigma.bmm(sigma)),
                                                        label, self.images[batch_id],
                                                        pos_thres, neg_thres,
                                                        voronoi=self.voronoi_type)
                    self.vis[batch_id] = self.loss_voronoi.vis
            
            loss_bbox_edg = ori_mu_all.new_tensor(0)
            if self.epoch >= self.edge_loss_start_epoch:
                batched_rbox = []
                for batch_id in range(len(batch_gt_instances)):
                    group_mask = (ins_batch == batch_id) & (ins_bids != 0)
                    rbox = ins_rbox_preds[group_mask]
                    label = ins_labels[group_mask]
                    edge_loss_mask = torch.zeros_like(label, dtype=torch.bool)
                    for c in self.edge_loss_cls:
                        edge_loss_mask = torch.logical_or(edge_loss_mask, label == c)
                    batched_rbox.append(rbox[edge_loss_mask])
                loss_bbox_edg = self.loss_bbox_edg(batched_rbox, self.edges)
            
            loss_bbox_ovl /= len(batch_gt_instances)
            loss_bbox_vor /= len(batch_gt_instances)
            loss_bbox_edg /= len(batch_gt_instances)

            loss_ss = pos_bbox_preds.sum() * 0.0
            
            pos_batch_idxs = pos_bid_targets[:, 0].long()
            loss_perspective = self.loss_perspective(
                pred_bboxes=pos_rbox_preds_geo,
                scores=pos_cls_scores,
                labels=pos_labels,
                batch_idxs=pos_batch_idxs,  
                img_metas=batch_img_metas   
            )
            loss_naoa = self.loss_naoa(
                pos_bbox_preds=pos_rbox_preds_geo,
                pos_scores=pos_cls_scores,
                pos_labels=pos_labels,
                batch_idxs=pos_batch_idxs
            )

        else:
            loss_bbox = pos_bbox_preds.sum()
            loss_bbox_vor = pos_bbox_preds.sum()
            loss_bbox_ovl = pos_bbox_preds.sum()
            loss_bbox_edg = pos_bbox_preds.sum()
            loss_ss = pos_bbox_preds.sum()
            loss_perspective = pos_bbox_preds.sum()
            loss_naoa = pos_bbox_preds.sum()
            loss_roi_cls = pos_bbox_preds.sum()

        return dict(
            loss_roi_cls=loss_roi_cls,
            loss_bbox=loss_bbox,
            loss_bbox_vor=loss_bbox_vor,
            loss_bbox_ovl=loss_bbox_ovl,
            loss_bbox_edg=loss_bbox_edg,
            loss_ss=loss_ss,
            loss_perspective=loss_perspective,
            loss_naoa=loss_naoa,
            )

    # 【关键修复】：手动切片替代 split，确保 labels_list 结构稳定
    def get_targets(
        self, points: List[Tensor], batch_gt_instances: InstanceList
    ) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
        """Compute regression, classification and centerness targets for points."""
        assert len(points) == len(self.regress_ranges)
        num_levels = len(points)
        expanded_regress_ranges = [
            points[i].new_tensor(self.regress_ranges[i])[None].expand_as(
                points[i]) for i in range(num_levels)
        ]
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(points, dim=0)
        
        # num_points 是每层的点数列表，例如 [16384]
        num_points = [center.size(0) for center in points]

        # labels_list[img_id] 是该图片所有层的 cat 后的 tensor
        labels_list, bbox_targets_list, bid_targets_list = multi_apply(
            self._get_targets_single,
            batch_gt_instances,
            points=concat_points,
            regress_ranges=concat_regress_ranges,
            num_points_per_lvl=num_points)

        # 【手动切片逻辑】
        # 1. 初始化每个 Level 的列表
        concat_lvl_labels = [[] for _ in range(num_levels)]
        concat_lvl_bbox_targets = [[] for _ in range(num_levels)]
        concat_lvl_bid_targets = [[] for _ in range(num_levels)]

        # 2. 遍历每张图片的 Target
        for i in range(len(labels_list)):
            img_labels = labels_list[i]
            img_bbox_targets = bbox_targets_list[i]
            img_bid_targets = bid_targets_list[i]
            
            start_idx = 0
            # 3. 按 Level 切片
            for lvl in range(num_levels):
                end_idx = start_idx + num_points[lvl]
                
                # 切片并添加到对应 Level 的列表中
                concat_lvl_labels[lvl].append(img_labels[start_idx:end_idx])
                concat_lvl_bbox_targets[lvl].append(img_bbox_targets[start_idx:end_idx])
                concat_lvl_bid_targets[lvl].append(img_bid_targets[start_idx:end_idx])
                
                start_idx = end_idx

        # 4. 对每个 Level 进行 Batch 维度的 Concat
        final_labels = []
        final_bbox_targets = []
        final_bid_targets = []
        
        for lvl in range(num_levels):
            final_labels.append(torch.cat(concat_lvl_labels[lvl]))
            final_bbox_targets.append(torch.cat(concat_lvl_bbox_targets[lvl]))
            final_bid_targets.append(torch.cat(concat_lvl_bid_targets[lvl]))

        return (final_labels, final_bbox_targets, final_bid_targets)

    def _get_targets_single(
            self, gt_instances: InstanceData, points: Tensor,
            regress_ranges: Tensor,
            num_points_per_lvl: List[int]) -> Tuple[Tensor, Tensor, Tensor]:
        """Compute regression and classification targets for a single image."""
        num_points = points.size(0)
        num_gts = len(gt_instances)
        gt_bboxes = gt_instances.bboxes
        gt_labels = gt_instances.labels
        gt_bids = gt_instances.bids

        if num_gts == 0:
            return gt_labels.new_full((num_points,), self.num_classes), \
                   gt_bboxes.new_zeros((num_points, 4)), \
                   gt_bids.new_zeros((num_points, 4))

        areas = gt_bboxes.areas
        gt_bboxes = gt_bboxes.tensor

        areas = areas[None].repeat(num_points, 1)
        regress_ranges = regress_ranges[:, None, :].expand(
            num_points, num_gts, 2)
        points = points[:, None, :].expand(num_points, num_gts, 2)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 5)
        gt_ctr, gt_wh, gt_angle = torch.split(gt_bboxes, [2, 2, 1], dim=2)
        
        offset = points - gt_ctr
        w, h = gt_wh[..., 0].clone(), gt_wh[..., 1].clone()

        center_r = torch.clamp((w * h).sqrt() / 64, 1, 5)[..., None]
        offset_x, offset_y = offset[..., 0], offset[..., 1]
        left = w / 2 + offset_x
        right = w / 2 - offset_x
        top = h / 2 + offset_y
        bottom = h / 2 - offset_y
        bbox_targets = torch.stack((left, top, right, bottom), -1)

        if self.center_sampling:
            radius = self.center_sample_radius
            stride = offset.new_zeros(offset.shape)
            lvl_begin = 0
            for lvl_idx, num_points_lvl in enumerate(num_points_per_lvl):
                lvl_end = lvl_begin + num_points_lvl
                stride[lvl_begin:lvl_end] = self.strides[lvl_idx] * radius
                lvl_begin = lvl_end
            inside_gt_bbox_mask = (abs(offset) < stride * center_r).all(dim=-1)

        max_regress_distance = bbox_targets.max(-1)[0]
        inside_regress_range = (
            (max_regress_distance >= regress_ranges[..., 0])
            & (max_regress_distance <= regress_ranges[..., 1]))

        areas[inside_gt_bbox_mask == 0] = INF
        areas[inside_regress_range == 0] = INF
        min_area, min_area_inds = areas.min(dim=1)

        labels = gt_labels[min_area_inds]
        labels[min_area == INF] = self.num_classes
        bbox_targets = bbox_targets[range(num_points), min_area_inds]
        angle_targets = gt_angle[range(num_points), min_area_inds]
        bid_targets = gt_bids[min_area_inds]
        bbox_targets = torch.cat((bbox_targets, angle_targets), -1)

        return labels, bbox_targets, bid_targets

    def predict(self,
                x: Tuple[Tensor],
                batch_data_samples: SampleList,
                rescale: bool = False) -> InstanceList:
        outs = self(x)
        predictions = self.predict_by_feat(
            x, 
            *outs, 
            batch_data_samples=unpack_gt_instances(batch_data_samples), 
            rescale=rescale
        )
        return predictions
    
    def predict_by_feat(
            self,
            x: Tuple[Tensor],
            cls_scores: List[Tensor],
            bbox_preds: List[Tensor],
            angle_preds: List[Tensor],
            batch_data_samples: Optional[List[dict]] = None,
            cfg: Optional[ConfigDict] = None,
            rescale: bool = False,
            with_nms: bool = True) -> InstanceList:
        
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)

        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes, dtype=cls_scores[0].dtype, device=cls_scores[0].device)

        result_list = []
        for img_id in range(len(batch_data_samples[2])):
            data_sample = (batch_data_samples[0][img_id], batch_data_samples[1][img_id], batch_data_samples[2][img_id])
            
            cls_score_list = select_single_mlvl(cls_scores, img_id, detach=True)
            bbox_pred_list = select_single_mlvl(bbox_preds, img_id, detach=True)
            angle_pred_list = select_single_mlvl(angle_preds, img_id, detach=True)
            
            single_feat = x[0][img_id:img_id+1]

            if self.training or self.pseudo_generator:
                predict_by_feat = self._predict_by_feat_single_pseudo
            else:
                predict_by_feat = self._predict_by_feat_single

            results = predict_by_feat(
                single_feat=single_feat, 
                cls_score_list=cls_score_list,
                bbox_pred_list=bbox_pred_list,
                angle_pred_list=angle_pred_list,
                mlvl_priors=mlvl_priors,
                data_sample=data_sample,
                cfg=cfg,
                rescale=rescale,
                with_nms=with_nms)
            result_list.append(results)
        return result_list

    def _predict_by_feat_single(self,
                                single_feat: Tensor, 
                                cls_score_list: List[Tensor],
                                bbox_pred_list: List[Tensor],
                                angle_pred_list: List[Tensor],
                                mlvl_priors: List[Tensor],
                                data_sample: dict,
                                cfg: ConfigDict,
                                rescale: bool = False,
                                with_nms: bool = True) -> InstanceData:
        
        cfg = self.test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg)
        img_meta = data_sample[2]
        nms_pre = cfg.get('nms_pre', -1)

        mlvl_bbox_preds = []
        mlvl_valid_priors = []
        mlvl_scores = [] 
        mlvl_labels = [] 

        for level_idx, (cls_score, bbox_pred, angle_pred, priors) in enumerate(zip(cls_score_list, bbox_pred_list, angle_pred_list, mlvl_priors)):
            
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            angle_pred = angle_pred.permute(1, 2, 0).reshape(-1, self.angle_coder.encode_size)
            
            decoded_angle = self.angle_coder.decode(angle_pred, keepdim=True)
            rbox_pred = torch.cat((priors + bbox_pred[:, 2:], bbox_pred[:, :2] * 2, decoded_angle), -1)
            
            batch_inds = rbox_pred.new_zeros(rbox_pred.size(0), 1)
            rois = torch.cat([batch_inds, rbox_pred], dim=1)
            
            roi_feats = self.roi_align(single_feat, rois)
            cls_logits = self.roi_cls_head(roi_feats)
            all_scores = F.softmax(cls_logits, dim=1)
            object_scores = all_scores[:, :-1]
            
            scores, labels = object_scores.max(dim=1)
            
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                _, topk_inds = scores.topk(nms_pre)
                rbox_pred = rbox_pred[topk_inds]
                scores = scores[topk_inds]
                labels = labels[topk_inds]
                # object_scores = object_scores[topk_inds] 

            mlvl_bbox_preds.append(rbox_pred)
            mlvl_scores.append(scores)
            mlvl_labels.append(labels) 

        scores = torch.cat(mlvl_scores)
        bboxes = torch.cat(mlvl_bbox_preds)
        
        results = InstanceData()
        results.bboxes = RotatedBoxes(bboxes)
        results.scores = scores
        results.labels = torch.arange(len(scores), device=scores.device)
        
        results = self._bbox_post_process(
            results=results,
            cfg=cfg,
            rescale=rescale,
            with_nms=with_nms,
            img_meta=img_meta)

        return results

    def _predict_by_feat_single_pseudo(self, *args, **kwargs):
        return self._predict_by_feat_single(*args, **kwargs)