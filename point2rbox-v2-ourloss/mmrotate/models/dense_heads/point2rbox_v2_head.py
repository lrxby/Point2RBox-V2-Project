# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from mmdet.models.dense_heads import AnchorFreeHead
from mmdet.models.utils import (filter_scores_and_topk, multi_apply,
                                select_single_mlvl, unpack_gt_instances)
from mmdet.structures import SampleList
from mmdet.utils import (ConfigType, InstanceList, OptInstanceList, reduce_mean)
from mmengine.structures import InstanceData
from torch import Tensor

from mmrotate.registry import MODELS, TASK_UTILS
from mmrotate.structures import RotatedBoxes
# 确保引用路径正确
from mmrotate.models.utils.differentiable_roi_align import DifferentiableRoIAlignRotated

INF = 1e8

@MODELS.register_module()
class Point2RBoxV2Head(AnchorFreeHead):
    """
    Point2RBoxV2Head: Prior Box + Multi-scale Fusion + Center Sampling.
    """

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 strides: list = [8], 
                 regress_ranges: list = [(-1, 1e8)],
                 center_sampling: bool = True,
                 center_sample_radius: float = 1.5,
                 angle_version: str = 'le90',
                 # [NEW] Prior Box Size (Config: [28.0, 66.0])
                 prior_box_size: list = [32.0, 32.0], 
                 # Loss configs
                 loss_cls: ConfigType = dict(
                     type='mmdet.FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox: ConfigType = dict(
                     type='GWDLoss', loss_weight=5.0),
                 # Geometry Losses
                 loss_perspective: ConfigType = None,
                 loss_naoa: ConfigType = None,
                 
                 bbox_coder: ConfigType = dict(type='DistanceAnglePointCoder'),
                 angle_coder: ConfigType = dict(
                    type='PSCCoder',
                    angle_version='le90',
                    dual_freq=False,
                    num_step=3,
                    thr_mod=0),
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 init_cfg=dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=[
                         dict(type='Normal', name='conv_reg', std=0.01)
                     ]),
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
        
        # [NEW] Register prior size as buffer
        self.register_buffer('prior_box_size', torch.tensor(prior_box_size))

        # Losses
        self.loss_perspective = MODELS.build(loss_perspective) if loss_perspective else None
        self.loss_naoa = MODELS.build(loss_naoa) if loss_naoa else None
        
        # Clean up unused attributes
        self.loss_overlap = None
        self.loss_centerness = None 
        self.loss_voronoi = None
        self.loss_bbox_edg = None
        self.loss_ss = None

        # ====================================================================
        # [核心组件] Multi-scale Feature Fusion
        # ====================================================================
        # 1. 多层提取器 (P3, P4, P5)
        self.roi_extractors = nn.ModuleList([
            DifferentiableRoIAlignRotated(
                output_size=(7, 7),
                spatial_scale=1.0 / (strides[0] * (2 ** i)), 
                sampling_ratio=2,
                clockwise=True 
            ) for i in range(3)
        ])
        
        # 2. 分类器：全连接层 (输入维度 x3)
        self.cls_fc = nn.Sequential(
            nn.Linear(in_channels * 7 * 7 * 3, 256), 
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)
        )

    def _init_layers(self):
        super()._init_layers()
        self.conv_reg = nn.Conv2d(self.feat_channels, 4, 3, padding=1)
        self.conv_angle = nn.Conv2d(self.feat_channels, self.angle_coder.encode_size, 3, padding=1)
    
    def init_weights(self):
        super().init_weights()
        # [关键] 强制 0 初始化
        nn.init.constant_(self.conv_reg.weight, 0)
        nn.init.constant_(self.conv_reg.bias, 0)
        
    def forward(self, x: Tuple[Tensor]) -> Tuple[List[Tensor]]:
        feats = x 
        reg_feat = feats[0] # P3 Only
        
        for reg_layer in self.reg_convs:
            reg_feat = reg_layer(reg_feat)
        
        bbox_deltas = self.conv_reg(reg_feat) 
        angle_pred_raw = self.conv_angle(reg_feat)
        
        # [关键] Safe Decode Logic
        dx = bbox_deltas[:, 0].sigmoid() * 4 - 2 
        dy = bbox_deltas[:, 1].sigmoid() * 4 - 2 
        
        prior_w, prior_h = self.prior_box_size
        
        scale_factor = 0.1
        dw = bbox_deltas[:, 2].clamp(-10, 10) * scale_factor
        dh = bbox_deltas[:, 3].clamp(-10, 10) * scale_factor
        
        pred_w = prior_w * dw.exp()
        pred_h = prior_h * dh.exp()
        
        bbox_pred = torch.stack((pred_w, pred_h, dx, dy), 1)

        return (feats,), (bbox_pred,), (angle_pred_raw,)

    def loss(self, x: Tuple[Tensor], batch_data_samples: SampleList) -> dict:
        outs = self(x)
        outputs = unpack_gt_instances(batch_data_samples)
        batch_gt_instances, batch_gt_instances_ignore, batch_img_metas = outputs

        loss_inputs = outs + (batch_gt_instances, batch_img_metas, batch_gt_instances_ignore)
        losses = self.loss_by_feat(*loss_inputs)
        return losses

    def loss_by_feat(
        self,
        feats_tuple: Tuple[List[Tensor]], 
        bbox_preds: List[Tensor],
        angle_preds: List[Tensor],
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        batch_gt_instances_ignore: OptInstanceList = None
    ) -> Dict[str, Tensor]:
        
        fpn_feats = feats_tuple[0] # [P3, P4, P5]
        bbox_pred = bbox_preds[0]  # P3 pred
        angle_pred = angle_preds[0] # P3 pred
        
        # 1. Grid Points
        featmap_size = fpn_feats[0].size()[-2:]
        points = self.prior_generator.single_level_grid_priors(
            featmap_size, level_idx=0, dtype=fpn_feats[0].dtype, device=fpn_feats[0].device)
        
        # 2. Get Targets
        cls_reg_targets = self.get_targets([points], batch_gt_instances)
        (labels_list, bbox_targets_list, _) = cls_reg_targets
        
        B, C, H, W = fpn_feats[0].shape
        N = H * W
        
        # Flatten Predictions
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(B, N, 4)
        angle_pred = angle_pred.permute(0, 2, 3, 1).reshape(B, N, -1)
        decoded_angle = self.angle_coder.decode(angle_pred, keepdim=True)
        
        # Reconstruct Boxes
        wh_preds = bbox_pred[..., :2].clamp(min=1.0, max=4096.0)
        offset_preds = bbox_pred[..., 2:] * self.strides[0] 
        center_preds = points.unsqueeze(0).expand(B, N, 2) + offset_preds
        rbox_preds = torch.cat([center_preds, wh_preds, decoded_angle], dim=-1)

        # Flatten Targets
        flatten_labels = torch.cat(labels_list) 
        flatten_bbox_targets = torch.cat(bbox_targets_list) 
        flatten_rbox_preds = rbox_preds.reshape(-1, 5) 
        
        # 3. Sampling
        pos_inds = torch.nonzero(
            (flatten_labels >= 0) & (flatten_labels < self.num_classes), 
            as_tuple=False).squeeze(1)
        
        neg_inds = torch.nonzero(
            flatten_labels == self.num_classes, 
            as_tuple=False).squeeze(1)
        
        num_pos = len(pos_inds)
        num_neg = min(len(neg_inds), max(num_pos * 5, 2000))
        if num_neg > 0:
            perm = torch.randperm(len(neg_inds), device=neg_inds.device)[:num_neg]
            neg_inds = neg_inds[perm]
        sample_inds = torch.cat([pos_inds, neg_inds])
        
        # 4. Multi-scale Extraction & Classification
        if len(sample_inds) > 0:
            batch_inds = (sample_inds // N).float().unsqueeze(1)
            sampled_rboxes = flatten_rbox_preds[sample_inds]
            rois = torch.cat([batch_inds, sampled_rboxes], dim=1)
            
            multi_scale_feats = []
            for i, feat_level in enumerate(fpn_feats):
                if i >= 3: break 
                roi_feat_i = self.roi_extractors[i](feat_level, rois)
                roi_feat_i_flat = roi_feat_i.view(roi_feat_i.size(0), -1)
                multi_scale_feats.append(roi_feat_i_flat)
            
            fused_feats = torch.cat(multi_scale_feats, dim=1)
            cls_scores = self.cls_fc(fused_feats)
            targets = flatten_labels[sample_inds]
            
            loss_cls = self.loss_cls(
                cls_scores, 
                targets, 
                avg_factor=max(num_pos, 1.0)
            )
        else:
            loss_cls = bbox_pred.sum() * 0.0

        # 5. Regression & Geometry Losses (Only Positive)
        if num_pos > 0:
            pos_rbox_preds = flatten_rbox_preds[pos_inds]
            
            # Decode Targets (l,t,r,b -> cx, cy, w, h)
            pos_bbox_targets_ltrb = flatten_bbox_targets[pos_inds]
            pos_points = points.unsqueeze(0).expand(B, N, 2).reshape(-1, 2)[pos_inds]
            
            tgt_w = pos_bbox_targets_ltrb[:, 0] + pos_bbox_targets_ltrb[:, 2]
            tgt_h = pos_bbox_targets_ltrb[:, 1] + pos_bbox_targets_ltrb[:, 3]
            tgt_cx = pos_points[:, 0] - (pos_bbox_targets_ltrb[:, 0] - tgt_w/2)
            tgt_cy = pos_points[:, 1] - (pos_bbox_targets_ltrb[:, 1] - tgt_h/2)
            tgt_a = pos_bbox_targets_ltrb[:, 4]
            
            pos_rbox_targets = torch.stack([tgt_cx, tgt_cy, tgt_w, tgt_h, tgt_a], dim=1)
            
            loss_bbox = self.loss_bbox(
                pos_rbox_preds,
                pos_rbox_targets,
                avg_factor=num_pos
            )
            
            force_scores = torch.ones_like(flatten_labels[pos_inds], dtype=torch.float32)

            loss_perspective = bbox_pred.sum() * 0.0
            if self.loss_perspective is not None:
                pos_batch_idx = (pos_inds // N)
                loss_perspective = self.loss_perspective(
                    pred_bboxes=pos_rbox_preds, 
                    scores=force_scores, 
                    labels=flatten_labels[pos_inds],
                    batch_idxs=pos_batch_idx,
                    img_metas=batch_img_metas
                )
                
            loss_naoa = bbox_pred.sum() * 0.0
            if self.loss_naoa is not None:
                pos_batch_idx = (pos_inds // N)
                loss_naoa = self.loss_naoa(
                    pos_bbox_preds=pos_rbox_preds,
                    pos_scores=force_scores, 
                    pos_labels=flatten_labels[pos_inds],
                    batch_idxs=pos_batch_idx
                )

        else:
            loss_bbox = bbox_pred.sum() * 0.0
            loss_perspective = bbox_pred.sum() * 0.0
            loss_naoa = bbox_pred.sum() * 0.0

        return dict(
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_perspective=loss_perspective,
            loss_naoa=loss_naoa
        )

    # =================================================================
    # [推理逻辑]
    # =================================================================
    def predict_by_feat(self, feats_tuple, bbox_preds, angle_preds, 
                        batch_img_metas=None, cfg=None, rescale=False, with_nms=True):
        if cfg is None: cfg = self.test_cfg
        
        fpn_feats = feats_tuple[0]
        bbox_pred = bbox_preds[0]
        angle_pred = angle_preds[0]
        
        featmap_size = fpn_feats[0].size()[-2:]
        points = self.prior_generator.single_level_grid_priors(
            featmap_size, level_idx=0, dtype=fpn_feats[0].dtype, device=fpn_feats[0].device)
        
        results_list = []
        for img_id in range(len(batch_img_metas)):
            img_bbox = bbox_pred[img_id].permute(1, 2, 0).reshape(-1, 4)
            img_angle = angle_pred[img_id].permute(1, 2, 0).reshape(-1, self.angle_coder.encode_size)
            
            decoded_angle = self.angle_coder.decode(img_angle, keepdim=True)
            wh_preds = img_bbox[:, :2].clamp(min=1.0, max=4096.0)
            offset_preds = img_bbox[:, 2:] * self.strides[0]
            center_preds = points + offset_preds
            rbox_preds = torch.cat([center_preds, wh_preds, decoded_angle], dim=-1)
            
            batch_inds = torch.zeros((rbox_preds.size(0), 1), device=rbox_preds.device)
            rois = torch.cat([batch_inds, rbox_preds], dim=1)
            
            multi_scale_feats = []
            img_feats = [f[img_id].unsqueeze(0) for f in fpn_feats[:3]]
            for i, feat_level in enumerate(img_feats):
                roi_feat_i = self.roi_extractors[i](feat_level, rois)
                roi_feat_i_flat = roi_feat_i.view(roi_feat_i.size(0), -1)
                multi_scale_feats.append(roi_feat_i_flat)
            
            fused_feats = torch.cat(multi_scale_feats, dim=1)
            cls_scores = self.cls_fc(fused_feats).sigmoid()
            
            padding = cls_scores.new_zeros(cls_scores.shape[0], 1)
            cls_scores_pad = torch.cat([cls_scores, padding], dim=1)
            
            results = filter_scores_and_topk(
                cls_scores_pad, cfg.get('score_thr', 0.05), cfg.get('nms_pre', 2000), dict(bboxes=rbox_preds))
            
            scores, labels, _, filtered_results = results
            bboxes = filtered_results['bboxes']
            
            if rescale:
                scale_factor = batch_img_metas[img_id]['scale_factor']
                bboxes[:, :4] /= bboxes.new_tensor(scale_factor).repeat(2)
            
            res = InstanceData()
            res.bboxes = RotatedBoxes(bboxes)
            res.scores = scores
            res.labels = labels
            results_list.append(res)
            
        return results_list

    # =================================================================
    # [LEGACY] 正样本分配逻辑 (Center Sampling)
    # =================================================================
    def get_targets(
        self, points: List[Tensor], batch_gt_instances: InstanceList
    ) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
        assert len(points) == len(self.regress_ranges)
        num_levels = len(points)
        expanded_regress_ranges = [
            points[i].new_tensor(self.regress_ranges[i])[None].expand_as(
                points[i]) for i in range(num_levels)
        ]
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(points, dim=0)
        num_points = [center.size(0) for center in points]
        labels_list, bbox_targets_list, bid_targets_list = multi_apply(
            self._get_targets_single,
            batch_gt_instances,
            points=concat_points,
            regress_ranges=concat_regress_ranges,
            num_points_per_lvl=num_points)
        labels_list = [labels.split(num_points, 0) for labels in labels_list]
        bbox_targets_list = [
            bbox_targets.split(num_points, 0)
            for bbox_targets in bbox_targets_list
        ]
        bid_targets_list = [
            bid_targets.split(num_points, 0)
            for bid_targets in bid_targets_list
        ]
        concat_lvl_labels = []
        concat_lvl_bbox_targets = []
        concat_lvl_bid_targets = []
        for i in range(num_levels):
            concat_lvl_labels.append(
                torch.cat([labels[i] for labels in labels_list]))
            bbox_targets = torch.cat(
                [bbox_targets[i] for bbox_targets in bbox_targets_list])
            bid_targets = torch.cat(
                [bid_targets[i] for bid_targets in bid_targets_list])
            concat_lvl_bbox_targets.append(bbox_targets)
            concat_lvl_bid_targets.append(bid_targets)
        return (concat_lvl_labels, concat_lvl_bbox_targets,
                concat_lvl_bid_targets)

    # [FIX] 缩进修正，现在它是类的方法了
    def _get_targets_single(
            self, gt_instances: InstanceData, points: Tensor,
            regress_ranges: Tensor,
            num_points_per_lvl: List[int]) -> Tuple[Tensor, Tensor, Tensor]:
        num_points = points.size(0)
        num_gts = len(gt_instances)
        gt_bboxes = gt_instances.bboxes
        gt_labels = gt_instances.labels
        
        # [FIX] Bids 兼容性修复
        if hasattr(gt_instances, 'bids'):
            gt_bids = gt_instances.bids
        else:
            gt_bids = gt_labels.new_zeros((num_gts, 4))

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