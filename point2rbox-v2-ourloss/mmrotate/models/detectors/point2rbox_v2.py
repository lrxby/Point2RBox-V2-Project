# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Tuple, Union

import torch
from torch import Tensor

from mmrotate.registry import MODELS
from mmdet.models.detectors.single_stage import SingleStageDetector
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmdet.structures import DetDataSample, SampleList
from mmengine.structures import InstanceData

@MODELS.register_module()
class Point2RBoxV2(SingleStageDetector):
    """
    Simplified Point2RBoxV2 Detector.
    """

    def __init__(self,
                 backbone: ConfigType,
                 neck: ConfigType,
                 bbox_head: ConfigType,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)
        
        # 初始化 epoch 属性，防止第一次调用前不存在
        self.epoch = 0

    # [FIXED] 必须保留这个方法，因为 SetEpochInfoHook 会调用它
    def set_epoch(self, epoch):
        self.epoch = epoch
        # 将 epoch 信息传递给 head (如果 head 需要根据 epoch 调整逻辑)
        if hasattr(self.bbox_head, 'set_epoch'):
            self.bbox_head.set_epoch(epoch)
        else:
            # 即使 Head 没有定义 set_epoch 方法，直接设置属性也是 Python 允许的
            self.bbox_head.epoch = epoch

    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        """Extract features."""
        x = self.backbone(batch_inputs)
        if self.with_neck:
            x = self.neck(x)
        return x

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: List['DetDataSample']) -> Union[Dict, List]:
        x = self.extract_feat(batch_inputs)
        losses = self.bbox_head.loss(x, batch_data_samples)
        return losses

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        x = self.extract_feat(batch_inputs)
        results_list = self.bbox_head.predict(
            x, batch_data_samples, rescale=rescale)
        
        predictions = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return predictions