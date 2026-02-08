# Copyright (c) OpenMMLab. All rights reserved.
import glob
import os.path as osp
from typing import List
import json

from mmengine.dataset import BaseDataset
from mmrotate.registry import DATASETS


@DATASETS.register_module()
class CODRONEDataset(BaseDataset):
    """CODrone dataset for detection.

    Format: x1 y1 x2 y2 x3 y3 x4 y4 class_name difficulty
    Example: 54.0 156.0 ... 185.0 boat 0
    """

    METAINFO = {
        'classes':
        ('car', 'truck', 'bus', 'traffic-light',
         'traffic-sign', 'bridge', 'people', 'bicycle',
         'motor', 'tricycle', 'boat', 'ship'),
        # palette is a list of color tuples, which is used for visualization.
        'palette': [
            (165, 42, 42), (189, 183, 107), (0, 255, 0), (255, 0, 0),
            (138, 43, 226), (255, 128, 0), (255, 0, 255), (0, 255, 255),
            (255, 193, 193), (0, 51, 153), (255, 250, 205), (0, 139, 139)
        ]
    }

    def __init__(self,
                 diff_thr: int = 100,
                 img_suffix: str = 'jpg',  # 注意：通常 DOTA 类数据集可能是 png，请根据实际情况确认
                 **kwargs) -> None:
        self.diff_thr = diff_thr
        self.img_suffix = img_suffix
        super().__init__(**kwargs)

    def load_data_list(self) -> List[dict]:
        """Load annotations from an annotation file named as ``self.ann_file``"""
        # 建立 字符串 -> ID 的映射 (e.g., 'boat': 10)
        cls_map = {c: i for i, c in enumerate(self.metainfo['classes'])}
        data_list = []

        # 情况1: 如果 ann_file 为空 (通常用于 test 模式)
        if self.ann_file == '':
            img_files = glob.glob(
                osp.join(self.data_prefix['img_path'], f'*.{self.img_suffix}'))
            for img_path in img_files:
                data_info = {}
                data_info['img_path'] = img_path
                img_name = osp.split(img_path)[1]
                data_info['file_name'] = img_name
                img_id = img_name[:-len(self.img_suffix)-1] # 去掉后缀和点
                data_info['img_id'] = img_id
                
                # 生成一个空的实例占位符
                instance = dict(bbox=[], bbox_label=[], ignore_flag=0)
                data_info['instances'] = [instance]
                data_list.append(data_info)

        # 情况2: 标准的 TXT 标注文件读取
        else:
            txt_files = glob.glob(osp.join(self.ann_file, '*.txt'))
            if len(txt_files) == 0:
                raise ValueError(f'No .txt files found in {self.ann_file}')
            
            for txt_file in txt_files:
                data_info = {}
                img_id = osp.split(txt_file)[1][:-4]
                data_info['img_id'] = img_id
                img_name = img_id + f'.{self.img_suffix}'
                data_info['file_name'] = img_name
                data_info['img_path'] = osp.join(self.data_prefix['img_path'], img_name)

                instances = []
                with open(txt_file, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        parts = line.strip().split()
                        
                        # 1. 基础格式检查：至少要有9列 (8坐标 + 1类别)
                        # 标准格式应该是10列 (加难度)，但为了健壮性，允许缺省难度
                        if len(parts) < 9:
                            continue

                        try:
                            # 解析坐标
                            bbox = [float(x) for x in parts[:8]]
                            # 解析类别名称 (字符串)
                            cls_name = parts[8]
                        except ValueError:
                            continue

                        # 过滤不在类别表中的物体
                        if cls_name not in cls_map:
                            continue
                        
                        bbox_label = cls_map[cls_name]

                        # 解析难度 (如果有第10列)
                        if len(parts) > 9:
                            try:
                                difficulty = int(parts[9])
                            except ValueError:
                                difficulty = 0
                        else:
                            difficulty = 0

                        # 难度过滤
                        ignore_flag = 1 if difficulty > self.diff_thr else 0

                        # ==========================================
                        # 2. 【核心修复】防崩溃检查
                        # 过滤掉极其微小的噪点框，防止 Point2RBox 崩溃
                        # ==========================================
                        xs = bbox[0::2]
                        ys = bbox[1::2]
                        w = max(xs) - min(xs)
                        h = max(ys) - min(ys)
                        if w < 1 or h < 1: 
                            continue

                        instance = {
                            'bbox': bbox,
                            'bbox_label': bbox_label,
                            'ignore_flag': ignore_flag
                        }
                        instances.append(instance)

                # 只有当包含有效数据或训练模式下才加入
                # 为了防止空图报错，通常保留空图但 instances 为空列表即可，MMDet能处理
                data_info['instances'] = instances
                data_list.append(data_info)

        return data_list

    def filter_data(self) -> List[dict]:
        """Filter annotations according to filter_cfg."""
        if self.test_mode:
            return self.data_list

        filter_empty_gt = self.filter_cfg.get('filter_empty_gt', False) \
            if self.filter_cfg is not None else False

        valid_data_infos = []
        for i, data_info in enumerate(self.data_list):
            if filter_empty_gt and len(data_info['instances']) == 0:
                continue
            valid_data_infos.append(data_info)

        return valid_data_infos

    def get_cat_ids(self, idx: int) -> List[int]:
        """Get category ids by index."""
        instances = self.get_data_info(idx)['instances']
        return [instance['bbox_label'] for instance in instances]