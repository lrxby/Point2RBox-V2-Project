# import glob
# import os.path as osp
# from typing import List

# from mmengine.dataset import BaseDataset

# from mmrotate.registry import DATASETS


# @DATASETS.register_module()
# class DroneVehicleDataset(BaseDataset):
#     """DroneVehicle dataset for rotation detection.

#     Note: ``ann_file`` in DroneVehicleDataset refers to the path of the folder
#     containing TXT annotation files (e.g., 'data/dronevehicle/train/annfiles/').
#     """

#     # 固定结构：类别与可视化颜色（根据数据集特性定制内容）
#     METAINFO = {
#         'classes': ('car', 'bus', 'truck', 'van', 'feright_car'),
#         'palette': [
#             (255, 0, 0),    # 红色（car）
#             (0, 255, 0),    # 绿色（bus）
#             (0, 0, 255),    # 蓝色（truck）
#             (255, 255, 0),  # 黄色（van）
#             (128, 0, 128)   # 紫色（feright_car）
#         ]
#     }

#     def __init__(self,
#                  img_suffix: str = 'jpg',  # 图片后缀（根据特性定制）
#                  diff_thr: int = 100,      # 难度过滤阈值（复用DOTA逻辑）
#                  **kwargs) -> None:
#         self.img_suffix = img_suffix
#         self.diff_thr = diff_thr
#         super().__init__(** kwargs)

#     def load_data_list(self) -> List[dict]:
#         """加载数据集信息（核心定制部分，适配TXT标注格式）"""
#         # 建立类别ID到索引的映射（标注中的类别ID对应METAINFO的classes顺序）
#         cls_map = {i: idx for idx, i in enumerate(range(len(self.METAINFO['classes'])))}
#         data_list = []

#         # 处理无标注的测试集场景（复用DOTA的空标注逻辑）
#         if self.ann_file == '':
#             img_files = glob.glob(osp.join(self.data_prefix['img_path'], f'*.{self.img_suffix}'))
#             for img_path in img_files:
#                 img_name = osp.split(img_path)[1]
#                 data_info = {
#                     'img_id': img_name[:-len(self.img_suffix)-1],
#                     'file_name': img_name,
#                     'img_path': img_path,
#                     'instances': [dict(bbox=[], bbox_label=[], ignore_flag=0)]
#                 }
#                 data_list.append(data_info)
#             return data_list

#         # 加载标注文件（TXT格式）
#         txt_files = glob.glob(osp.join(self.ann_file, '*.txt'))
#         if not txt_files:
#             raise ValueError(f'No TXT annotation files found in {self.ann_file}')

#         for txt_file in txt_files:
#             img_id = osp.split(txt_file)[1][:-4]  # 文件名作为img_id
#             img_name = f'{img_id}.{self.img_suffix}'
#             data_info = {
#                 'img_id': img_id,
#                 'file_name': img_name,
#                 'img_path': osp.join(self.data_prefix['img_path'], img_name)
#             }

#             # 解析TXT标注（8点坐标 + 类别ID）
#             instances = []
#             with open(txt_file, 'r') as f:
#                 for line in f.readlines():
#                     line = line.strip().split()
#                     if len(line) != 9:  # 校验标注格式（8坐标 + 1类别ID）
#                         continue

#                     # 提取四点旋转框（直接使用8点坐标，无需转换）
#                     bbox = [float(coord) for coord in line[:8]]
#                     # 映射类别ID到索引（标注中的ID需与classes顺序一致）
#                     cls_id = int(line[8])
#                     if cls_id not in cls_map:
#                         continue  # 跳过未定义类别
#                     bbox_label = cls_map[cls_id]

#                     # 难度过滤（此处简化处理，如需可扩展）
#                     ignore_flag = 1 if 0 > self.diff_thr else 0  # 示例逻辑

#                     instances.append({
#                         'bbox': bbox,
#                         'bbox_label': bbox_label,
#                         'ignore_flag': ignore_flag
#                     })

#             data_info['instances'] = instances
#             data_list.append(data_info)

#         return data_list

#     def filter_data(self) -> List[dict]:
#         """过滤无效样本（复用DOTA过滤逻辑）"""
#         if self.test_mode:
#             return self.data_list

#         filter_empty_gt = self.filter_cfg.get('filter_empty_gt', False) \
#             if self.filter_cfg is not None else False

#         valid_data = [info for info in self.data_list 
#                       if not (filter_empty_gt and len(info['instances']) == 0)]
#         return valid_data

#     def get_cat_ids(self, idx: int) -> List[int]:
#         """获取图片中的类别ID（固定逻辑）"""
#         instances = self.get_data_info(idx)['instances']
#         return [instance['bbox_label'] for instance in instances]


# Copyright (c) OpenMMLab. All rights reserved.
import glob
import os.path as osp
from typing import List

from mmengine.dataset import BaseDataset
from mmrotate.registry import DATASETS


@DATASETS.register_module()
class DroneVehicleDataset(BaseDataset):
    """DroneVehicle Dataset (Raw Format: 9 columns).

    Format: x1 y1 x2 y2 x3 y3 x4 y4 class_id
    Example: 133 532 160 532 165 461 134 459 4
    """

    METAINFO = {
        # 对应 class_id: 0, 1, 2, 3, 4
        'classes': ('car', 'bus', 'truck', 'van', 'freight_car'),
        'palette': [
            (220, 20, 60),  # car (Red)
            (119, 11, 32),  # bus (Dark Red)
            (0, 0, 142),    # truck (Dark Blue)
            (0, 0, 230),    # van (Blue)
            (106, 0, 228)   # freight_car (Purple)
        ]
    }

    def __init__(self,
                 img_suffix: str = 'jpg',
                 diff_thr: int = 100,  # 保留参数以兼容标准接口，虽暂未使用
                 **kwargs) -> None:
        self.img_suffix = img_suffix
        self.diff_thr = diff_thr
        super().__init__(**kwargs)

    def load_data_list(self) -> List[dict]:
        """Load annotations from the raw text files."""
        data_list = []
        
        # 确保 ann_file 指向的是包含 .txt 文件的文件夹路径
        txt_files = glob.glob(osp.join(self.ann_file, '*.txt'))
        
        if len(txt_files) == 0:
            raise ValueError(f'No .txt files found in {self.ann_file}')

        # 预先计算合法的类别ID集合 (0-4)
        valid_cat_ids = set(range(len(self.METAINFO['classes'])))

        for txt_file in txt_files:
            data_info = {}
            # 获取文件名（不含扩展名），作为 img_id
            img_id = osp.split(txt_file)[1][:-4]
            data_info['img_id'] = img_id
            
            # 拼接图片文件名
            img_name = img_id + f'.{self.img_suffix}'
            data_info['file_name'] = img_name
            data_info['img_path'] = osp.join(self.data_prefix['img_path'], img_name)

            instances = []
            with open(txt_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    
                    # 1. 基础格式检查：必须是 9 列 (8坐标 + 1类别)
                    if len(parts) != 9:
                        continue

                    try:
                        # 解析坐标 (前8个)
                        bbox = [float(x) for x in parts[:8]]
                        # 解析类别ID (第9个)
                        label_id = int(parts[8])
                    except ValueError:
                        continue

                    # 2. 类别ID检查
                    if label_id not in valid_cat_ids:
                        continue

                    # 3. 【新增】数据有效性检查：剔除宽高极小的噪点框
                    # 防止 Point2RBox 计算 Voronoi/Gaussian 时报错
                    xs = bbox[0::2]
                    ys = bbox[1::2]
                    w = max(xs) - min(xs)
                    h = max(ys) - min(ys)
                    if w < 1 or h < 1: 
                        continue

                    instance = {
                        'bbox': bbox,
                        'bbox_label': label_id, # 直接使用原始ID
                        'ignore_flag': 0        # 原始数据无难度分级，全部视为有效样本
                    }
                    instances.append(instance)

            data_info['instances'] = instances
            data_list.append(data_info)

        return data_list

    def filter_data(self) -> List[dict]:
        """Filter images with no ground truths."""
        if self.test_mode:
            return self.data_list

        filter_empty_gt = self.filter_cfg.get('filter_empty_gt', False) \
            if self.filter_cfg is not None else False

        valid_data_infos = []
        for data_info in self.data_list:
            if filter_empty_gt and len(data_info['instances']) == 0:
                continue
            valid_data_infos.append(data_info)

        return valid_data_infos

    def get_cat_ids(self, idx: int) -> List[int]:
        """Get category ids by index."""
        instances = self.get_data_info(idx)['instances']
        return [instance['bbox_label'] for instance in instances]