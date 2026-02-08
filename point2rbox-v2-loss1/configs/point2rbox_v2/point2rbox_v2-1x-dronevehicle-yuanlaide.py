_base_ = [
    '../_base_/datasets/dronevehicle-yuanlaide.py',  # 替换为dronevehicle数据集基础配置
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]
angle_version = 'le90'  # 保持与point2rbox_v2系列一致的角度模式

# model settings
model = dict(
    type='Point2RBoxV2',
    ss_prob=[0.68, 0.07, 0.25],  # 保持与同类模型一致的半监督概率配置
    copy_paste_start_epoch=6,
    data_preprocessor=dict(
        type='mmdet.DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],  # 采用与point2rbox_v2相同的图像预处理参数
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32,
        boxtype2tensor=False),
    backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),  # 与point2rbox_v2-dota配置一致的输出索引
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='mmdet.FPN',
        in_channels=[512, 1024, 2048],  # 匹配ResNet50的输出通道
        out_channels=128,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=3,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='Point2RBoxV2Head',
        num_classes=5,  # 关键修改：dronevehicle数据集为5类（car, bus, truck, van, freight_car）
        in_channels=128,
        feat_channels=128,
        strides=[8],
        edge_loss_start_epoch=6,
        joint_angle_start_epoch=1,
        voronoi_type='standard',
        # voronoi_type='gaussian-orientation',
        # voronoi_type='gaussian-full',
        # 调整voronoi阈值以适应无人机车辆数据集特性（参考同类配置格式）
        voronoi_thres=dict(
            default=[0.98, 0.01],
            override=(([0, 3], [0.99, 0.01]), 
                      ([1, 3, 4], [0.95, 0.005]))),  # 针对truck、van、freight_car
        square_cls=[],  # 假设car为近似正方形类别（根据实际数据调整）
        edge_loss_cls=[0, 1, 2, 3, 4],  # 对非正方形类别启用边缘损失
        post_process={},  # 无特殊后处理需求
        angle_coder=dict(
            type='PSCCoder',
            angle_version='le90',
            dual_freq=False,
            num_step=3,
            thr_mod=0),
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='GDLoss', loss_type='gwd', loss_weight=5.0),
        loss_overlap=dict(
            # type='GaussianOverlapLoss', loss_weight=4, lamb=0),
            type='GaussianOverlapLoss', loss_weight=10.0, lamb=0),
        loss_voronoi=dict(
            type='VoronoiWatershedLoss', loss_weight=5.0, debug=False),
        loss_bbox_edg=dict(
            type='EdgeLoss', loss_weight=0.3),
        loss_ss=dict(
            type='Point2RBoxV2ConsistencyLoss', loss_weight=1.0)),
    # training and testing settings
    train_cfg=None,
    test_cfg=dict(
        nms_pre=2000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms_rotated', iou_threshold=0.1),
        max_per_img=2000))

# 适配dronevehicle的训练管道
train_pipeline = [
    dict(type='mmdet.LoadImageFromFile'),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(type='RandomRotate', prob=0.5, angle_range=20),  
    dict(type='ConvertWeakSupervision', point_proportion=1., hbox_proportion=0),
    dict(type='mmdet.FixShapeResize', width=800, height=800, keep_ratio=True),
    #dict(type='mmdet.Resize', scale=(840, 712), keep_ratio=True),  
    dict(
        type='mmdet.RandomFlip',
        prob=0.75,
        direction=['horizontal', 'vertical', 'diagonal']),
    dict(type='mmdet.PackDetInputs')
]

# 数据加载器配置
train_dataloader = dict(
    batch_size=2,  # 保持与point2rbox_v2系列相同的批次大小
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=None,
    dataset=dict(
        type=_base_.dataset_type,  # 继承base中的DroneVehicleDataset
        pipeline=train_pipeline))  # 应用上述训练管道

# 验证和测试加载器保持与base配置一致，仅调整批次大小
val_dataloader = dict(
    batch_size=4,
    num_workers=2)

test_dataloader = val_dataloader

# 评估器配置（适配无人机车辆数据集的评估需求）
val_evaluator = dict(
    type='DOTAMetric',
    metric='mAP',
    iou_thrs=[0.5],  # 无人机目标常用0.5 IoU阈值
    )

test_evaluator = dict(
    type='DOTAMetric',
    format_only=False,
    merge_patches=False,
    outfile_prefix='/home/xiekaikai/point2R/point2rbox-v2/work_dirs/dvall/',  # 输出路径调整
    )

# 优化器与训练配置保持与point2rbox_v2系列一致
optim_wrapper = dict(
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=0.00005,
        betas=(0.9, 0.999),
        weight_decay=0.05))

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=1)
custom_hooks = [dict(type='mmdet.SetEpochInfoHook')]
