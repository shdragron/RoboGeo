# Modified by the RoboGeo authors, 2026: RoboGeo version of CAPE's
# projects/configs/CAPE/cape_r50_1408x512_24ep_wocbgs_imagenet_pretrain.py.
"""CAPE on RoboGeo (CARLA), Sedan platform: the configuration of the released
checkpoint cape_sedan.pth. cape_carla_suv.py and cape_carla_bus.py differ only
in the annotation files.

Derived from upstream cape_r50_1408x512_24ep_wocbgs_imagenet_pretrain.py
(R50-DCN caffe backbone from MSRA ImageNet weights, single frame, query
denoising, no CBGS). Changes:
  * data: CarlaNuScenesDataset, 6 classes, use_valid_flag=True (visibility >= 2)
  * no data augmentation: GridMask, random resize/flip and the BEV
    rotation/scale augmentation are removed. Every image is resized by 0.88
    (1600x900 -> 1408x792) and cropped to rows 280-792 (1408x512), identically
    in training and testing.
  * point_cloud_range [-51.2, -51.2, -5, 51.2, 51.2, 3], equal to the range
    hard-coded in cape_transformer.py
  * transformer num_classes=6 (also used for the denoising labels)
  * batch size 8 per GPU on 2 GPUs (total 16), AdamW lr 4e-4 (upstream: total
    batch size 8, lr 2e-4), fp32 with gradient clipping 35 (upstream: fp16)
"""
_base_ = [
    '../../../mmdetection3d/configs/_base_/datasets/nus-3d.py',
    '../../../mmdetection3d/configs/_base_/default_runtime.py'
]
backbone_norm_cfg = dict(type='LN', requires_grad=True)
plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]

image_height = 512
image_width = 1408

img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)

class_names = [
    'car', 'truck', 'bus', 'motorcycle', 'bicycle', 'pedestrian'
]
num_classes = len(class_names)

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)
model = dict(
    type='CAPE',
    use_grid_mask=False,
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(2, 3,),
        frozen_stages=-1,
        norm_cfg=dict(type='BN2d', requires_grad=False),
        norm_eval=True,
        style='caffe',
        with_cp=True,
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, False, True, True),
        pretrained='ckpts/resnet50_msra-5891d200.pth',
        ),
    img_neck=dict(
        type='CPFPN',
        in_channels=[1024, 2048],
        out_channels=256,
        num_outs=2),
    pts_bbox_head=dict(
        type='CAPETemporalDNHead',
        num_classes=num_classes,
        num_query=900,
        num_reg_fcs=2,
        normedlinear=False,
        dn_weight=1.0,
        split=0.75,
        transformer=dict(
            type='CAPETransformer',
            num_cameras=6,
            num_layers=6,
            feat_dim=256,
            feat_stride=32,
            image_height=image_height,
            image_width=image_width,
            bound=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            with_fpe=True,
            depth_start=1,
            depth_num=64,
            att_layer=dict(
                type='CrossViewAttention',
                hidden_dim=512,
                num_queries=900,
                qkv_bias=True,
                heads=8,
                dim_head=64,
                conditional=True),
            scalar=10,
            noise_scale=1.0,
            num_classes=num_classes,
            with_time=False),
        with_time=False,
        bbox_coder=dict(
            type='NMSFreeCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=point_cloud_range,
            max_num=300,
            voxel_size=voxel_size,
            num_classes=num_classes),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
        with_prev_aux_loss=False,
        prev_aux_loss_weight=0.0),
    train_cfg=dict(
        pts=dict(
            grid_size=[512, 512, 1],
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            out_size_factor=4,
            assigner=dict(
                type='HungarianAssigner3D',
                cls_cost=dict(type='FocalLossCost', weight=2.0),
                reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
                iou_cost=dict(type='IoUCost', weight=0.0),
                pc_range=point_cloud_range))))

dataset_type = 'CarlaNuScenesDataset'
# data/nuscenes is a symlink to $ROBOGEO_DATA/carla_geobev (see README.md).
data_root = 'data/nuscenes/'

file_client_args = dict(backend='disk')

# Deterministic resize 0.88 and crop to 1408x512; no flip, rotation or jitter.
ida_aug_conf = {
        "resize_lim": (0.88, 0.88),
        "final_dim": (image_height, image_width),
        "bot_pct_lim": (0.0, 0.0),
        "rot_lim": (0.0, 0.0),
        "H": 900,
        "W": 1600,
        "rand_flip": False,
    }
train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='ResizeCropFlipImage', data_aug_conf=ida_aug_conf, training=True),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['gt_bboxes_3d', 'gt_labels_3d', 'img'],
                    meta_keys=['lidar2img', 'intrinsics', 'extrinsics', 'img_shape', 'pad_shape',
                                'img_norm_cfg', 'sample_idx', 'gt_bboxes_3d', 'gt_labels_3d'])]
test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='ResizeCropFlipImage', data_aug_conf=ida_aug_conf, training=False),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['img'],
            meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img', 'intrinsics', 'extrinsics',
                'pad_shape', 'scale_factor', 'flip', 'box_mode_3d', 'box_type_3d',
                'img_norm_cfg', 'sample_idx', 'gt_bboxes_3d', 'gt_labels_3d'))
            ])
]

# Batch size 8 per GPU on 2 GPUs (total 16). The number of denoising query
# groups is 600 // (max GT boxes in the per-GPU batch), so another batch size
# per GPU also changes the denoising setup.
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'sedan_infos_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        box_type_3d='LiDAR'),
    val=dict(type=dataset_type,
             data_root=data_root,
             ann_file=data_root + 'sedan_infos_val.pkl',
             pipeline=test_pipeline, classes=class_names, modality=input_modality),
    test=dict(type=dataset_type,
              data_root=data_root,
              ann_file=data_root + 'sedan_infos_val.pkl',
              pipeline=test_pipeline, classes=class_names, modality=input_modality))

# lr 4e-4 for the total batch size 16 (upstream: 2e-4 for total batch size 8).
optimizer = dict(
    type='AdamW',
    lr=4e-4,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }),
    weight_decay=0.01)

# fp32 training with gradient clipping (no fp16 optimizer hook).
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
    )
total_epochs = 24
evaluation = dict(interval=2, pipeline=test_pipeline)
find_unused_parameters = False

custom_hooks = []

runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
# One checkpoint per epoch.
checkpoint_config = dict(interval=1)
resume_from = None
load_from = None
