# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
# Modified by the RoboGeo authors, 2026: register the CARLA dataset; drop the multi-frame dataset.
# ------------------------------------------------------------------------
from .nuscenes_dataset import CustomNuScenesDataset
from .carla_nuscenes_dataset import CarlaNuScenesDataset
__all__ = [
    'CustomNuScenesDataset', 'CarlaNuScenesDataset'
]
