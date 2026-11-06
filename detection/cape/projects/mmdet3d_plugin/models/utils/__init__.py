# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified by the RoboGeo authors, 2026: register only the modules used by single-frame CAPE.
# ------------------------------------------------------------------------
from .cape_transformer import CAPETransformer, CrossViewAttention, CrossAttention

__all__ = ['CAPETransformer', 'CrossViewAttention', 'CrossAttention']
