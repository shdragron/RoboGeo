"""CAPE on RoboGeo (CARLA), SUV platform: the configuration of the released
checkpoint cape_suv.pth. Identical to cape_carla_sedan.py except for the
annotation files.
"""
_base_ = ['./cape_carla_sedan.py']

data_root = 'data/nuscenes/'
data = dict(
    train=dict(ann_file=data_root + 'suv_infos_train.pkl'),
    val=dict(ann_file=data_root + 'suv_infos_val.pkl'),
    test=dict(ann_file=data_root + 'suv_infos_val.pkl'))
