# Added by the RoboGeo authors, 2026.
"""BEVFormer-tiny on RoboGeo, SUV platform (the config of the released checkpoint).

Inherits everything from bevformer_tiny_carla.py (Sedan) and only points
ann_file to the suv info files. All platforms share the scene split
(split/{train,val}.txt); the databases differ (v1.0-carla_suv and
v1.0-carla_suv_eval) and are recorded in suv_infos_{train,val}.pkl by
tools/create_carla_data.py.
"""
_base_ = ['./bevformer_tiny_carla.py']

vehicle = 'suv'
data_root = 'data/nuscenes/'

data = dict(
    train=dict(ann_file=data_root + f'{vehicle}_infos_train.pkl'),
    val=dict(ann_file=data_root + f'{vehicle}_infos_val.pkl'),
    test=dict(ann_file=data_root + f'{vehicle}_infos_val.pkl'),
)

