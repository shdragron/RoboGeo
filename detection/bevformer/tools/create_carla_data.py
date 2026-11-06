# Added by the RoboGeo authors, 2026. The per-sample conversion is adapted from
# tools/data_converter/nuscenes_converter.py of mmdetection3d as shipped with
# BEVFormer (Copyright (c) OpenMMLab, Apache-2.0).
"""Generate BEVFormer info pkls from the RoboGeo nuScenes-format databases.

Multi-process variant. Each worker processes a slice of samples.

Per-vehicle (viewpoint) layout under carla_geobev/:
  v1.0-carla_<veh>       -- train DB  (veh in {sedan, suv, bus})
  v1.0-carla_<veh>_eval  -- eval  DB
  split/train.txt, split/val.txt  -- scene-name split shared by ALL vehicles

The SAME split.txt applies to every vehicle (scene names are identical across
sedan/suv/bus). train.txt matches each v1.0-carla_<veh> exactly; val.txt is a
subset of each v1.0-carla_<veh>_eval (scenes not listed are dropped). Data is
read in place -- nothing is copied or moved; we only filter samples by scene.

Differences from the mmdetection3d nuScenes converter:
  1. version  = 'v1.0-carla_<veh>' / 'v1.0-carla_<veh>_eval' (per-vehicle CARLA db)
  2. can_bus  = zeros(18)            -- CARLA has no can-bus stream; use_can_bus=False in cfg
  3. valid_flag = visibility_token >= '2'  -- camera-visible objects (>= 40% visible)
  4. sweeps left empty -- BEVFormer is camera-only; temporal forced off via scene_token
  5. train/val membership comes from split/{train,val}.txt, not the simbev json

Output (in --out-dir): <veh>_infos_train.pkl, <veh>_infos_val.pkl per vehicle.

Run:
  python tools/create_carla_data.py \
      --root-path data/nuscenes \
      --out-dir   data/nuscenes \
      --vehicles  sedan suv bus \
      --workers 16
"""
import argparse
import logging
import os
import sys
import time
from os import path as osp
from multiprocessing import Pool

import mmcv
import numpy as np
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion

from mmdet3d.datasets import NuScenesDataset

# ---- Visibility filter ----
VISIBILITY_MIN_TOKEN = '2'        # >= '2' means >= 40% visible
VISIBLE_TOKENS = {'2', '3', '4'}

CAMERA_TYPES = [
    'CAM_FRONT',
    'CAM_FRONT_RIGHT',
    'CAM_FRONT_LEFT',
    'CAM_BACK',
    'CAM_BACK_LEFT',
    'CAM_BACK_RIGHT',
]

# Globals set per-worker (NuScenes object is huge, build once per process)
_NUSC = None
_ROOT = None
_VERSION = None


def _init_worker(root_path, version):
    """Initialize a per-process NuScenes instance."""
    global _NUSC, _ROOT, _VERSION
    _ROOT = root_path
    _VERSION = version
    _NUSC = NuScenes(version=version, dataroot=root_path, verbose=False)
    sys.stdout.write(f'[pid {os.getpid()}] NuScenes ready\n')
    sys.stdout.flush()


def _obtain_sensor2lidar(nusc, sensor_token, l2e_t, l2e_r_mat, e2g_t, e2g_r_mat,
                         sensor_type):
    sd_rec = nusc.get('sample_data', sensor_token)
    cs_rec = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
    pose_rec = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    data_path = osp.join(nusc.dataroot, sd_rec['filename'])

    sweep = {
        'data_path': data_path,
        'type': sensor_type,
        'sample_data_token': sd_rec['token'],
        'sensor2ego_translation': cs_rec['translation'],
        'sensor2ego_rotation': cs_rec['rotation'],
        'ego2global_translation': pose_rec['translation'],
        'ego2global_rotation': pose_rec['rotation'],
        'timestamp': sd_rec['timestamp'],
    }
    l2e_r_s = sweep['sensor2ego_rotation']
    l2e_t_s = sweep['sensor2ego_translation']
    e2g_r_s = sweep['ego2global_rotation']
    e2g_t_s = sweep['ego2global_translation']

    l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
    e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix
    R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T -= (e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T) +
          l2e_t @ np.linalg.inv(l2e_r_mat).T)
    sweep['sensor2lidar_rotation'] = R.T
    sweep['sensor2lidar_translation'] = T
    return sweep


def _process_sample_token(sample_token):
    """Convert a single sample to BEVFormer info dict. Returns (scene_token, info, n_valid, n_total)."""
    nusc = _NUSC
    sample = nusc.get('sample', sample_token)
    lidar_token = sample['data']['LIDAR_TOP']
    sd_rec = nusc.get('sample_data', lidar_token)
    cs_rec = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
    pose_rec = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    lidar_path, boxes, _ = nusc.get_sample_data(lidar_token)

    info = {
        'lidar_path': lidar_path,
        'token': sample['token'],
        'prev': sample['prev'],
        'next': sample['next'],
        'can_bus': np.zeros(18, dtype=np.float64),
        'frame_idx': 0,  # set later if temporal needed (not used here)
        'sweeps': [],
        'cams': {},
        # NOTE: We assign a UNIQUE scene_token per sample so that BEVFormer's
        # union2one() always sees a "scene boundary" and sets prev_bev_exists=False.
        # This effectively disables temporal aggregation while keeping queue_length=2
        # (the minimum that forward_train can handle without code modification).
        'scene_token': sample['token'],
        'lidar2ego_translation': cs_rec['translation'],
        'lidar2ego_rotation': cs_rec['rotation'],
        'ego2global_translation': pose_rec['translation'],
        'ego2global_rotation': pose_rec['rotation'],
        'timestamp': sample['timestamp'],
    }

    l2e_t = info['lidar2ego_translation']
    l2e_r_mat = Quaternion(info['lidar2ego_rotation']).rotation_matrix
    e2g_t = info['ego2global_translation']
    e2g_r_mat = Quaternion(info['ego2global_rotation']).rotation_matrix

    for cam in CAMERA_TYPES:
        cam_token = sample['data'][cam]
        cam_path, _, cam_intrinsic = nusc.get_sample_data(cam_token)
        cam_info = _obtain_sensor2lidar(nusc, cam_token, l2e_t, l2e_r_mat,
                                        e2g_t, e2g_r_mat, cam)
        cam_info['cam_intrinsic'] = cam_intrinsic
        info['cams'][cam] = cam_info

    annotations = [nusc.get('sample_annotation', t) for t in sample['anns']]
    if len(boxes) == 0:
        info['gt_boxes'] = np.zeros((0, 7), dtype=np.float32)
        info['gt_names'] = np.array([], dtype=object)
        info['gt_velocity'] = np.zeros((0, 2), dtype=np.float32)
        info['num_lidar_pts'] = np.zeros((0,), dtype=np.int64)
        info['num_radar_pts'] = np.zeros((0,), dtype=np.int64)
        info['valid_flag'] = np.zeros((0,), dtype=bool)
        n_valid, n_total = 0, 0
    else:
        locs = np.array([b.center for b in boxes]).reshape(-1, 3)
        dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
        rots = np.array(
            [b.orientation.yaw_pitch_roll[0] for b in boxes]).reshape(-1, 1)

        velocity = np.array(
            [nusc.box_velocity(t)[:2] for t in sample['anns']])
        for i in range(len(boxes)):
            velo = np.array([*velocity[i], 0.0])
            velo = velo @ np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(
                l2e_r_mat).T
            velocity[i] = velo[:2]
        velocity = np.nan_to_num(velocity, nan=0.0)

        names = [b.name for b in boxes]
        for i, n in enumerate(names):
            if n in NuScenesDataset.NameMapping:
                names[i] = NuScenesDataset.NameMapping[n]
        names = np.array(names)

        gt_boxes = np.concatenate([locs, dims, -rots - np.pi / 2], axis=1)

        vis_tokens = [a.get('visibility_token', '1') for a in annotations]
        valid_flag = np.array([t in VISIBLE_TOKENS for t in vis_tokens],
                              dtype=bool)
        n_valid = int(valid_flag.sum())
        n_total = len(valid_flag)

        info['gt_boxes'] = gt_boxes.astype(np.float32)
        info['gt_names'] = names
        info['gt_velocity'] = velocity.reshape(-1, 2).astype(np.float32)
        info['num_lidar_pts'] = np.array(
            [a.get('num_lidar_pts', 0) for a in annotations], dtype=np.int64)
        info['num_radar_pts'] = np.array(
            [a.get('num_radar_pts', 0) for a in annotations], dtype=np.int64)
        info['valid_flag'] = valid_flag

    return (sample['scene_token'], info, n_valid, n_total)


def _load_split(root_path):
    """Read scene-name splits from carla_geobev/split/{train,val}.txt.

    One scene name per line (e.g. 'scene_0000'). The same split is shared by
    all vehicles. Returns (train_scene_names, val_scene_names) as sets.
    """
    split_dir = osp.join(root_path, 'split')
    with open(osp.join(split_dir, 'train.txt')) as f:
        train = {ln.strip() for ln in f if ln.strip()}
    with open(osp.join(split_dir, 'val.txt')) as f:
        val = {ln.strip() for ln in f if ln.strip()}
    return train, val


def setup_logger(log_path):
    fmt = '%(asctime)s [%(levelname)s] %(message)s'
    logging.basicConfig(level=logging.INFO, format=fmt, datefmt='%H:%M:%S',
                        handlers=[
                            logging.StreamHandler(sys.stdout),
                            logging.FileHandler(log_path, mode='w'),
                        ])


def _build_split(root_path, out_path, version, keep_scene_names, workers, log):
    """Build one info pkl from a single CARLA DB, keeping only samples whose
    scene name is in keep_scene_names. Reads in place; copies/moves nothing."""
    t0 = time.time()
    log.info(f'[{version}] loading NuScenes (main proc, verbose) ...')
    nusc_main = NuScenes(version=version, dataroot=root_path, verbose=True)
    db_scene_names = {s['name'] for s in nusc_main.scene}
    keep_scene_tokens = {s['token'] for s in nusc_main.scene
                         if s['name'] in keep_scene_names}
    dropped = sorted(db_scene_names - keep_scene_names)
    sample_tokens = [s['token'] for s in nusc_main.sample
                     if s['scene_token'] in keep_scene_tokens]
    n_samples = len(sample_tokens)
    log.info(f'[{version}] scenes db={len(db_scene_names)} '
             f'kept={len(keep_scene_tokens)} dropped={len(dropped)}'
             f"{'  ' + ','.join(dropped) if dropped else ''}")
    log.info(f'[{version}] samples to convert={n_samples} '
             f'(loaded in {time.time()-t0:.1f}s)')
    del nusc_main  # free memory before forking workers

    t0 = time.time()
    infos_by_scene = {}
    n_valid_total = 0
    n_total_total = 0
    processed = 0
    last_log = time.time()
    with Pool(processes=workers, initializer=_init_worker,
              initargs=(root_path, version)) as pool:
        for result in pool.imap_unordered(_process_sample_token, sample_tokens,
                                          chunksize=256):
            scene_token, info, n_valid, n_total = result
            infos_by_scene.setdefault(scene_token, []).append(info)
            n_valid_total += n_valid
            n_total_total += n_total
            processed += 1
            if time.time() - last_log > 5.0:
                el = time.time() - t0
                rate = processed / max(el, 1e-3)
                eta = (n_samples - processed) / max(rate, 1e-3)
                log.info(f'  [{version}] {processed}/{n_samples} '
                         f'({100*processed/max(n_samples,1):.1f}%)  '
                         f'rate={rate:.1f}/s  ETA={eta:.0f}s')
                last_log = time.time()

    infos = []
    for token in sorted(infos_by_scene.keys()):
        scene_infos = infos_by_scene[token]
        scene_infos.sort(key=lambda e: e['timestamp'])
        infos.extend(scene_infos)

    log.info(f'[{version}] converted {len(infos)} samples in '
             f'{time.time()-t0:.1f}s  valid_flag kept '
             f'{n_valid_total}/{n_total_total} '
             f'({100*n_valid_total/max(n_total_total,1):.2f}%)')
    metadata = dict(version=version)
    mmcv.dump(dict(infos=infos, metadata=metadata), out_path)
    log.info(f'[{version}] dumped -> {out_path}')
    return len(infos)


def create_carla_infos(root_path, out_dir, vehicles, workers=16):
    log_path = osp.join(out_dir, 'carla_converter.log')
    mmcv.mkdir_or_exist(out_dir)
    setup_logger(log_path)
    log = logging.getLogger()
    log.info(f'root_path={root_path}  out_dir={out_dir}')
    log.info(f'vehicles={list(vehicles)}  workers={workers}')
    log.info(f'visibility filter: token in {sorted(VISIBLE_TOKENS)} '
             f'(>= {VISIBILITY_MIN_TOKEN})')

    train_scenes, val_scenes = _load_split(root_path)
    log.info(f'split: train={len(train_scenes)} val={len(val_scenes)} '
             f'overlap={len(train_scenes & val_scenes)}')

    summary = {}
    for veh in vehicles:
        train_out = osp.join(out_dir, f'{veh}_infos_train.pkl')
        val_out = osp.join(out_dir, f'{veh}_infos_val.pkl')
        n_tr = _build_split(root_path, train_out, f'v1.0-carla_{veh}',
                            train_scenes, workers, log)
        n_va = _build_split(root_path, val_out, f'v1.0-carla_{veh}_eval',
                            val_scenes, workers, log)
        summary[veh] = (n_tr, n_va)

    log.info('==== SUMMARY ====')
    for veh, (n_tr, n_va) in summary.items():
        log.info(f'  {veh}: train={n_tr}  val={n_va}')
    log.info('DONE.')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root-path', default='data/nuscenes')
    ap.add_argument('--out-dir', default='data/nuscenes')
    ap.add_argument('--vehicles', nargs='+', default=['sedan', 'suv', 'bus'],
                    help='ego-vehicle viewpoints to convert')
    ap.add_argument('--workers', type=int, default=16)
    args = ap.parse_args()
    create_carla_infos(args.root_path, args.out_dir, args.vehicles,
                       args.workers)


if __name__ == '__main__':
    main()
