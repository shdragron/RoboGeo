"""Build the per-cell evaluation info files of the RoboGeo robustness protocols.

Every cell is an mmdet3d info pkl in the format of {platform}_infos_val.pkl.
Only camera fields are changed, so ground truth, sample tokens and the
dataset version stay those of the evaluated platform.

Viewpoint perturbation (protocol ``vp``), on the Sedan evaluation set:
  NORMAL  stock Sedan images and extrinsics
  EXT     base images + perturbed extrinsics
  IMG     perturbed images + base extrinsics
  CAL     perturbed images + the corresponding perturbed extrinsics
  modes   one of the 6 cameras, or ``all`` cameras at once
  grid    {roll, pitch, yaw} x {+-4, +-8, +-12, +-16, +-20} deg
Perturbed images come from carla_VR: carla_geobev frame N corresponds to
carla_VR frame 2N.

Cross-platform transfer (protocol ``cts``), on the SUV or Bus evaluation set:
  IMG     target-platform images + Sedan extrinsics (Sedan checkpoint)
  CAL     target-platform images + target extrinsics (Sedan checkpoint)
  ORACLE  as CAL, evaluated with the target-platform checkpoint

Usage:
  python build_conditions.py --data-root $ROBOGEO_DATA --protocol vp|cts \
      [--cells all|<comma list of cell ids>] [--subset full|fps16] --out DIR

Writes DIR/cells.json and one pkl per cell under DIR/pkls/.
"""
import argparse
import json
import os
import pickle
import re
import sys

import numpy as np
from pyquaternion import Quaternion

CAMS = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
        'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
VP_MODES = CAMS + ['all']
VP_CONDITIONS = {             # condition: (perturbed image, perturbed extrinsics)
    'EXT': (False, True),
    'IMG': (True, False),
    'CAL': (True, True),
}
VP_AXES = ['roll', 'pitch', 'yaw']
VP_ANGLES = [-20, -16, -12, -8, -4, 4, 8, 12, 16, 20]
CTS_TARGETS = ['suv', 'bus']
CTS_CONDITIONS = ['IMG', 'CAL', 'ORACLE']

SCENE_FRAME_RE = re.compile(r'scene-(\d+)-frame-(\d+)')
CELL_RE = re.compile(
    r'^vp/(EXT|IMG|CAL)/(' + '|'.join(VP_MODES) + r')/(roll|pitch|yaw)([+-])(\d+)$')
# Info pkls store image paths relative to the repository root of the
# detector (data/nuscenes -> carla_geobev).
REL_PREFIX = 'data/nuscenes/'


# --------------------------------------------------------------------------- #
# cell ids
# --------------------------------------------------------------------------- #
def vp_cell_id(condition, mode, axis, angle):
    return f'vp/{condition}/{mode}/{axis}{angle:+d}'


def all_cells(protocol):
    if protocol == 'vp':
        cells = ['vp/NORMAL']
        for cond in VP_CONDITIONS:
            for mode in VP_MODES:
                for axis in VP_AXES:
                    for angle in VP_ANGLES:
                        cells.append(vp_cell_id(cond, mode, axis, angle))
        return cells
    return [f'cts/{t}/{c}' for t in CTS_TARGETS for c in CTS_CONDITIONS]


def parse_cell(cell):
    """Cell id -> the cells.json record without the pkl path."""
    rec = dict(cell=cell, protocol=None, condition=None, mode=None, axis=None,
               angle=None, target=None, model_platform='sedan')
    if cell == 'vp/NORMAL':
        rec.update(protocol='vp', condition='NORMAL', target='sedan')
        return rec
    m = CELL_RE.match(cell)
    if m:
        angle = int(m.group(5)) * (1 if m.group(4) == '+' else -1)
        if angle not in VP_ANGLES:
            raise ValueError(f'angle not in the grid: {cell}')
        rec.update(protocol='vp', condition=m.group(1), mode=m.group(2),
                   axis=m.group(3), angle=angle, target='sedan')
        return rec
    parts = cell.split('/')
    if (len(parts) == 3 and parts[0] == 'cts' and parts[1] in CTS_TARGETS
            and parts[2] in CTS_CONDITIONS):
        rec.update(protocol='cts', condition=parts[2], target=parts[1])
        if parts[2] == 'ORACLE':
            rec['model_platform'] = parts[1]
        return rec
    raise ValueError(f'unknown cell id: {cell}')


def cell_file_name(cell):
    return cell.replace('/', '__')


# --------------------------------------------------------------------------- #
# info helpers
# --------------------------------------------------------------------------- #
def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def dump_pkl(obj, path):
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def scene_frame(info):
    """(scene, frame) strings parsed from the CAM_FRONT image path.

    Sample tokens differ between platforms; scene and frame do not.
    """
    m = SCENE_FRAME_RE.search(info['cams']['CAM_FRONT']['data_path'])
    if m is None:
        raise ValueError('no scene-/frame- token in '
                         + info['cams']['CAM_FRONT']['data_path'])
    return m.group(1), m.group(2)


def absolute_paths(infos, geobev_root):
    """Copy of infos with every camera data_path made absolute."""
    out = []
    for info in infos:
        info = dict(info)
        cams = {}
        for name, cam in info['cams'].items():
            cam = dict(cam)
            p = cam['data_path']
            if not os.path.isabs(p):
                if p.startswith(REL_PREFIX):
                    p = p[len(REL_PREFIX):]
                p = os.path.join(geobev_root, p)
            cam['data_path'] = p
            cams[name] = cam
        info['cams'] = cams
        out.append(info)
    return out


def sensor2lidar_from_sensor2ego(info, s2e_rotation, s2e_translation):
    """sensor->lidar (3x3, 3) implied by sensor->ego and the sample's lidar->ego."""
    l2e_r = Quaternion(info['lidar2ego_rotation']).rotation_matrix
    l2e_t = np.asarray(info['lidar2ego_translation'], dtype=np.float64)
    s2e_r = Quaternion(s2e_rotation).rotation_matrix
    s2e_t = np.asarray(s2e_translation, dtype=np.float64)
    return l2e_r.T @ s2e_r, l2e_r.T @ (s2e_t - l2e_t)


def check_extrinsics(info, cam, where, atol=1e-6):
    rot, trans = sensor2lidar_from_sensor2ego(
        info, cam['sensor2ego_rotation'], cam['sensor2ego_translation'])
    if not (np.allclose(rot, cam['sensor2lidar_rotation'], atol=atol)
            and np.allclose(trans, cam['sensor2lidar_translation'], atol=atol)):
        raise ValueError(f'sensor2lidar and sensor2ego disagree ({where})')


# --------------------------------------------------------------------------- #
# viewpoint perturbation
# --------------------------------------------------------------------------- #
class ViewpointSource:
    """Perturbed images and extrinsics of carla_VR."""

    def __init__(self, vr_root):
        self.vr_root = vr_root
        with open(os.path.join(vr_root, 'viewpoint_metadata.json')) as f:
            meta = json.load(f)
        # meta['vr_root'] is a path on the machine that rendered the data and
        # is ignored; images are resolved under vr_root.
        self.scenes = meta['scenes']
        self._cache = {}

    @staticmethod
    def variant(axis, angle):
        vals = {'yaw': 0, 'pitch': 0, 'roll': 0}
        vals[axis] = angle
        return f"yaw{vals['yaw']}pitch{vals['pitch']}roll{vals['roll']}"

    def image_path(self, scene, frame, cam, variant):
        # carla_VR renders at twice the frame rate of carla_geobev.
        return os.path.join(
            self.vr_root, 'sweeps', f'RGB-{cam}_{variant}',
            f'SimBEV-scene-{scene}-frame-{2 * int(frame):04d}-RGB-{cam}_{variant}.jpg')

    def extrinsics(self, info, scene, cam, variant):
        """Perturbed sensor2lidar_* and sensor2ego_* of one camera."""
        key = (scene, cam, variant)
        if key not in self._cache:
            rec = self.scenes[f'scene_{scene}'][cam][variant]
            ext = dict(
                sensor2lidar_rotation=Quaternion(
                    rec['sensor2lidar_rotation']).rotation_matrix,
                sensor2lidar_translation=np.array(
                    rec['sensor2lidar_translation'], dtype=np.float64),
                sensor2ego_rotation=[float(v) for v in rec['sensor2ego_rotation']],
                sensor2ego_translation=[float(v) for v in rec['sensor2ego_translation']])
            check_extrinsics(info, ext, f'carla_VR scene_{scene} {cam} {variant}')
            self._cache[key] = ext
        return self._cache[key]


def make_vp_infos(base_infos, rec, vr):
    """Infos of one viewpoint-perturbation cell (base_infos is not modified)."""
    if rec['condition'] == 'NORMAL':
        return base_infos
    swap_img, swap_ext = VP_CONDITIONS[rec['condition']]
    variant = vr.variant(rec['axis'], rec['angle'])
    cams = CAMS if rec['mode'] == 'all' else [rec['mode']]
    out = []
    for info in base_infos:
        scene, frame = scene_frame(info)
        info = dict(info)
        info['cams'] = dict(info['cams'])
        for name in cams:
            cam = dict(info['cams'][name])
            if swap_img:
                cam['data_path'] = vr.image_path(scene, frame, name, variant)
            if swap_ext:
                for k, v in vr.extrinsics(info, scene, name, variant).items():
                    cam[k] = np.array(v) if isinstance(v, np.ndarray) else list(v)
            info['cams'][name] = cam
        out.append(info)
    return out


# --------------------------------------------------------------------------- #
# cross-platform transfer
# --------------------------------------------------------------------------- #
EXTRINSIC_KEYS = ['sensor2lidar_rotation', 'sensor2lidar_translation',
                  'sensor2ego_rotation', 'sensor2ego_translation']


def make_cts_infos(target_infos, sedan_index, condition):
    """Infos of one cross-platform cell on the target evaluation set."""
    if condition in ('CAL', 'ORACLE'):
        return target_infos
    out = []
    for info in target_infos:
        key = scene_frame(info)
        if key not in sedan_index:
            raise KeyError(f'no Sedan sample for scene {key[0]} frame {key[1]}')
        sedan_cams = sedan_index[key]['cams']
        info = dict(info)
        info['cams'] = dict(info['cams'])
        for name, cam in info['cams'].items():
            cam = dict(cam)
            for k in EXTRINSIC_KEYS:
                v = sedan_cams[name][k]
                cam[k] = np.array(v) if isinstance(v, np.ndarray) else list(v)
            check_extrinsics(info, cam, f'cts scene {key[0]} {name}')
            info['cams'][name] = cam
        out.append(info)
    return out


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--data-root', required=True,
                    help='RoboGeo root holding carla_geobev/ and carla_VR/')
    ap.add_argument('--protocol', required=True, choices=['vp', 'cts'])
    ap.add_argument('--cells', default='all',
                    help='"all" or a comma-separated list of cell ids')
    ap.add_argument('--subset', default='full', choices=['full', 'fps16'],
                    help='full = all 3,792 evaluation frames (the benchmark); '
                         'fps16 = 768-frame subset for quick checks only')
    ap.add_argument('--out', required=True)
    return ap.parse_args()


def main():
    args = parse_args()
    data_root = os.path.abspath(args.data_root)
    geobev = os.path.join(data_root, 'carla_geobev')
    suffix = '_fps16' if args.subset == 'fps16' else ''

    if args.cells == 'all':
        cells = all_cells(args.protocol)
    else:
        cells = [c.strip() for c in args.cells.split(',') if c.strip()]
    recs = [parse_cell(c) for c in cells]
    wrong = [r['cell'] for r in recs if r['protocol'] != args.protocol]
    if wrong:
        sys.exit(f'cells of another protocol: {wrong}')

    out_dir = os.path.abspath(args.out)
    pkl_dir = os.path.join(out_dir, 'pkls')
    os.makedirs(pkl_dir, exist_ok=True)

    if args.protocol == 'vp':
        base = load_pkl(os.path.join(geobev, f'sedan_infos_val{suffix}.pkl'))
        base_infos = absolute_paths(base['infos'], geobev)
        vr = ViewpointSource(os.path.join(data_root, 'carla_VR'))
    else:
        sedan = load_pkl(os.path.join(geobev, 'sedan_infos_val.pkl'))
        sedan_index = {scene_frame(i): i for i in sedan['infos']}
        targets = {}

    for n, rec in enumerate(recs, 1):
        if args.protocol == 'vp':
            metadata = base['metadata']
            infos = make_vp_infos(base_infos, rec, vr)
        else:
            t = rec['target']
            if t not in targets:
                d = load_pkl(os.path.join(geobev, f'{t}_infos_val{suffix}.pkl'))
                targets[t] = (d['metadata'], absolute_paths(d['infos'], geobev))
            metadata, target_infos = targets[t]
            infos = make_cts_infos(target_infos, sedan_index, rec['condition'])
        rec['pkl'] = os.path.join(pkl_dir, cell_file_name(rec['cell']) + '.pkl')
        dump_pkl(dict(metadata=metadata, infos=infos), rec['pkl'])
        print(f'[{n}/{len(recs)}] {rec["cell"]}: {len(infos)} frames', flush=True)

    with open(os.path.join(out_dir, 'cells.json'), 'w') as f:
        json.dump(recs, f, indent=1)
    print(f'wrote {len(recs)} cells ({args.subset}) -> {out_dir}/cells.json')


if __name__ == '__main__':
    main()
