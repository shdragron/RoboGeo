# Added by the RoboGeo authors, 2026.
"""Sanity-check the generated CARLA pkl by projecting 3D GT boxes onto camera images.

What we verify:
  1. lidar2img matrix (built exactly like CustomNuScenesDataset.get_data_info does)
     puts box centers within image bounds for cameras that should see them.
  2. Visible vs hidden distribution matches our valid_flag (visibility>='2').
  3. Cross-camera consistency: a box in front of ego shows up in CAM_FRONT only,
     boxes to the side show up in CAM_FRONT_LEFT/RIGHT, etc.
  4. Renders 8 sample images with projected boxes to disk for human eyeball check.

Output:
  work_dirs/proj_check/<split>/<sample_token>_<cam>.jpg
  Per-camera projection counts printed to stdout.
"""
import argparse
import os
import pickle
from os import path as osp

import cv2
import mmcv
import numpy as np


CAM_NAMES = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
             'CAM_BACK',  'CAM_BACK_LEFT',   'CAM_BACK_RIGHT']

# CARLA class -> color (BGR)
COLORS = {
    'car':         (0, 255, 0),
    'truck':       (0, 200, 200),
    'bus':         (255, 100, 0),
    'motorcycle':  (255, 0, 255),
    'bicycle':     (0, 0, 255),
    'pedestrian':  (255, 255, 0),
    'construction_vehicle': (128, 128, 128),
    'trailer': (128, 0, 128),
    'traffic_cone': (200, 200, 0),
    'barrier': (50, 50, 50),
}

# visibility_token -> color (BGR). '1' (0-40%) is what valid_flag DROPS.
VIS_COLORS = {
    '1': (130, 130, 130),  # gray   0-40%   (dropped by valid_flag)
    '2': (0,   0, 255),    # red    40-60%
    '3': (0, 165, 255),    # orange 60-80%
    '4': (0, 255,   0),    # green  80-100%
}
VIS_LEVEL = {'1': 'v0-40', '2': 'v40-60', '3': 'v60-80', '4': 'v80-100'}


def _draw_3d_box(img, pts, color, thick=2):
    """pts: 8x2 int corners (bottom 0..3, top 4..7)."""
    for j in range(4):
        cv2.line(img, tuple(pts[j]), tuple(pts[(j + 1) % 4]), color, thick)
    for j in range(4, 8):
        cv2.line(img, tuple(pts[j]), tuple(pts[4 + (j - 4 + 1) % 4]), color,
                 max(1, thick - 1))
    for j in range(4):
        cv2.line(img, tuple(pts[j]), tuple(pts[j + 4]), color, max(1, thick - 1))


def box_corners(center, dims, yaw):
    """SECOND/MMDet3D LiDARInstance3DBoxes -> 8 corners in lidar frame.
    center: (x, y, z) box center
    dims:   gt_boxes[3:6] from converter = (w, l, h), which MMDet3D stores as
            (x_size=w, y_size=l, z_size=h). yaw=0 means width axis along +x_lidar.
    yaw:    rotation around z (radians), SECOND-format (yaw = -nusc_yaw - pi/2).
    """
    w, l, h = dims  # (x_size, y_size, z_size)
    # corners in box frame: x=x_size=width, y=y_size=length, z=z_size=height
    x_corners = np.array([ w/2,  w/2, -w/2, -w/2,  w/2,  w/2, -w/2, -w/2])
    y_corners = np.array([ l/2, -l/2, -l/2,  l/2,  l/2, -l/2, -l/2,  l/2])
    z_corners = np.array([-h/2, -h/2, -h/2, -h/2,  h/2,  h/2,  h/2,  h/2])
    corners = np.stack([x_corners, y_corners, z_corners], axis=0)  # 3x8

    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0],
                  [s,  c, 0],
                  [0,  0, 1]])
    corners = R @ corners
    corners[0, :] += center[0]
    corners[1, :] += center[1]
    corners[2, :] += center[2]
    return corners.T  # 8x3


def build_lidar2img(cam_info):
    """Identical to projects/mmdet3d_plugin/datasets/nuscenes_dataset.py:get_data_info."""
    lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
    lidar2cam_t = cam_info['sensor2lidar_translation'] @ lidar2cam_r.T
    lidar2cam_rt = np.eye(4)
    lidar2cam_rt[:3, :3] = lidar2cam_r.T
    lidar2cam_rt[3, :3] = -lidar2cam_t
    intrinsic = cam_info['cam_intrinsic']
    viewpad = np.eye(4)
    viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
    lidar2img_rt = (viewpad @ lidar2cam_rt.T)
    return lidar2img_rt, lidar2cam_rt.T


def project_points(pts_lidar, lidar2img):
    """pts_lidar: Nx3 in lidar frame.
    Returns Nx2 pixel coords and N depth (cam z, positive = in front of cam)."""
    n = pts_lidar.shape[0]
    homo = np.concatenate([pts_lidar, np.ones((n, 1))], axis=1)  # Nx4
    cam_homo = homo @ lidar2img.T  # Nx4
    # cam_homo[:, 2] is depth before normalization
    z = cam_homo[:, 2]
    uv = cam_homo[:, :2] / np.where(np.abs(z) < 1e-6, 1e-6, z)[:, None]
    return uv, z


def check_one_sample(info, out_dir, save_imgs=True, vis_compare=False):
    """Returns dict: cam -> (n_visible, n_projected, n_in_front, n_in_image).

    vis_compare=True: render ALL boxes colored by visibility_token (needs
    info['vis_tokens']) so the valid_flag filter (keep vis>='2') can be eyeballed
    -- gray boxes are the vis=1 (0-40%) boxes that get DROPPED.
    """
    boxes = info['gt_boxes']           # Nx7  [x,y,z,w,l,h,yaw]
    names = info['gt_names']
    valid = info['valid_flag']
    vis_tokens = info.get('vis_tokens')
    sample_tok = info['token'][:8]
    H, W = 900, 1600

    stats = {}
    for cam_name in CAM_NAMES:
        cam_info = info['cams'][cam_name]
        img_path = cam_info['data_path']
        lidar2img, _ = build_lidar2img(cam_info)

        # Project all valid boxes (for the stats table)
        centers = boxes[valid][:, :3]
        if len(centers) == 0:
            stats[cam_name] = (0, 0, 0, 0)
        else:
            uv, z = project_points(centers, lidar2img)
            in_front = z > 0.5
            in_img = ((uv[:, 0] >= 0) & (uv[:, 0] < W)
                      & (uv[:, 1] >= 0) & (uv[:, 1] < H))
            visible = in_front & in_img
            stats[cam_name] = (int(valid.sum()), len(centers),
                               int(in_front.sum()), int(visible.sum()))

        if not save_imgs or not osp.isfile(img_path):
            continue
        img = cv2.imread(img_path)
        if img is None:
            continue

        if vis_compare and vis_tokens is not None:
            # Draw EVERY box that lands in this camera, colored by visibility.
            uv_all, z_all = project_points(boxes[:, :3], lidar2img)
            for i in range(len(boxes)):
                if z_all[i] <= 0.5:
                    continue
                if not (0 <= uv_all[i, 0] < W and 0 <= uv_all[i, 1] < H):
                    continue
                lvl = str(vis_tokens[i])
                color = VIS_COLORS.get(lvl, (255, 255, 255))
                corners = box_corners(boxes[i, :3], boxes[i, 3:6], boxes[i, 6])
                cuv, cz = project_points(corners, lidar2img)
                if (cz > 0.5).all():
                    _draw_3d_box(img, cuv.astype(int), color,
                                 thick=1 if lvl == '1' else 2)
                c = tuple(uv_all[i].astype(int))
                cv2.circle(img, c, 4, color, -1)
                cv2.putText(img, f'v{lvl}', (c[0] + 4, c[1] - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            out = osp.join(out_dir, f'{sample_tok}_{cam_name}.jpg')
            cv2.imwrite(out, img)
            continue

        # Default: draw only valid (kept) boxes, colored by class.
        if len(centers) == 0:
            continue
        names_v = names[valid]
        boxes_v = boxes[valid]
        for i in range(len(boxes_v)):
            if not visible[i]:
                continue
            cls = names_v[i]
            color = COLORS.get(cls, (255, 255, 255))
            corners = box_corners(boxes_v[i, :3], boxes_v[i, 3:6], boxes_v[i, 6])
            cuv, cz = project_points(corners, lidar2img)
            if (cz > 0.5).all():
                _draw_3d_box(img, cuv.astype(int), color, thick=2)
            c = tuple(uv[i].astype(int))
            cv2.circle(img, c, 5, color, -1)
            cv2.putText(img, cls, (c[0]+5, c[1]-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        out = osp.join(out_dir, f'{sample_tok}_{cam_name}.jpg')
        cv2.imwrite(out, img)
    return stats


def infos_from_db(version, root_path, n_samples):
    """Build a few info dicts straight from a CARLA DB via the converter's own
    per-sample logic -- no full conversion / pkl needed. Lets us verify the
    GT<->mmdet3d coordinate convention BEFORE running the heavy converter.

    Reuses create_carla_data._process_sample_token so the boxes/extrinsics are
    produced by EXACTLY the same code path the real pkl uses.
    """
    import create_carla_data as cc  # same tools/ dir
    from nuscenes.nuscenes import NuScenes

    print(f'\n{"="*70}\nLoading DB {version} from {root_path} (direct, no pkl)')
    cc._NUSC = NuScenes(version=version, dataroot=root_path, verbose=False)
    cc._ROOT, cc._VERSION = root_path, version
    sample_tokens = [s['token'] for s in cc._NUSC.sample]
    print(f'samples in DB: {len(sample_tokens)}')
    idx = np.linspace(0, len(sample_tokens) - 1, n_samples).astype(int)
    infos = []
    for i in idx:
        tok = sample_tokens[i]
        _, info, _, _ = cc._process_sample_token(tok)
        # Attach raw visibility tokens aligned with gt_boxes (annotation order ==
        # box order) so vis_compare can color/keep-drop by visibility level.
        sample = cc._NUSC.get('sample', tok)
        vt = [cc._NUSC.get('sample_annotation', t).get('visibility_token', '1')
              for t in sample['anns']]
        if len(vt) == len(info['gt_boxes']):
            info['vis_tokens'] = np.array(vt)
        infos.append(info)
    return infos


def _run_checks(infos, out_dir, n_samples, vis_compare=False):
    # Picks: take samples spread across the file
    idx = np.linspace(0, len(infos)-1, n_samples).astype(int)
    print(f'inspecting sample indices: {idx.tolist()}')

    os.makedirs(out_dir, exist_ok=True)
    print(f'output dir: {out_dir}')

    print('\nPer-camera projection visibility (valid_box_count, projected, in_front_cam, in_image):')
    print(f'{"sample_idx":12s}{"cam":18s}{"n_valid":>10s}{"projected":>12s}'
          f'{"in_front":>10s}{"in_image":>10s}')
    total_by_cam = {c: 0 for c in CAM_NAMES}
    for i in idx:
        info = infos[i]
        stats = check_one_sample(info, out_dir, save_imgs=True,
                                 vis_compare=vis_compare)
        for cam_name in CAM_NAMES:
            n_valid, n_proj, n_front, n_img = stats[cam_name]
            total_by_cam[cam_name] += n_img
            print(f'{i:<12d}{cam_name:18s}{n_valid:>10d}{n_proj:>12d}'
                  f'{n_front:>10d}{n_img:>10d}')

    # Aggregate over many samples (no image saving) — sanity stats
    print('\n--- Aggregate (no image render) over 200 random samples ---')
    rng = np.random.RandomState(0)
    sample_idx = rng.choice(len(infos), size=min(200, len(infos)), replace=False)
    agg = {c: [] for c in CAM_NAMES}
    for i in sample_idx:
        stats = check_one_sample(infos[i], out_dir, save_imgs=False)
        for cam_name in CAM_NAMES:
            agg[cam_name].append(stats[cam_name][3])
    print(f'{"cam":18s} mean_visible_boxes/frame  total')
    for cam_name in CAM_NAMES:
        a = np.array(agg[cam_name])
        print(f'  {cam_name:16s} {a.mean():8.2f}                {a.sum()}')

    return infos


def check_pkl(pkl_path, out_dir, n_samples=4, vis_compare=False):
    print(f'\n{"="*70}\nLoading {pkl_path}')
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    infos = data['infos']
    print(f'metadata: {data["metadata"]}')
    print(f'total infos: {len(infos)}')
    return _run_checks(infos, out_dir, n_samples, vis_compare)


def check_db(version, root_path, out_dir, n_samples=4, vis_compare=False):
    infos = infos_from_db(version, root_path, n_samples)
    return _run_checks(infos, out_dir, n_samples, vis_compare)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-pkl',
                    default='data/nuscenes/sedan_infos_train.pkl')
    ap.add_argument('--val-pkl',
                    default='data/nuscenes/sedan_infos_val.pkl')
    ap.add_argument('--out-dir', default='work_dirs/proj_check')
    ap.add_argument('--n-samples', type=int, default=4)
    # Direct-from-DB mode: verify coords BEFORE building any pkl.
    ap.add_argument('--db', default=None,
                    help='DB version (e.g. v1.0-carla_sedan_eval) to check '
                         'directly without a pkl')
    ap.add_argument('--root-path', default='data/nuscenes',
                    help='nuScenes dataroot for --db mode')
    ap.add_argument('--vis-compare', action='store_true',
                    help='render ALL boxes colored by visibility level '
                         '(gray=v0-40 dropped, red/orange/green=kept); '
                         'requires --db mode')
    args = ap.parse_args()

    if args.db is not None:
        sub = args.db + ('_viscompare' if args.vis_compare else '')
        check_db(args.db, args.root_path,
                 osp.join(args.out_dir, sub), args.n_samples,
                 vis_compare=args.vis_compare)
        return

    if osp.exists(args.train_pkl):
        check_pkl(args.train_pkl, osp.join(args.out_dir, 'train'),
                  args.n_samples, vis_compare=args.vis_compare)
    if osp.exists(args.val_pkl):
        check_pkl(args.val_pkl, osp.join(args.out_dir, 'val'),
                  args.n_samples, vis_compare=args.vis_compare)


if __name__ == '__main__':
    main()
