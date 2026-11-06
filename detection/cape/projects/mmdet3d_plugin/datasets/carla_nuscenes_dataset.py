# Copyright (c) 2026 The RoboGeo authors.
# Adapted from mmdet3d (Apache-2.0), https://github.com/open-mmlab/mmdetection3d
"""RoboGeo (CARLA) dataset for CAPE.

Training and data loading are inherited unchanged from CAPE's
CustomNuScenesDataset (which builds lidar2img from each camera's
sensor2lidar_* and cam_intrinsic). Only evaluation is adapted so that the
nuscenes-devkit detection evaluator runs on the RoboGeo evaluation databases
(v1.0-carla_<platform>_eval):

1. Any 'v1.0-carla*' version is evaluated as the 'val' split, and the ground
   truth is restricted to the samples in the info pkl (all frames of its scenes
   for the full evaluation split, or a frame subset such as fps16).
2. The devkit's trainval-only assertion in load_gt is bypassed.
3. GT validity is visibility >= 2 (the same rule as the training annotations,
   use_valid_flag=True). Each GT annotation gets num_lidar_pts = 1 if its
   visibility_token >= 2 else 0, so the devkit's num_pts == 0 filter acts as the
   visibility filter.
4. mAP and the CARLA Detection Score (CDS) are computed over the 6 RoboGeo
   classes: CDS = (5 mAP + sum over the 5 TP errors of (1 - min(1, err))) / 10.

All patches are scoped to the evaluate() call; installed packages are not
modified.
"""
from os import path as osp

import mmcv
import numpy as np
from mmdet.datasets import DATASETS

from .nuscenes_dataset import CustomNuScenesDataset

# GT annotations with visibility_token below this value are ignored in evaluation.
CARLA_MIN_VISIBILITY = 2


def _patch_nuscenes_eval_for_carla(carla_val_scene_names, sample_tokens):
    """Patch devkit splits + load_gt for CARLA eval. Returns a restore()."""
    from nuscenes.utils import splits as nusc_splits
    from nuscenes.eval.common import loaders as nusc_loaders
    import nuscenes.eval.detection.evaluate as nusc_eval_mod

    orig_create = nusc_splits.create_splits_scenes
    orig_load_gt = nusc_loaders.load_gt

    def patched_create_splits_scenes(verbose=False):
        d = {k: list(v) for k, v in orig_create(verbose=verbose).items()}
        d['val'] = list(carla_val_scene_names)
        return d

    def patched_load_gt(nusc, eval_split, box_cls, verbose=False):
        real_version = nusc.version
        try:
            if not real_version.endswith('trainval'):
                nusc.version = 'v1.0-trainval'
            gt = orig_load_gt(nusc, eval_split, box_cls, verbose=verbose)
        finally:
            nusc.version = real_version
        for token in list(gt.boxes):
            if token not in sample_tokens:
                del gt.boxes[token]
        return gt

    nusc_splits.create_splits_scenes = patched_create_splits_scenes
    nusc_loaders.load_gt = patched_load_gt
    # loaders.load_gt and NuScenesEval call the names bound in their own module
    # namespaces at import time, so patch those bindings as well.
    nusc_loaders.create_splits_scenes = patched_create_splits_scenes
    nusc_eval_mod.load_gt = patched_load_gt
    nusc_eval_mod.create_splits_scenes = patched_create_splits_scenes

    def restore():
        nusc_splits.create_splits_scenes = orig_create
        nusc_loaders.load_gt = orig_load_gt
        nusc_loaders.create_splits_scenes = orig_create
        nusc_eval_mod.load_gt = orig_load_gt
        nusc_eval_mod.create_splits_scenes = orig_create

    return restore


@DATASETS.register_module()
class CarlaNuScenesDataset(CustomNuScenesDataset):
    """RoboGeo CARLA dataset (v1.0-carla_<platform>[_eval]) for CAPE.

    Same loading and training behaviour as CustomNuScenesDataset; evaluation
    reports the 6-class mAP and CDS with the visibility >= 2 GT filter.
    """

    def format_results(self, results, jsonfile_prefix=None, **kwargs):
        """Format predictions with the box convention of the CARLA GT.

        mmdet3d 1.0.x ``output_to_nusc_box`` swaps dims to [1, 0, 2] and uses a
        yaw convention that rotates boxes by 90 degrees relative to the CARLA
        GT. Within this call the classic convention (dims as-is,
        ``yaw = -yaw - pi/2``, as in mmdet3d 0.17.x) is used instead, so mASE
        and mAOE are correct. Only the exported boxes change; installed
        mmdet3d code is not modified.
        """
        import pyquaternion
        import mmdet3d.datasets.nuscenes_dataset as _nd
        _orig = _nd.output_to_nusc_box

        def _carla_output_to_nusc_box(detection, *args, **kw):
            boxes = _orig(detection, *args, **kw)
            dims = detection['boxes_3d'].dims.numpy()
            yaw = detection['boxes_3d'].yaw.numpy()
            for i, box in enumerate(boxes):
                box.wlh = dims[i]
                box.orientation = pyquaternion.Quaternion(
                    axis=[0, 0, 1], radians=float(-yaw[i] - np.pi / 2))
            return boxes

        _nd.output_to_nusc_box = _carla_output_to_nusc_box
        try:
            return super().format_results(results, jsonfile_prefix, **kwargs)
        finally:
            _nd.output_to_nusc_box = _orig

    def _evaluate_single(self,
                         result_path,
                         logger=None,
                         metric='bbox',
                         result_name='pts_bbox'):
        from nuscenes import NuScenes
        from nuscenes.eval.detection.evaluate import NuScenesEval

        output_dir = osp.join(*osp.split(result_path)[:-1])
        nusc = NuScenes(version=self.version, dataroot=self.data_root,
                        verbose=False)

        # visibility >= 2 GT filter through the devkit's num_pts == 0 rule.
        for _ann in nusc.sample_annotation:
            _vis = int(_ann.get('visibility_token', '1'))
            _ann['num_lidar_pts'] = 1 if _vis >= CARLA_MIN_VISIBILITY else 0
            _ann['num_radar_pts'] = 0

        eval_set_map = {'v1.0-mini': 'mini_val', 'v1.0-trainval': 'val'}
        if self.version in eval_set_map:
            eval_split = eval_set_map[self.version]
        elif 'carla' in self.version:
            eval_split = 'val'
        else:
            raise KeyError(f'Unknown version for eval: {self.version!r}')

        # Evaluate exactly the samples present in the info pkl.
        eval_sample_tokens = {info['token'] for info in self.data_infos}
        eval_scene_tokens = {nusc.get('sample', t)['scene_token']
                             for t in eval_sample_tokens}
        carla_val_scene_names = [nusc.get('scene', st)['name']
                                 for st in eval_scene_tokens]
        restore = _patch_nuscenes_eval_for_carla(carla_val_scene_names,
                                                 eval_sample_tokens)
        try:
            nusc_eval = NuScenesEval(
                nusc,
                config=self.eval_detection_configs,
                result_path=result_path,
                eval_set=eval_split,
                output_dir=output_dir,
                verbose=False)
            nusc_eval.main(render_curves=False)
        finally:
            restore()

        metrics = mmcv.load(osp.join(output_dir, 'metrics_summary.json'))
        detail = dict()
        metric_prefix = f'{result_name}_NuScenes'
        for name in self.CLASSES:
            for k, v in metrics['label_aps'][name].items():
                detail['{}/{}_AP_dist_{}'.format(metric_prefix, name, k)] = \
                    float('{:.4f}'.format(v))
            for k, v in metrics['label_tp_errors'][name].items():
                detail['{}/{}_{}'.format(metric_prefix, name, k)] = \
                    float('{:.4f}'.format(v))

        # The devkit averages over its 10 nuScenes classes; recompute mAP and
        # the TP errors over self.CLASSES only.
        tp_keys = list(metrics['tp_errors'].keys())
        mean_dist_aps = [np.mean(list(metrics['label_aps'][c].values()))
                         for c in self.CLASSES]
        mAP = float(np.mean(mean_dist_aps))
        tp_errs = {m: float(np.nanmean(
            [metrics['label_tp_errors'][c][m] for c in self.CLASSES]))
            for m in tp_keys}
        tp_scores = [max(0.0, 1.0 - tp_errs[m]) for m in tp_keys]
        cds = (5.0 * mAP + float(np.sum(tp_scores))) / (5.0 + len(tp_keys))
        detail['{}/CDS'.format(metric_prefix)] = cds
        detail['{}/mAP'.format(metric_prefix)] = mAP
        for m in tp_keys:
            detail['{}/{}'.format(metric_prefix, self.ErrNameMapping[m])] = \
                round(tp_errs[m], 4)
        print(f'[CARLA-EVAL] CDS={cds:.4f} mAP={mAP:.4f} '
              f'({len(self.CLASSES)} classes, visibility >= '
              f'{CARLA_MIN_VISIBILITY}, {len(self.data_infos)} frames)')
        mmcv.dump(dict(CDS=cds, mAP=mAP, n_frames=len(self.data_infos),
                       version=self.version, metrics=detail),
                  osp.join(output_dir, 'carla_metrics.json'))
        return detail
