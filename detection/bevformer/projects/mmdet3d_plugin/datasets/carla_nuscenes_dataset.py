# Added by the RoboGeo authors, 2026.
# _evaluate_single is adapted from CustomNuScenesDataset._evaluate_single in
# nuscenes_dataset.py (BEVFormer, Apache-2.0), which follows
# NuScenesDataset._evaluate_single of mmdetection3d
# (Copyright (c) OpenMMLab, Apache-2.0).
"""RoboGeo CARLA dataset for BEVFormer.

The RoboGeo detection data uses the nuScenes database format with its own
versions (v1.0-carla_{sedan,suv,bus}[_eval]) and scene names. Loading and
training behave exactly as in CustomNuScenesDataset; only evaluation differs:

* the nuScenes-devkit evaluator is pointed at the CARLA scenes and samples of
  the info file, through patches that are installed for the duration of
  evaluate() only (neither the devkit nor nuscnes_eval.py is modified);
* ground truth is kept iff visibility_token >= 2 (>= 40 % visible), the same
  rule as the training valid_flag, instead of the devkit's num_pts > 0;
* the score is the CARLA Detection Score (CDS): the nuScenes detection score
  recomputed over the 6 RoboGeo classes,
  CDS = (5 mAP + sum over {mATE, mASE, mAOE, mAVE, mAAE} of (1 - min(1, mTP))) / 10.
"""
from os import path as osp

import mmcv
import numpy as np
from mmdet.datasets import DATASETS

from .nuscenes_dataset import CustomNuScenesDataset
from .nuscnes_eval import NuScenesEval_custom

# Evaluation keeps ground truth with visibility_token >= this value.
CARLA_MIN_VISIBILITY = 2


def _patch_nuscenes_eval_for_carla(scene_names, sample_tokens):
    """Point the BEVFormer nuScenes evaluator at the CARLA evaluation samples.

    nuscnes_eval.py binds create_splits_scenes and its own load_gt at import
    time, so both are replaced on that module (and on the devkit modules):

    * create_splits_scenes()['val'] becomes the CARLA scene names;
    * load_gt runs with nusc.version temporarily set to 'v1.0-trainval' (it
      asserts a trainval version), then keeps only the samples of the info
      file and the boxes with visibility >= CARLA_MIN_VISIBILITY. num_pts is
      set to 1 so the devkit's num_pts == 0 filter removes nothing.

    Returns a callable that restores the originals.
    """
    from nuscenes.eval.common import loaders as nusc_loaders
    from nuscenes.eval.common.data_classes import EvalBoxes
    from nuscenes.utils import splits as nusc_splits

    from projects.mmdet3d_plugin.datasets import nuscnes_eval as bev_eval

    original_create = nusc_splits.create_splits_scenes
    original_devkit_load_gt = nusc_loaders.load_gt
    original_bev_load_gt = bev_eval.load_gt

    def patched_create_splits_scenes(verbose=False):
        d = {k: list(v) for k, v in original_create(verbose=verbose).items()}
        d['val'] = list(scene_names)
        return d

    def patched_load_gt(nusc, eval_split, box_cls, verbose=False):
        real_version = nusc.version
        try:
            if not real_version.endswith('trainval'):
                nusc.version = 'v1.0-trainval'
            gt = original_bev_load_gt(nusc, eval_split, box_cls, verbose=verbose)
        finally:
            nusc.version = real_version
        filtered = EvalBoxes()
        for st in gt.sample_tokens:
            if st not in sample_tokens:
                continue
            keep = []
            for b in gt[st]:
                vis = getattr(b, 'visibility', None)
                if vis is None or int(vis) >= CARLA_MIN_VISIBILITY:
                    b.num_pts = 1
                    keep.append(b)
            filtered.add_boxes(st, keep)
        return filtered

    nusc_splits.create_splits_scenes = patched_create_splits_scenes
    nusc_loaders.load_gt = patched_load_gt
    bev_eval.create_splits_scenes = patched_create_splits_scenes
    bev_eval.load_gt = patched_load_gt

    def restore():
        nusc_splits.create_splits_scenes = original_create
        nusc_loaders.load_gt = original_devkit_load_gt
        bev_eval.create_splits_scenes = original_create
        bev_eval.load_gt = original_bev_load_gt

    return restore


@DATASETS.register_module()
class CarlaNuScenesDataset(CustomNuScenesDataset):
    """RoboGeo CARLA data in nuScenes format (v1.0-carla_{sedan,suv,bus}[_eval])."""

    def _evaluate_single(self,
                         result_path,
                         logger=None,
                         metric='bbox',
                         result_name='pts_bbox'):
        from nuscenes import NuScenes

        self.nusc = NuScenes(version=self.version,
                             dataroot=self.data_root,
                             verbose=False)
        output_dir = osp.join(*osp.split(result_path)[:-1])

        if self.version == 'v1.0-mini':
            eval_split = 'mini_val'
        elif self.version == 'v1.0-trainval' or 'carla' in self.version:
            eval_split = 'val'
        else:
            raise KeyError(f'Unknown dataset version for eval: {self.version!r}')

        # Evaluate exactly the samples of the info file. Their scenes are
        # looked up in the database because scene_token in the info file is
        # set to the sample token (this disables temporal fusion).
        sample_tokens = {info['token'] for info in self.data_infos}
        scene_names = sorted({
            self.nusc.get('scene', self.nusc.get('sample', t)['scene_token'])['name']
            for t in sample_tokens})

        restore = _patch_nuscenes_eval_for_carla(scene_names, sample_tokens)
        try:
            self.nusc_eval = NuScenesEval_custom(
                self.nusc,
                config=self.eval_detection_configs,
                result_path=result_path,
                eval_set=eval_split,
                output_dir=output_dir,
                verbose=True,
                overlap_test=self.overlap_test,
                data_infos=self.data_infos,
            )
            self.nusc_eval.main(plot_examples=0, render_curves=False)
        finally:
            restore()

        metrics = mmcv.load(osp.join(output_dir, 'metrics_summary.json'))
        detail = dict()
        metric_prefix = f'{result_name}_NuScenes'
        for name in self.CLASSES:
            for k, v in metrics['label_aps'][name].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}_AP_dist_{}'.format(metric_prefix, name, k)] = val
            for k, v in metrics['label_tp_errors'][name].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}_{}'.format(metric_prefix, name, k)] = val

        # The devkit averages over its 10 nuScenes classes, 4 of which do not
        # exist in RoboGeo. Recompute mAP and the TP errors over self.CLASSES.
        tp_keys = list(metrics['tp_errors'].keys())
        mean_ap = float(np.mean([np.mean(list(metrics['label_aps'][c].values()))
                                 for c in self.CLASSES]))
        tp_errs = {m: float(np.nanmean([metrics['label_tp_errors'][c][m]
                                        for c in self.CLASSES]))
                   for m in tp_keys}
        tp_scores = [max(0.0, 1.0 - tp_errs[m]) for m in tp_keys]
        cds = (5.0 * mean_ap + float(np.sum(tp_scores))) / (5.0 + len(tp_keys))
        for m in tp_keys:
            detail['{}/{}'.format(metric_prefix, self.ErrNameMapping[m])] = \
                round(tp_errs[m], 4)
        detail['{}/mAP'.format(metric_prefix)] = mean_ap
        detail['{}/CDS'.format(metric_prefix)] = cds
        print(f'[RoboGeo] CDS={cds:.4f} mAP={mean_ap:.4f} '
              f'({len(self.CLASSES)} classes, GT visibility >= '
              f'{CARLA_MIN_VISIBILITY}, {len(sample_tokens)} frames)')
        return detail
