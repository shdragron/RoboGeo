# Added by the RoboGeo authors, 2026.
"""Evaluate RoboGeo robustness cells with the released BEVFormer-tiny checkpoints.

Reads the cells.json written by detection/robustness/build_conditions.py,
evaluates every cell pkl on one GPU with the checkpoint of its
model_platform, and writes one result file per cell:

  <out>/<cell id with / replaced by __>.json
  {"cell", "metric": "CDS", "value", "mAP", "n_frames", "checkpoint_sha256"}

Inference runs at batch 1 with cuDNN autotuning and TF32 disabled, so
repeated runs give identical scores. Cells that already have a result file
are skipped (use --overwrite to recompute them).

Usage (from detection/bevformer):
  python tools/robogeo_eval.py --cells DIR/cells.json \
      --data-root $ROBOGEO_DATA --ckpt-root $ROBOGEO_CKPT [--out DIR/results] [--shard i/n]

For several GPUs, start one process per GPU with CUDA_VISIBLE_DEVICES=i and --shard i/n.
"""
import argparse
import copy
import hashlib
import importlib
import json
import os
import os.path as osp
import shutil
import sys
import time

import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint

ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, ROOT)

CONFIGS = {
    'sedan': 'projects/configs/bevformer/bevformer_tiny_carla.py',
    'suv': 'projects/configs/bevformer/bevformer_tiny_carla_suv.py',
    'bus': 'projects/configs/bevformer/bevformer_tiny_carla_bus.py',
}
CKPT = 'detection/bevformer_tiny/bevformer_tiny_{}.pth'


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--cells', required=True, help='cells.json from build_conditions.py')
    ap.add_argument('--data-root', required=True,
                    help='RoboGeo root holding carla_geobev/ (the nuScenes databases)')
    ap.add_argument('--ckpt-root', required=True,
                    help='checkpoint root laid out as the Hugging Face model repository')
    ap.add_argument('--out', default=None, help='result directory (default: <cells dir>/results)')
    ap.add_argument('--only', default=None,
                    help='comma-separated cell ids to evaluate (default: all in cells.json)')
    ap.add_argument('--shard', default='0/1',
                    help='i/n: evaluate every n-th cell starting at i, e.g. one shard per GPU')
    ap.add_argument('--workers', type=int, default=8, help='dataloader workers')
    ap.add_argument('--overwrite', action='store_true')
    return ap.parse_args()


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 24), b''):
            h.update(chunk)
    return h.hexdigest()


def set_deterministic():
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def build_detector(platform, ckpt_root):
    from mmdet3d.models import build_model
    cfg = Config.fromfile(osp.join(ROOT, CONFIGS[platform]))
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    ckpt_path = osp.join(ckpt_root, CKPT.format(platform))
    # strict=True: any missing or unexpected key is an error
    ckpt = load_checkpoint(model, ckpt_path, map_location='cpu', strict=True)
    model.CLASSES = ckpt['meta']['CLASSES']
    model = MMDataParallel(model.cuda(), device_ids=[0])
    model.eval()
    print(f'[{platform}] loaded {ckpt_path} (strict)', flush=True)
    return cfg, model, sha256(ckpt_path)


def evaluate_cell(cfg, model, cell, data_root, workers, tmp_dir):
    from mmdet3d.datasets import build_dataset
    from projects.mmdet3d_plugin.datasets.builder import build_dataloader
    test_cfg = copy.deepcopy(cfg.data.test)
    test_cfg.ann_file = cell['pkl']
    test_cfg.data_root = osp.join(data_root, 'carla_geobev') + '/'
    test_cfg.test_mode = True
    dataset = build_dataset(test_cfg)
    loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=workers,
                              dist=False, shuffle=False)
    outputs = []
    for data in loader:
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        outputs.extend(result)
    detail = dataset.evaluate(outputs, metric='bbox',
                              jsonfile_prefix=osp.join(tmp_dir, 'eval'))
    return dict(cell=cell['cell'], metric='CDS',
                value=float(detail['pts_bbox_NuScenes/CDS']),
                mAP=float(detail['pts_bbox_NuScenes/mAP']),
                n_frames=len(dataset))


def main():
    args = parse_args()
    with open(args.cells) as f:
        cells = json.load(f)
    if args.only:
        wanted = {c.strip() for c in args.only.split(',') if c.strip()}
        unknown = wanted - {c['cell'] for c in cells}
        if unknown:
            sys.exit(f'cells not in {args.cells}: {sorted(unknown)}')
        cells = [c for c in cells if c['cell'] in wanted]
    shard, n_shards = (int(x) for x in args.shard.split('/'))
    cells = cells[shard::n_shards]
    out_dir = args.out or osp.join(osp.dirname(osp.abspath(args.cells)), 'results')
    os.makedirs(out_dir, exist_ok=True)
    data_root = osp.abspath(args.data_root)

    set_deterministic()
    importlib.import_module('projects.mmdet3d_plugin')

    todo = []
    for cell in cells:
        path = osp.join(out_dir, cell['cell'].replace('/', '__') + '.json')
        if args.overwrite or not osp.exists(path):
            todo.append((cell, path))
    print(f'{len(todo)} of {len(cells)} cells to evaluate -> {out_dir}', flush=True)

    models = {}
    for n, (cell, path) in enumerate(todo, 1):
        platform = cell['model_platform']
        if platform not in models:
            models[platform] = build_detector(platform, args.ckpt_root)
        cfg, model, ckpt_sha = models[platform]
        tmp_dir = osp.join(out_dir, '.tmp', cell['cell'].replace('/', '__'))
        t0 = time.time()
        res = evaluate_cell(cfg, model, cell, data_root, args.workers, tmp_dir)
        res['checkpoint_sha256'] = ckpt_sha
        with open(path + '.tmp', 'w') as f:
            json.dump(res, f)
        os.replace(path + '.tmp', path)
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print(f'[{n}/{len(todo)}] {cell["cell"]}: CDS={res["value"]:.5f} '
              f'mAP={res["mAP"]:.5f} ({res["n_frames"]} frames, '
              f'{time.time() - t0:.0f} s)', flush=True)
    try:
        os.rmdir(osp.join(out_dir, '.tmp'))   # only if no other shard is using it
    except OSError:
        pass


if __name__ == '__main__':
    main()
