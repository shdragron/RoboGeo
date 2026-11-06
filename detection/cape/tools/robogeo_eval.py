# Copyright (c) 2026 The RoboGeo authors.
"""Evaluate the released CAPE checkpoints on one RoboGeo robustness protocol.

    python tools/robogeo_eval.py <vp|cts> OUT [--cells ...] [--subset full|fps16] [--shard K/N]

1. Builds the cells with detection/robustness/build_conditions.py into OUT
   (OUT/cells.json and OUT/pkls/), unless OUT already holds the cells of the
   same --cells and --subset.
2. Evaluates every cell's info pkl with the single-GPU tools/test.py (CAPE's
   test pipeline does not support multi-GPU testing), using the checkpoint of
   the cell's model_platform (cape_<platform>.pth). Inference runs in full fp32
   (TF32 disabled) with deterministic cuDNN, so all cells share the same
   numerics.
3. Writes OUT/results/<cell id with '/' replaced by '__'>.json:
   {"cell", "metric": "CDS", "value", "mAP", "n_frames", "checkpoint_sha256"}

Cells that already have a result are skipped, so an interrupted run resumes
with the same command. --shard K/N evaluates cells K, K+N, K+2N, ... (one
process per GPU); shard 0 builds the cells and the other shards wait for them.
Use a separate OUT per model. tools/robogeo_eval.sh runs this from
detection/cape.
"""
import argparse
import functools
import hashlib
import json
import os
import os.path as osp
import shutil
import subprocess
import sys
import time

CAPE_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
BUILD_CONDITIONS = osp.join(osp.dirname(CAPE_ROOT), 'robustness',
                            'build_conditions.py')
CONFIG = 'projects/configs/CAPE/cape_carla_{}.py'
CKPT = 'detection/cape/cape_{}.pth'
PLATFORMS = ('sedan', 'suv', 'bus')
BUILD_ARGS = 'build_args.json'
# Runs tools/test.py (argv[1:]) with TF32 disabled for convolutions and matmuls.
TEST_FP32 = ('import runpy, sys, torch\n'
             'torch.backends.cudnn.allow_tf32 = False\n'
             'torch.backends.cuda.matmul.allow_tf32 = False\n'
             'sys.argv = sys.argv[1:]\n'
             "runpy.run_path(sys.argv[0], run_name='__main__')\n")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('protocol', choices=['vp', 'cts'])
    ap.add_argument('out', help='output directory (one per model)')
    ap.add_argument('--cells', default='all',
                    help='"all" (default) or comma-separated cell ids')
    ap.add_argument('--subset', default='full', choices=['full', 'fps16'],
                    help='full = all 3,792 evaluation frames (default); '
                         'fps16 = 768-frame subset for quick checks only')
    ap.add_argument('--shard', default='0/1',
                    help='K/N: evaluate cells K, K+N, K+2N, ... (default 0/1)')
    ap.add_argument('--workers', type=int, default=4,
                    help='dataloader workers (default 4)')
    ap.add_argument('--keep-work', action='store_true',
                    help='keep the per-cell prediction files')
    ap.add_argument('--data-root', default=os.environ.get('ROBOGEO_DATA'),
                    help='RoboGeo data root (default: $ROBOGEO_DATA)')
    ap.add_argument('--ckpt-root', default=os.environ.get('ROBOGEO_CKPT'),
                    help='checkpoint root laid out as on the hub '
                         '(default: $ROBOGEO_CKPT)')
    args = ap.parse_args()
    if not args.data_root:
        ap.error('--data-root or $ROBOGEO_DATA is required')
    if not args.ckpt_root:
        ap.error('--ckpt-root or $ROBOGEO_CKPT is required')
    try:
        args.shard_k, args.shard_n = (int(x) for x in args.shard.split('/'))
        assert 0 <= args.shard_k < args.shard_n
    except (ValueError, AssertionError):
        ap.error(f'--shard must be K/N with 0 <= K < N, got {args.shard!r}')
    args.out = osp.abspath(args.out)
    args.data_root = osp.abspath(args.data_root)
    args.ckpt_root = osp.abspath(args.ckpt_root)
    return args


def read_json(path):
    with open(path) as f:
        return json.load(f)


def write_json(path, obj):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def prepare_cells(args):
    """Return the cells of OUT/cells.json, building them first if needed."""
    request = dict(protocol=args.protocol, subset=args.subset,
                   cells=args.cells)
    cells_file = osp.join(args.out, 'cells.json')
    args_file = osp.join(args.out, BUILD_ARGS)

    def ready():
        return (osp.exists(cells_file) and osp.exists(args_file)
                and read_json(args_file) == request)

    if args.shard_k != 0:
        while not ready():
            print(f'waiting for shard 0 to build {cells_file}', flush=True)
            time.sleep(30)
        return read_json(cells_file)
    if ready():
        return read_json(cells_file)

    if osp.exists(args_file):
        old = read_json(args_file)
        results = osp.join(args.out, 'results')
        if ((old['protocol'], old['subset']) != (args.protocol, args.subset)
                and osp.isdir(results) and os.listdir(results)):
            sys.exit(f'{args.out} holds results of protocol {old["protocol"]} '
                     f'/ subset {old["subset"]}; use another output directory')
        os.remove(args_file)
    os.makedirs(args.out, exist_ok=True)
    subprocess.check_call(
        [sys.executable, BUILD_CONDITIONS, '--data-root', args.data_root,
         '--protocol', args.protocol, '--cells', args.cells,
         '--subset', args.subset, '--out', args.out])
    write_json(args_file, request)
    return read_json(cells_file)


@functools.lru_cache(maxsize=None)
def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 24), b''):
            h.update(block)
    return h.hexdigest()


def cell_name(cell_id):
    return cell_id.replace('/', '__')


def eval_cell(cell, args, work_root, log_root, out_dir):
    platform = cell['model_platform']
    if platform not in PLATFORMS:
        raise ValueError(f'unknown model_platform {platform!r}')
    pkl = cell['pkl']
    if not osp.isabs(pkl):
        pkl = osp.join(args.out, pkl)
    config = CONFIG.format(platform)
    ckpt = osp.join(args.ckpt_root, CKPT.format(platform))
    name = cell_name(cell['cell'])
    work = osp.join(work_root, name)
    log = osp.join(log_root, name + '.log')
    shutil.rmtree(work, ignore_errors=True)

    cmd = [sys.executable, '-c', TEST_FP32, 'tools/test.py', config, ckpt,
           '--deterministic', '--eval', 'bbox',
           '--eval-options', f'jsonfile_prefix={work}',
           '--cfg-options', f'data.test.ann_file={pkl}',
           f'data.test.data_root={osp.join(args.data_root, "carla_geobev")}/',
           f'data.workers_per_gpu={args.workers}']
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join(
        p for p in (CAPE_ROOT, env.get('PYTHONPATH')) if p)
    with open(log, 'w') as f:
        f.write(' '.join(cmd) + '\n')
        f.flush()
        rc = subprocess.call(cmd, cwd=CAPE_ROOT, env=env, stdout=f,
                             stderr=subprocess.STDOUT)
    metrics_file = osp.join(work, 'pts_bbox', 'carla_metrics.json')
    if rc != 0 or not osp.exists(metrics_file):
        raise RuntimeError(f'tools/test.py failed (rc={rc}), see {log}')
    m = read_json(metrics_file)
    result = {
        'cell': cell['cell'],
        'metric': 'CDS',
        'value': float(m['CDS']),
        'mAP': float(m['mAP']),
        'n_frames': int(m['n_frames']),
        'checkpoint_sha256': sha256(ckpt),
    }
    write_json(osp.join(out_dir, name + '.json'), result)
    if not args.keep_work:
        shutil.rmtree(work, ignore_errors=True)
    return result


def main():
    args = parse_args()
    cells = prepare_cells(args)[args.shard_k::args.shard_n]

    out_dir = osp.join(args.out, 'results')
    work_root = osp.join(args.out, 'work')
    log_root = osp.join(args.out, 'logs')
    for d in (out_dir, work_root, log_root):
        os.makedirs(d, exist_ok=True)

    failed = []
    for i, cell in enumerate(cells, 1):
        tag = f'[{i}/{len(cells)}] {cell["cell"]}'
        if osp.exists(osp.join(out_dir, cell_name(cell['cell']) + '.json')):
            print(f'{tag}: done, skipped', flush=True)
            continue
        t0 = time.time()
        try:
            r = eval_cell(cell, args, work_root, log_root, out_dir)
            print(f'{tag}: CDS={r["value"]:.4f} mAP={r["mAP"]:.4f} '
                  f'n={r["n_frames"]} ({time.time() - t0:.0f}s)', flush=True)
        except Exception as e:  # keep going; report at the end
            failed.append(cell['cell'])
            print(f'{tag}: FAILED: {e}', flush=True)
    if failed:
        sys.exit(f'{len(failed)} cell(s) failed: {failed}')


if __name__ == '__main__':
    main()
