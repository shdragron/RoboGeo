"""Aggregate per-cell results into the RoboGeo robustness scores.

Input: a directory of per-cell JSON files
  {"cell", "metric", "value", "n_frames", "checkpoint_sha256"[, "mAP"]}
as written by the model runners.

Scores (reported in percent):
  VRS(m)_c  = mean over the 30 (axis, angle) cells of P_c / P_NORMAL, per mode m
  mVRS_c    = mean of VRS over the 7 modes = (6 * mean per-camera VRS + all-camera VRS) / 7
  CTS(r)_c  = P_c(r) / P_ORACLE(r), for c in {IMG, CAL} and r in {suv, bus}

A protocol whose grid is incomplete is refused unless --allow-partial is given.

Usage:
  python aggregate.py --results DIR/results --out summary.json [--allow-partial]
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_conditions import (CTS_CONDITIONS, CTS_TARGETS, VP_ANGLES, VP_AXES,
                              VP_CONDITIONS, VP_MODES, all_cells, parse_cell,
                              vp_cell_id)

FULL_FRAMES = 3792


def load_results(results_dir):
    res = {}
    for path in sorted(glob.glob(os.path.join(results_dir, '*.json'))):
        with open(path) as f:
            r = json.load(f)
        parse_cell(r['cell'])                      # validates the id
        if r['cell'] in res:
            sys.exit(f'duplicate result for {r["cell"]}: {path}')
        res[r['cell']] = r
    return res


def mean(xs):
    return sum(xs) / len(xs)


def check_same(res, cells, field, what):
    vals = {res[c][field] for c in cells if c in res}
    if len(vals) > 1:
        sys.exit(f'{what}: cells disagree on {field}: {sorted(map(str, vals))}')
    return vals.pop() if vals else None


def aggregate_vp(res, allow_partial):
    grid = all_cells('vp')
    present = [c for c in grid if c in res]
    if not present:
        return None
    missing = [c for c in grid if c not in res]
    if 'vp/NORMAL' not in res:
        sys.exit('viewpoint perturbation: vp/NORMAL is missing (P_NORMAL)')
    if missing and not allow_partial:
        sys.exit(f'viewpoint perturbation: {len(missing)} of {len(grid)} cells '
                 f'missing (e.g. {missing[:3]}); use --allow-partial')
    n_frames = check_same(res, present, 'n_frames', 'viewpoint perturbation')
    p_normal = res['vp/NORMAL']['value']

    out = dict(P_NORMAL=p_normal, n_frames=n_frames,
               complete=not missing, n_cells=len(present),
               VRS={}, mVRS={}, n_cells_per_mode={})
    for cond in VP_CONDITIONS:
        vrs, counts = {}, {}
        for mode in VP_MODES:
            vals = [res[c]['value'] / p_normal
                    for c in (vp_cell_id(cond, mode, a, t)
                              for a in VP_AXES for t in VP_ANGLES)
                    if c in res]
            counts[mode] = len(vals)
            if vals:
                vrs[mode] = 100.0 * mean(vals)
        out['VRS'][cond] = vrs
        out['n_cells_per_mode'][cond] = counts
        if len(vrs) == len(VP_MODES):
            per_cam = mean([vrs[m] for m in VP_MODES if m != 'all'])
            out['mVRS'][cond] = (6.0 * per_cam + vrs['all']) / 7.0
        else:
            out['mVRS'][cond] = None
    if n_frames != FULL_FRAMES:
        out['note'] = (f'{n_frames}-frame subset: a quick check, not the '
                       f'benchmark protocol ({FULL_FRAMES} frames)')
    return out


def aggregate_cts(res, allow_partial):
    grid = all_cells('cts')
    present = [c for c in grid if c in res]
    if not present:
        return None
    missing = [c for c in grid if c not in res]
    if missing and not allow_partial:
        sys.exit(f'cross-platform transfer: missing cells {missing}; '
                 f'use --allow-partial')
    out = dict(P_ORACLE={}, P={}, CTS={}, complete=not missing)
    for t in CTS_TARGETS:
        cells = [f'cts/{t}/{c}' for c in CTS_CONDITIONS]
        check_same(res, cells, 'n_frames', f'cross-platform transfer ({t})')
        oracle = res.get(f'cts/{t}/ORACLE')
        out['P'][t] = {c: res[f'cts/{t}/{c}']['value']
                       for c in CTS_CONDITIONS if f'cts/{t}/{c}' in res}
        if oracle is None:
            continue
        out['P_ORACLE'][t] = oracle['value']
        out['CTS'][t] = {c: 100.0 * res[f'cts/{t}/{c}']['value'] / oracle['value']
                         for c in ('IMG', 'CAL') if f'cts/{t}/{c}' in res}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--results', required=True, help='directory of per-cell JSON files')
    ap.add_argument('--out', required=True, help='summary JSON to write')
    ap.add_argument('--allow-partial', action='store_true',
                    help='score incomplete grids from the cells present')
    args = ap.parse_args()

    res = load_results(args.results)
    if not res:
        sys.exit(f'no results in {args.results}')
    metrics = {r['metric'] for r in res.values()}
    if len(metrics) != 1:
        sys.exit(f'mixed metrics: {sorted(metrics)}')
    # Every cell except cts/<r>/ORACLE is evaluated with the Sedan checkpoint.
    check_same(res, [c for c in res if not c.endswith('/ORACLE')],
               'checkpoint_sha256', 'Sedan-checkpoint cells')

    summary = dict(metric=metrics.pop(), units='VRS, mVRS and CTS in percent',
                   viewpoint_perturbation=aggregate_vp(res, args.allow_partial),
                   cross_platform_transfer=aggregate_cts(res, args.allow_partial))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(summary, f, indent=1)

    vp, cts = summary['viewpoint_perturbation'], summary['cross_platform_transfer']
    if vp:
        print(f'P_NORMAL ({summary["metric"]}, {vp["n_frames"]} frames) = {vp["P_NORMAL"]:.5f}')
        for cond, v in vp['mVRS'].items():
            per_mode = ' '.join(f'{m}={x:.1f}' for m, x in vp['VRS'][cond].items())
            mv = 'n/a' if v is None else f'{v:.1f}'
            print(f'  {cond}: mVRS={mv}  [{per_mode}]')
        if vp.get('note'):
            print(f'  note: {vp["note"]}')
    if cts:
        for t, d in cts['CTS'].items():
            print(f'CTS {t}: oracle={cts["P_ORACLE"][t]:.5f}  '
                  + '  '.join(f'{c}={x:.1f}' for c, x in d.items()))
    print(f'summary -> {args.out}')


if __name__ == '__main__':
    main()
