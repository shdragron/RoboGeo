# Robustness protocols for the detectors

Framework-free tools (Python standard library, numpy and pyquaternion) shared by the
detectors in this repository:

- `build_conditions.py` writes one mmdet3d info file per evaluation cell;
- each model's runner evaluates the cells and writes one result file per cell
  (BEVFormer: [`../bevformer/tools/robogeo_eval.py`](../bevformer/tools/robogeo_eval.py));
- `aggregate.py` turns the per-cell results into the robustness scores.

## Protocols and cells

**Viewpoint perturbation** (protocol `vp`): the Sedan model on the Sedan evaluation set, with
one camera (6 modes) or all cameras (mode `all`) rotated by
{roll, pitch, yaw} x {±4, ±8, ±12, ±16, ±20}°.

| condition | images | extrinsics |
|---|---|---|
| NORMAL | stock | stock |
| EXT | stock | perturbed |
| IMG | perturbed | stock |
| CAL | perturbed | perturbed (consistent calibration) |

Cell ids: `vp/NORMAL` and `vp/<EXT|IMG|CAL>/<camera|all>/<roll|pitch|yaw><+|-><deg>`, for
example `vp/IMG/CAM_FRONT/pitch+12`; 631 cells in total. Perturbed images come from
`carla_VR/sweeps/RGB-<CAM>_yaw<y>pitch<p>roll<r>/`; carla_geobev frame N corresponds to
carla_VR frame 2N. Perturbed extrinsics come from `carla_VR/viewpoint_metadata.json`; both
`sensor2lidar_*` and `sensor2ego_*` are replaced and checked against each other.

**Cross-platform transfer** (protocol `cts`): the Sedan model on the SUV or Bus evaluation set.

| cell | images | extrinsics | checkpoint |
|---|---|---|---|
| `cts/<suv\|bus>/IMG` | target platform | Sedan | Sedan |
| `cts/<suv\|bus>/CAL` | target platform | target platform | Sedan |
| `cts/<suv\|bus>/ORACLE` | target platform | target platform | target platform |

Ground truth, sample tokens and the database version are always those of the evaluated
platform.

## Usage

```bash
python build_conditions.py --data-root $ROBOGEO_DATA --protocol vp|cts \
    [--cells all|<comma list of cell ids>] [--subset full|fps16] --out DIR
```
`--data-root` holds `carla_geobev/` and `carla_VR/`. The default `--subset full` uses all
3,792 evaluation frames (the benchmark protocol). `--subset fps16` uses the 768-frame
`{platform}_infos_val_fps16.pkl` lists and is meant for quick checks only.

Outputs: `DIR/cells.json`, a list of
`{cell, protocol, condition, mode, axis, angle, target, model_platform, pkl}`, and one info file
per cell in `DIR/pkls/` with absolute image paths. `model_platform` names the checkpoint that
evaluates the cell (Sedan for every cell except `cts/<r>/ORACLE`). The full viewpoint grid
writes about 33 GB of info files (5 GB for `fps16`).

A runner writes `DIR/results/<cell id with / replaced by __>.json`:
```
{"cell": "vp/IMG/CAM_FRONT/pitch+12", "metric": "CDS", "value": <float>, "mAP": <float>,
 "n_frames": <int>, "checkpoint_sha256": "<sha256>"}
```

```bash
python aggregate.py --results DIR/results --out summary.json [--allow-partial]
```
computes, in percent:
- VRS(m)_c = mean over the 30 (axis, angle) cells of P_c / P_NORMAL, for each mode m;
- mVRS_c = mean of VRS over the 7 modes = (6 x mean per-camera VRS + all-camera VRS) / 7;
- CTS(r)_c = P_c(r) / P_ORACLE(r) for c in {IMG, CAL} and r in {suv, bus}, with P_ORACLE
  measured from the target-platform checkpoint in the same run.

It refuses an incomplete grid unless `--allow-partial` is given, and checks that all cells use
one metric, that the viewpoint cells share their frame count and that every non-oracle cell
was evaluated with the same checkpoint. A result set on fewer than 3,792 frames is marked as a
subset in the summary.
