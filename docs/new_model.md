# Benchmarking a new model

To add a model to the benchmark:
1. Train it on the Sedan training split.
2. Train it on the SUV and Bus training splits as well, for the cross-platform transfer oracles.
3. Evaluate the trained checkpoints on every cell of the two protocols, then aggregate the results.

## 3D object detection

1. **Build the cells.** Each cell gets an mmdet3d info file in the format of
   `{platform}_infos_val.pkl`. The perturbed image paths and extrinsics are already filled in, so
   a model that reads nuScenes-style info files needs no other change.

   ```bash
   python detection/robustness/build_conditions.py --data-root $ROBOGEO_DATA --protocol vp  --out runs/vp
   python detection/robustness/build_conditions.py --data-root $ROBOGEO_DATA --protocol cts --out runs/cts
   ```

   `cells.json` lists every cell with its info file (`pkl`) and the checkpoint platform that
   evaluates it (`model_platform`). This is the Sedan for every cell except the cross-platform
   oracles.

2. **Evaluate every cell.** Write one file per cell to `runs/<protocol>/results/<cell id, with / replaced by __>.json`:

   ```json
   {"cell": "vp/IMG/CAM_FRONT/pitch+12", "metric": "CDS", "value": <float>,
    "n_frames": <int>, "checkpoint_sha256": "<sha256 of the checkpoint>"}
   ```

   CDS (CARLA Detection Score) is the nuScenes detection score over 6 classes: car, truck, bus,
   motorcycle, bicycle and pedestrian. Ground truth is kept when its visibility level is 2 or more.
   The reference implementation is `_evaluate_single` in
   [`carla_nuscenes_dataset.py`](../detection/bevformer/projects/mmdet3d_plugin/datasets/carla_nuscenes_dataset.py).

3. **Aggregate.**

   ```bash
   python detection/robustness/aggregate.py --results runs/vp/results  --out runs/vp/summary.json
   python detection/robustness/aggregate.py --results runs/cts/results --out runs/cts/summary.json
   ```

The cell ids and the formulas are documented in [`detection/robustness`](../detection/robustness).

## BEV segmentation

The simplest way is to add the model to [`segmentation`](../segmentation) and run `bevunify.eval`,
which handles the labels, the cells and the IoU. The released models are in `bevunify/` and
`config/experiment/`. Alternatively, write the same per-cell files with `"metric": "IoU"`. IoU
covers the vehicle classes 4, 5, 6, 7, 8, 10 and 11 of the labels, with the prediction thresholded
at 0.5 and cells of visibility < 2 ignored. `detection/robustness/aggregate.py` scores these files
the same way.
