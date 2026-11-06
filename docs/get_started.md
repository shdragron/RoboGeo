# Evaluation and training

Run the commands inside the model's container ([Installation](install.md)) with
`ROBOGEO_DATA` and `ROBOGEO_CKPT` set ([Data and checkpoints](data.md)). Options and recipe notes are
in each model's README:
[BEVFormer](../detection/bevformer) · [CAPE](../detection/cape) ·
[GaussianLSS and CVT](../segmentation).

## 1. Evaluate a checkpoint

These commands evaluate a checkpoint on the evaluation split of its own platform (3,792 frames).
On the Sedan this gives P<sub>NORMAL</sub>; on the SUV and the Bus it gives P<sub>oracle</sub>.

```bash
# BEVFormer, from detection/bevformer (N GPUs)
./tools/dist_test.sh projects/configs/bevformer/bevformer_tiny_carla.py \
    $ROBOGEO_CKPT/detection/bevformer_tiny/bevformer_tiny_sedan.pth N

# CAPE, from detection/cape (1 GPU)
PYTHONPATH=. python tools/test.py projects/configs/CAPE/cape_carla_sedan.py \
    $ROBOGEO_CKPT/detection/cape/cape_sedan.pth --eval bbox

# GaussianLSS, from segmentation (CVT: replace gaussianlss by cvt)
python -m bevunify.eval +experiment=gaussianlss data.platform=sedan \
    ckpt=$ROBOGEO_CKPT/segmentation/gaussianlss/gaussianlss_sedan.pth
```

For the SUV and the Bus:
- detection: use the `_suv` or `_bus` config with the matching checkpoint;
- segmentation: set `data.platform=suv|bus`.

## 2. Run the robustness protocols

**Viewpoint perturbation** evaluates the Sedan model on 631 cells: NORMAL plus
3 conditions × 7 modes × 30 rotations.
**Cross-platform transfer** evaluates it on the SUV and Bus layouts, under IMG and CAL, together
with the oracle of each platform.

Detection, from `detection/bevformer` or `detection/cape`:

```bash
./tools/robogeo_eval.sh vp  out/vp     # viewpoint perturbation
./tools/robogeo_eval.sh cts out/cts    # cross-platform transfer
python ../robustness/aggregate.py --results out/vp/results  --out out/vp/summary.json
python ../robustness/aggregate.py --results out/cts/results --out out/cts/summary.json
```

Segmentation, from `segmentation`:

```bash
CK=$ROBOGEO_CKPT/segmentation/gaussianlss
python -m bevunify.eval +experiment=gaussianlss ckpt=$CK/gaussianlss_sedan.pth eval.protocol=vp
python -m bevunify.eval +experiment=gaussianlss ckpt=$CK/gaussianlss_sedan.pth eval.protocol=cts \
    eval.target=suv eval.oracle_ckpt=$CK/gaussianlss_suv.pth      # and the same with bus
python -m bevunify.aggregate --results eval_results/gaussianlss_full/results
```

The summary reports P<sub>NORMAL</sub>, mVRS for EXT, IMG and CAL, and CTS for IMG and CAL on each
platform, which are the columns of the results table. Finished cells are skipped, so an
interrupted run resumes with the same command. `--subset fps16` (`eval.subset=fps16` for
segmentation) evaluates 768 of the 3,792 frames. Use it for quick checks only.

## 3. Train

The default configs reproduce the recipes of the released checkpoints:

```bash
# BEVFormer, from detection/bevformer: 2 GPUs, batch size 8 per GPU
./tools/dist_train.sh projects/configs/bevformer/bevformer_tiny_carla.py 2

# CAPE, from detection/cape: 2 GPUs, batch size 8 per GPU (download the backbone init first, see its README)
./tools/dist_train.sh projects/configs/CAPE/cape_carla_sedan.py 2

# GaussianLSS / CVT, from segmentation: 2 GPUs, batch size 16 per GPU
python -m bevunify.train +experiment=gaussianlss data.platform=sedan experiment.uuid=gaussianlss_sedan
```

To train the SUV and Bus models, use the `_suv` or `_bus` config, or set `data.platform=suv|bus`.
