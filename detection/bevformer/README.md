# BEVFormer-tiny

[BEVFormer](https://github.com/fundamentalvision/BEVFormer)-tiny (commit 66b65f3) for RoboGeo 3D object
detection. The model code is unchanged.

## Setup

Use the [`env/bevformer`](../../env/bevformer) image and run all commands from this folder.

```bash
mkdir -p data && ln -s $ROBOGEO_DATA/carla_geobev data/nuscenes
```

| platform | config (`projects/configs/bevformer/`) | checkpoint (`$ROBOGEO_CKPT/detection/bevformer_tiny/`) |
|---|---|---|
| Sedan | `bevformer_tiny_carla.py` | `bevformer_tiny_sedan.pth` |
| SUV | `bevformer_tiny_carla_suv.py` | `bevformer_tiny_suv.pth` |
| Bus | `bevformer_tiny_carla_bus.py` | `bevformer_tiny_bus.pth` |

## Evaluate

```bash
# CDS on the evaluation split of the config's platform (N GPUs)
./tools/dist_test.sh projects/configs/bevformer/bevformer_tiny_carla.py \
    $ROBOGEO_CKPT/detection/bevformer_tiny/bevformer_tiny_sedan.pth N

# viewpoint perturbation and cross-platform transfer (1 GPU)
./tools/robogeo_eval.sh vp  out/vp
./tools/robogeo_eval.sh cts out/cts
python ../robustness/aggregate.py --results out/vp/results  --out out/vp/summary.json
python ../robustness/aggregate.py --results out/cts/results --out out/cts/summary.json
```

A restarted run skips the cells that are already finished. One cell takes about 7 minutes on an
RTX 4090. To use N GPUs, write `out/vp/cells.json` first (with `robogeo_eval.sh` or
`../robustness/build_conditions.py`), then start one runner per GPU i:

```bash
CUDA_VISIBLE_DEVICES=i python tools/robogeo_eval.py --cells out/vp/cells.json \
    --data-root $ROBOGEO_DATA --ckpt-root $ROBOGEO_CKPT --shard i/N
```

## Train

```bash
./tools/dist_train.sh projects/configs/bevformer/bevformer_tiny_carla.py 2   # _suv / _bus for the other platforms
```

The configs follow the recipe of the released checkpoints:
- batch size 8 per GPU on 2 GPUs (total 16), 24 epochs, seed 0;
- the upstream BEVFormer-tiny optimizer and schedule, with lr 4e-4, scaled linearly from 2e-4 at
  total batch size 8;
- single frame, no GridMask or photometric augmentation, 800×450 input.

With a batch size above 1 per GPU, upstream `TemporalSelfAttention` mixes the BEV queries of
different samples ([#96](https://github.com/fundamentalvision/BEVFormer/issues/96),
[#309](https://github.com/fundamentalvision/BEVFormer/issues/309)). The released checkpoints were
trained with this behaviour, so it is left unpatched. Evaluation runs at batch size 1 and is not
affected.

## Citation

```bibtex
@article{li2022bevformer,
  title={BEVFormer: Learning Bird’s-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers},
  author={Li, Zhiqi and Wang, Wenhai and Li, Hongyang and Xie, Enze and Sima, Chonghao and Lu, Tong and Qiao, Yu and Dai, Jifeng},
  journal={arXiv preprint arXiv:2203.17270},
  year={2022}
}
```
