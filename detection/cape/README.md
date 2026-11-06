# CAPE

[CAPE](https://github.com/kaixinbear/CAPE) (single frame, R50-DCN) for RoboGeo 3D object detection.
An unaffiliated adaptation of CAPE at commit d7f0e0ba.

## Setup

Use the [`env/cape`](../../env/cape) image and run all commands from this folder.

```bash
mkdir -p data && ln -s $ROBOGEO_DATA/carla_geobev data/nuscenes
# training only: MSRA ResNet-50 ImageNet weights (sha256 5891d200ef31febc610b915837775354fa880be47283cd0b14648a95aa13465e)
mkdir -p ckpts && curl -L -o ckpts/resnet50_msra-5891d200.pth \
    https://download.openmmlab.com/pretrain/third_party/resnet50_msra-5891d200.pth
```

| platform | config (`projects/configs/CAPE/`) | checkpoint (`$ROBOGEO_CKPT/detection/cape/`) |
|---|---|---|
| Sedan | `cape_carla_sedan.py` | `cape_sedan.pth` |
| SUV | `cape_carla_suv.py` | `cape_suv.pth` |
| Bus | `cape_carla_bus.py` | `cape_bus.pth` |

## Evaluate

```bash
# CDS on the evaluation split of the config's platform (1 GPU)
PYTHONPATH=. python tools/test.py projects/configs/CAPE/cape_carla_sedan.py \
    $ROBOGEO_CKPT/detection/cape/cape_sedan.pth --eval bbox

# viewpoint perturbation and cross-platform transfer (1 GPU)
./tools/robogeo_eval.sh vp  out/vp
./tools/robogeo_eval.sh cts out/cts
python ../robustness/aggregate.py --results out/vp/results  --out out/vp/summary.json
python ../robustness/aggregate.py --results out/cts/results --out out/cts/summary.json
```

A restarted run skips the cells that are already finished. One cell takes about 10 minutes on an
RTX 4090; the runner disables TF32, so plain `tools/test.py` can differ in the last digits. To use
N GPUs, start `CUDA_VISIBLE_DEVICES=i ./tools/robogeo_eval.sh vp out/vp --shard i/N` for each GPU i;
shard 0 builds the cells and the others wait for them.

## Train

```bash
./tools/dist_train.sh projects/configs/CAPE/cape_carla_sedan.py 2   # _suv / _bus for the other platforms
```

The configs follow the recipe of the released checkpoints:
- batch size 8 per GPU on 2 GPUs (total 16; about 78 GB per GPU), 24 epochs, seed 0;
- the upstream optimizer and schedule, with lr 4e-4, scaled linearly from 2e-4 at total batch size
  8, and fp32 with gradient clipping 35 instead of fp16;
- 6 classes, ground truth with visibility >= 2, no augmentation: every image is resized by 0.88 and
  cropped to 1408×512.

The number of query-denoising groups is `600 // (max GT boxes in the per-GPU batch)`, so it depends
on the batch size per GPU (about 20 groups at batch size 8 per GPU, about 87 upstream at batch size
1). Training with another batch size per GPU, or with gradient accumulation, is not equivalent to
the released recipe. Evaluation runs at batch size 1.

## Citation

```bibtex
@inproceedings{Xiong2023CAPE,
  title={CAPE: Camera View Position Embedding for Multi-View 3D Object Detection},
  author={Xiong, Kaixin and Gong, Shi and Ye, Xiaoqing and Tan, Xiao and Wan, Ji and Ding, Errui and Wang, Jingdong and Bai, Xiang},
  booktitle={Computer Vision and Pattern Recognition},
  year={2023}
}
```
