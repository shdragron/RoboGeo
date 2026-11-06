# CAPE environment

Python 3.10, PyTorch 2.11.0 / torchvision 0.26.0 (CUDA 12.8), mmcv-full 1.7.1, mmdet 2.28.2,
mmsegmentation 0.30.0, mmdet3d 1.0.0rc6 and nuscenes-devkit 1.1.10, for
[detection/cape](../../detection/cape). Other packages are pinned in
[requirements.txt](requirements.txt).

## Docker

```bash
docker build -t robogeo-cape env/cape    # --build-arg TORCH_CUDA_ARCH_LIST="8.9": one GPU architecture only
docker run --gpus all --shm-size=16g -it \
    -v $PWD:/workspace/RoboGeo2027 -w /workspace/RoboGeo2027 \
    -v $ROBOGEO_DATA:$ROBOGEO_DATA:ro -e ROBOGEO_DATA \
    -v $ROBOGEO_CKPT:$ROBOGEO_CKPT:ro -e ROBOGEO_CKPT \
    robogeo-cape
```

## Without Docker

In a fresh Python 3.10 environment with the CUDA 12.8 toolkit (nvcc) and gcc 9-13:
```bash
TORCH_CUDA_ARCH_LIST="8.9" bash env/cape/install.sh   # the architecture(s) of your GPU(s)
```

## Patches

- `mmcv-full-1.7.1.patch` (as in `env/bevformer`): PyTorch 2.x build and API fixes. The
  `upfirdn2d` and `fused_bias_leakyrelu` ops are under the NVIDIA Source Code License
  (non-commercial); `install.sh` deletes their sources and the patch removes their bindings. CAPE
  does not use them.
- `mmdet3d-1.0.0rc6.patch`: accepts mmcv 1.7.x; replaces the numpy aliases removed in numpy 1.24.
- `nuscenes-devkit-1.1.10.patch`: `DetectionConfig.class_names` as a picklable list.

The released checkpoints were trained with PyTorch 2.7.1, mmcv-full 1.7.2 and Python 3.11;
evaluating them in this environment gives the same results up to floating-point differences.
