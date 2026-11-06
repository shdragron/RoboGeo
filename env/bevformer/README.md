# Environment for BEVFormer-tiny

The Docker image reproduces the software stack that trained and evaluated the released
BEVFormer-tiny checkpoints:

| package | version |
|---|---|
| CUDA / cuDNN (base image) | 12.8.1 / 9 (`nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04`) |
| Python | 3.10 |
| torch / torchvision | 2.11.0+cu128 / 0.26.0+cu128 |
| mmcv-full | 1.7.1, built from source with the patches below |
| mmdet / mmsegmentation | 2.14.0 / 0.14.1 |
| mmdet3d | 0.17.1, built from source with the patches below |
| nuscenes-devkit | 1.1.10 |
| numpy | 1.26.4 |

`requirements.txt` lists the direct dependencies; `requirements.lock` pins every package of
the original environment and is used as a pip constraints file.

## Build and run

```bash
docker build -t robogeo-bevformer env/bevformer        # from the repository root
docker run --gpus all --shm-size=16g -it \
    -v $PWD:/workspace/RoboGeo2027 -w /workspace/RoboGeo2027 \
    -v $ROBOGEO_DATA:$ROBOGEO_DATA:ro -e ROBOGEO_DATA \
    -v $ROBOGEO_CKPT:$ROBOGEO_CKPT:ro -e ROBOGEO_CKPT \
    robogeo-bevformer
```
The data are mounted at the same path as on the host, so the `data/nuscenes` link stays valid.
The CUDA ops are compiled for compute capabilities 8.0, 8.6, 8.9, 9.0, 10.0 and 12.0, plus PTX
(override with `--build-arg TORCH_CUDA_ARCH_LIST=...`; `--build-arg MAX_JOBS=...` sets the
compile parallelism). Building takes 30 to 60 minutes, mostly downloading the CUDA wheels and
compiling mmcv and mmdet3d (about 10 minutes with 24 compile jobs).
Inside the container, follow [`detection/bevformer/README.md`](../../detection/bevformer/README.md).

## Patches

| file | applies to | changes |
|---|---|---|
| `patches/mmcv-full-1.7.1.patch` | mmcv-full 1.7.1 sdist | C++17 for the torch 2.x headers; `THC/THCAtomics.cuh` -> `ATen/cuda/Atomic.cuh`; `_get_stream` gets a `torch.device`; DDP forward without the removed replicated-tensor module; removes the `upfirdn2d` and `fused_bias_leakyrelu` bindings |
| `patches/mmdetection3d-0.17.1.patch` | mmdetection3d v0.17.1 | THC removal in the point-cloud ops; C++17 for spconv; spconv kernels dispatched on the actual kernel volume (the fixed 4096 needs more local memory than a 24 GB GPU can reserve); spconv layers registered with `force=True` (mmcv 1.7.1 registers the same names); `numba.core.errors`; numpy >= 1.24 aliases; accepts mmcv 1.7.1 |
| `patches/site-packages.patch` | installed mmdet, mmseg, nuscenes-devkit | mmdet and mmseg accept mmcv 1.7.1; `DetectionConfig.class_names` is a list instead of a `dict_keys` view |

`upfirdn2d` and `fused_bias_leakyrelu` are distributed by mmcv under the NVIDIA Source Code
License (non-commercial). The Dockerfile deletes their sources before building mmcv, and the
patch removes their bindings and Python imports; BEVFormer uses neither. The patches contain
no code from those files.

## Without Docker

On a machine with CUDA 12.8 and gcc <= 13, the same steps work in a Python 3.10 virtual
environment: install `requirements.txt` with `-c requirements.lock`, then build mmcv-full and
mmdet3d and apply the site-packages patch exactly as the `RUN` steps of the Dockerfile do.
