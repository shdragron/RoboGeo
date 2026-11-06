# Installation

Each model family has its own environment, because BEVFormer and CAPE need different
OpenMMLab versions.

| models | environment | main packages |
|---|---|---|
| BEVFormer | [`env/bevformer`](../env/bevformer) | PyTorch 2.11 (CUDA 12.8), mmcv-full 1.7.1, mmdet 2.14.0, mmdet3d 0.17.1 |
| CAPE | [`env/cape`](../env/cape) | PyTorch 2.11 (CUDA 12.8), mmcv-full 1.7.1, mmdet 2.28.2, mmdet3d 1.0.0rc6 |
| GaussianLSS, CVT | [`env/segmentation`](../env/segmentation) | PyTorch 2.11 (CUDA 12.8), PyTorch Lightning 2.1.0, Hydra 1.3.2 |

All environments use Python 3.10.

## Docker (recommended)

```bash
git clone https://github.com/shdragron/RoboGeo.git
cd RoboGeo

docker build -t robogeo-bevformer env/bevformer
docker build -t robogeo-cape env/cape
docker build -t robogeo-seg -f env/segmentation/Dockerfile .
```

The images compile the CUDA operators for Ampere to Blackwell GPUs. To build faster, compile for
your GPU only, e.g. `--build-arg TORCH_CUDA_ARCH_LIST="8.9"` for an RTX 4090.

Start a container with the repository, the data and the checkpoints mounted
(`ROBOGEO_DATA` and `ROBOGEO_CKPT` are set in [Data and checkpoints](data.md)):

```bash
docker run --gpus all --shm-size=16g -it \
    -v $PWD:/workspace/RoboGeo -w /workspace/RoboGeo \
    -v $ROBOGEO_DATA:$ROBOGEO_DATA:ro -e ROBOGEO_DATA \
    -v $ROBOGEO_CKPT:$ROBOGEO_CKPT:ro -e ROBOGEO_CKPT \
    robogeo-bevformer    # or robogeo-cape, robogeo-seg
```

The data are mounted at the same path as on the host, so the `data/nuscenes` links of the
detection models stay valid inside the container.

## Without Docker

Each folder in [`env/`](../env) pins the packages and explains in its README how to install them
in a fresh Python 3.10 environment with the CUDA 12.8 toolkit.

> [!NOTE]
> GaussianLSS needs `diff-gaussian-rasterization`, which is licensed for non-commercial research
> and evaluation only. `bash env/segmentation/install.sh --cvt-only` installs CVT without it.
