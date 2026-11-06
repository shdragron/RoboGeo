# Segmentation environment

Python 3.10, PyTorch 2.11 (CUDA 12.8), PyTorch Lightning 2.1.0 and Hydra 1.3.2, for
[segmentation](../../segmentation). All packages are pinned in
[requirements.lock.txt](requirements.lock.txt).

## Docker

```bash
docker build -t robogeo-seg -f env/segmentation/Dockerfile .   # from the repository root
docker run --gpus all --shm-size=16g -it \
    -v $PWD:/workspace/RoboGeo -w /workspace/RoboGeo \
    -v $ROBOGEO_DATA:$ROBOGEO_DATA:ro -e ROBOGEO_DATA \
    -v $ROBOGEO_CKPT:$ROBOGEO_CKPT:ro -e ROBOGEO_CKPT \
    robogeo-seg
```

`--build-arg CVT_ONLY=1` builds an image without the Gaussian rasterizer, which only CVT can use.

## Without Docker

```bash
bash env/segmentation/install.sh              # GaussianLSS and CVT; builds the rasterizer (needs nvcc 12.8)
bash env/segmentation/install.sh --cvt-only   # CVT only
```
