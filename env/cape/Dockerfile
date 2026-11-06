# CAPE environment for RoboGeo: Ubuntu 22.04, Python 3.10, CUDA 12.8,
# PyTorch 2.11.0, mmcv-full 1.7.1 (patched, built from source), mmdet 2.28.2,
# mmsegmentation 0.30.0, mmdet3d 1.0.0rc6 (patched).
#
#   docker build -t robogeo-cape env/cape
#   docker build -t robogeo-cape --build-arg TORCH_CUDA_ARCH_LIST="8.9" env/cape
FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

ARG TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;10.0;12.0"
ARG MAX_JOBS=8

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        python3 python3-dev python3-pip python-is-python3 \
        build-essential ninja-build git curl ca-certificates patch \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt install.sh /opt/robogeo/env/
COPY patches /opt/robogeo/env/patches
RUN TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" MAX_JOBS="$MAX_JOBS" \
    bash /opt/robogeo/env/install.sh && rm -rf /root/.cache/pip

WORKDIR /workspace
