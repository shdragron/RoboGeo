# RoboGeo BEV segmentation (GaussianLSS, CVT). Build from the repository root:
#
#   docker build -f env/segmentation/Dockerfile -t robogeo-seg .
#   docker build -f env/segmentation/Dockerfile --build-arg CVT_ONLY=1 -t robogeo-seg:cvt .
#
# Run as described in env/segmentation/README.md and docs/install.md.
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ARG CVT_ONLY=0
ARG TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;10.0;12.0"

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3.10 python3.10-dev python3-pip git \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/RoboGeo
COPY env/segmentation env/segmentation
COPY segmentation segmentation
RUN TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" bash env/segmentation/install.sh $([ "$CVT_ONLY" = 1 ] && echo --cvt-only)

WORKDIR /workspace/RoboGeo/segmentation
