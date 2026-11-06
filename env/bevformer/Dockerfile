# Environment for the RoboGeo BEVFormer-tiny code: Python 3.10, torch 2.11.0+cu128,
# mmcv-full 1.7.1 (patched), mmdet 2.14.0, mmseg 0.14.1, mmdet3d 0.17.1 (patched),
# nuscenes-devkit 1.1.10, numpy 1.26.4.
#
# Build (from the repository root):
#   docker build -t robogeo-bevformer env/bevformer
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

# GPU architectures of the compiled ops (Ampere, Ada, Hopper, Blackwell + PTX)
ARG TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;10.0;12.0+PTX"
ARG MAX_JOBS=16

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Ubuntu 22.04 ships gcc 11 (CUDA 12.8 supports up to gcc 14) and Python 3.10.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-dev python3.10-venv \
        build-essential ninja-build git curl ca-certificates patch \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

RUN python3.10 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

COPY requirements.txt requirements.lock /opt/robogeo/env/
COPY patches /opt/robogeo/env/patches/
RUN pip install pip==26.1.1 setuptools==70.2.0 wheel==0.47.0 \
    && pip install -r /opt/robogeo/env/requirements.txt -c /opt/robogeo/env/requirements.lock

ENV CUDA_HOME=/usr/local/cuda \
    FORCE_CUDA=1 \
    MMCV_WITH_OPS=1 \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    MAX_JOBS=${MAX_JOBS}

WORKDIR /opt/build

# mmcv-full 1.7.1 with CUDA ops, patched for torch 2.11 (patches/mmcv-full-1.7.1.patch).
# upfirdn2d and fused_bias_leakyrelu are under the NVIDIA Source Code License
# (non-commercial): their sources are deleted before the build and the patch
# removes their bindings. BEVFormer does not use them.
RUN curl -fsSLO https://files.pythonhosted.org/packages/12/a3/9e4f4f6e44aa234498d995d98ee60b7f5dbab3148d7b4fbd4097396c928f/mmcv-full-1.7.1.tar.gz \
    && echo "83eb5e9b3adc31e63f201df4e13d4f5d68bc67fceb7b42c90e592790b1bdf228  mmcv-full-1.7.1.tar.gz" | sha256sum -c - \
    && tar xzf mmcv-full-1.7.1.tar.gz \
    && cd mmcv-full-1.7.1 \
    && find mmcv/ops -name '*upfirdn2d*' -delete \
    && find mmcv/ops -name '*fused_bias*' -delete \
    && patch -p1 < /opt/robogeo/env/patches/mmcv-full-1.7.1.patch \
    && pip install --no-build-isolation --no-deps . \
    && cd .. && rm -rf mmcv-full-1.7.1 mmcv-full-1.7.1.tar.gz

# mmdetection3d 0.17.1 ported to torch 2.11 (patches/mmdetection3d-0.17.1.patch)
RUN git clone --depth 1 --branch v0.17.1 https://github.com/open-mmlab/mmdetection3d.git \
    && cd mmdetection3d \
    && git apply /opt/robogeo/env/patches/mmdetection3d-0.17.1.patch \
    && pip install --no-build-isolation --no-deps . \
    && cd .. && rm -rf mmdetection3d

# mmdet / mmseg: accept mmcv-full 1.7.1; nuscenes-devkit: picklable DetectionConfig
RUN patch -d /opt/venv/lib/python3.10/site-packages -p1 < /opt/robogeo/env/patches/site-packages.patch

# import check: compiled ops load, the excluded ops are absent
RUN python -c "import torch, mmcv, mmdet, mmseg, mmdet3d, nuscenes; \
import mmcv.ops; from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttnFunction; \
from mmcv.utils import ext_loader; ext_loader.load_ext('_ext', ['ms_deform_attn_forward', 'nms']); \
import mmdet3d.ops; from mmdet3d.ops.iou3d import iou3d_cuda; from mmdet3d.ops.spconv import sparse_conv_ext; \
assert not hasattr(mmcv.ops, 'upfirdn2d') and not hasattr(mmcv.ops, 'fused_bias_leakyrelu'); \
print('torch', torch.__version__, 'mmcv', mmcv.__version__, 'mmdet', mmdet.__version__, \
      'mmseg', mmseg.__version__, 'mmdet3d', mmdet3d.__version__)"

# torch calls getpass.getuser(), which needs USER when the container runs with
# --user <uid> that has no passwd entry
ENV USER=robogeo

WORKDIR /workspace
