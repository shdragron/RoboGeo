#!/usr/bin/env bash
# Install the CAPE environment into the active Python 3.10 (the Docker image, a
# conda env or a venv). Needs the CUDA 12.8 toolkit (nvcc; CUDA_HOME or nvcc on
# PATH) and gcc/g++ 9-13 to build mmcv-full.
#
#   TORCH_CUDA_ARCH_LIST="8.9" bash env/cape/install.sh
#
# TORCH_CUDA_ARCH_LIST selects the GPU architectures the mmcv CUDA ops are
# built for (default: Ampere, Ada, Hopper, Blackwell).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PYTHON:-python}
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-"8.0;8.6;8.9;9.0;10.0;12.0"}
export MAX_JOBS=${MAX_JOBS:-8} FORCE_CUDA=1 MMCV_WITH_OPS=1

MMCV_URL=https://files.pythonhosted.org/packages/source/m/mmcv-full/mmcv-full-1.7.1.tar.gz
MMCV_SHA256=83eb5e9b3adc31e63f201df4e13d4f5d68bc67fceb7b42c90e592790b1bdf228
MMDET3D_URL=https://files.pythonhosted.org/packages/source/m/mmdet3d/mmdet3d-1.0.0rc6.tar.gz
MMDET3D_SHA256=5f598ebdc05bac44df16e384a663cf9bf800b33e64bfa32b275246f1db87c779

BUILD=$(mktemp -d)
trap 'rm -rf "$BUILD"' EXIT
fetch() {  # url sha256 -> unpacked into $BUILD
    curl -fsSL -o "$BUILD/src.tar.gz" "$1"
    echo "$2  $BUILD/src.tar.gz" | sha256sum -c -
    tar xzf "$BUILD/src.tar.gz" -C "$BUILD"
    rm "$BUILD/src.tar.gz"
}

$PY -m pip install --upgrade pip setuptools==70.2.0 wheel
$PY -m pip install torch==2.11.0 torchvision==0.26.0 \
    --index-url https://download.pytorch.org/whl/cu128
$PY -m pip install -r "$HERE/requirements.txt"

# mmcv-full 1.7.1, patched for torch 2.x. The upfirdn2d and fused_bias_leakyrelu
# ops are under the NVIDIA Source Code License (non-commercial) and are not built.
fetch "$MMCV_URL" "$MMCV_SHA256"
(
    cd "$BUILD/mmcv-full-1.7.1"
    find mmcv/ops -name '*upfirdn2d*' -delete
    find mmcv/ops -name '*fused_bias*' -delete
    patch -p1 < "$HERE/patches/mmcv-full-1.7.1.patch"
    $PY -m pip install --no-deps --no-build-isolation .
)

$PY -m pip install --no-deps mmdet==2.28.2 mmsegmentation==0.30.0

# mmdet3d 1.0.0rc6 (pure Python): allow mmcv 1.7.x and replace the numpy
# aliases removed in numpy 1.24.
fetch "$MMDET3D_URL" "$MMDET3D_SHA256"
(
    cd "$BUILD/mmdet3d-1.0.0rc6"
    patch -p1 < "$HERE/patches/mmdet3d-1.0.0rc6.patch"
    $PY -m pip install --no-deps --no-build-isolation .
)

# nuscenes-devkit 1.1.10: DetectionConfig.class_names as a list (a dict_keys view
# cannot be pickled).
SITE=$($PY -c "import nuscenes, os; print(os.path.dirname(os.path.dirname(nuscenes.__file__)))")
patch -d "$SITE" -p1 < "$HERE/patches/nuscenes-devkit-1.1.10.patch"

$PY -c "import torch, mmcv, mmcv.ops, mmdet, mmseg, mmdet3d; \
print('torch', torch.__version__, '| mmcv', mmcv.__version__, '| mmdet', mmdet.__version__, \
'| mmseg', mmseg.__version__, '| mmdet3d', mmdet3d.__version__)"
