#!/usr/bin/env bash
# Install the RoboGeo segmentation environment into the active Python 3.10 environment
# (conda or venv) and build the GaussianLSS rasterizer from the vendored source.
#
#   bash env/segmentation/install.sh            # GaussianLSS and CVT
#   bash env/segmentation/install.sh --cvt-only # CVT only: no CUDA toolkit or rasterizer needed
#
# The rasterizer build needs the CUDA 12.8 toolkit (nvcc) matching the torch wheels:
# set CUDA_HOME if nvcc is not on PATH, and TORCH_CUDA_ARCH_LIST to the GPUs you use
# (default below: Ampere to Blackwell).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
RASTERIZER="$REPO/segmentation/third_party/GaussianLSS/GaussianLSS/model/diff-gaussian-rasterization"

python - <<'EOF'
import sys
if sys.version_info[:2] != (3, 10):
    sys.exit(f"Python 3.10 is required, found {sys.version.split()[0]}")
EOF

python -m pip install -r "$HERE/requirements.lock.txt"

if [[ "${1:-}" == "--cvt-only" ]]; then
    echo "CVT-only environment installed (GaussianLSS needs the rasterizer: rerun without --cvt-only)."
    exit 0
fi

# Build out of tree so the vendored source stays clean.
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;8.6;8.9;9.0;10.0;12.0}"
BUILD="$(mktemp -d)"
trap 'rm -rf "$BUILD"' EXIT
cp -r "$RASTERIZER" "$BUILD/src"
python -m pip install --no-build-isolation --no-deps "$BUILD/src"
python -c "import torch, diff_gaussian_rasterization; print('rasterizer OK')"
