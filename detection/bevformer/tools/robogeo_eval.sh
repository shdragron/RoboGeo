#!/usr/bin/env bash
# Added by the RoboGeo authors, 2026.
# Build the cells of one RoboGeo protocol and evaluate them with the released
# BEVFormer-tiny checkpoints on one GPU.
#
#   ROBOGEO_DATA=... ROBOGEO_CKPT=... tools/robogeo_eval.sh <vp|cts> <out dir> \
#       [--cells ...] [--subset full|fps16]
#
# Writes <out dir>/cells.json, <out dir>/pkls/ and <out dir>/results/. Score the
# results with detection/robustness/aggregate.py.
set -e
: "${ROBOGEO_DATA:?set ROBOGEO_DATA}" "${ROBOGEO_CKPT:?set ROBOGEO_CKPT}"
PROTOCOL=$1
mkdir -p "$2"
OUT=$(cd "$2" && pwd)
shift 2

HERE=$(cd "$(dirname "$0")/.." && pwd)
ROBUSTNESS=$HERE/../robustness

python "$ROBUSTNESS/build_conditions.py" --data-root "$ROBOGEO_DATA" \
    --protocol "$PROTOCOL" --out "$OUT" "$@"
cd "$HERE"
python tools/robogeo_eval.py --cells "$OUT/cells.json" \
    --data-root "$ROBOGEO_DATA" --ckpt-root "$ROBOGEO_CKPT"
