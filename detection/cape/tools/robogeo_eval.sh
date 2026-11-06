#!/usr/bin/env bash
# Copyright (c) 2026 The RoboGeo authors.
# Build the cells of one RoboGeo protocol and evaluate them with the released
# CAPE checkpoints (single GPU per process).
#
#   ROBOGEO_DATA=... ROBOGEO_CKPT=... tools/robogeo_eval.sh <vp|cts> <out dir> \
#       [--cells ...] [--subset full|fps16] [--shard K/N]
#
# Writes <out dir>/cells.json, <out dir>/pkls/ and <out dir>/results/; rerun the
# same command to resume. Score the results with detection/robustness/aggregate.py.
set -e
: "${ROBOGEO_DATA:?set ROBOGEO_DATA}" "${ROBOGEO_CKPT:?set ROBOGEO_CKPT}"
[ $# -ge 2 ] || { echo "usage: $0 <vp|cts> <out dir> [options]" >&2; exit 2; }
PROTOCOL=$1
mkdir -p "$2"
OUT=$(cd "$2" && pwd)
shift 2
export ROBOGEO_DATA=$(cd "$ROBOGEO_DATA" && pwd) ROBOGEO_CKPT=$(cd "$ROBOGEO_CKPT" && pwd)
cd "$(dirname "$0")/.."
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" exec python tools/robogeo_eval.py "$PROTOCOL" "$OUT" "$@"
