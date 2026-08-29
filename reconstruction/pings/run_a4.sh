#!/usr/bin/env bash
# Run PINGS with the A4 config (Oxford recipe + deskew + step_frame 1, pose-fixed
# on the dataset's poses.txt). IMPORTANT: no --no-deskew here — the CLI flag
# unconditionally overrides the yaml (pings.py:108), and A4 needs deskew ON.
#   bash run_a4.sh <GPU_INDEX> <DATA_NAME> <TAG>
#   e.g. bash run_a4.sh 1 warehouse_liosam warehouse_A4
#        bash run_a4.sh 2 warehouse_fl2    warehouse_A4fl2
set -eo pipefail
GPU="${1:?usage: bash run_a4.sh <gpu index> <data name> <tag>}"
NAME="${2:?usage: bash run_a4.sh <gpu index> <data name> <tag>}"
TAG="${3:?usage: bash run_a4.sh <gpu index> <data name> <tag>}"
REPO="$(cd "$(dirname "$0")" && pwd)"
CACHE="${PINGS_TORCH_CACHE:-$HOME/.cache/torch}"
mkdir -p "$CACHE"

docker run --rm --gpus "device=${GPU}" \
  -v "$REPO":/packages/pings \
  -v "$CACHE":/root/.cache/torch \
  -w /packages/pings \
  pings:cu118 \
  bash -c "git config --global --add safe.directory /packages/pings 2>/dev/null;
           python3 pings.py ./config/run_geoscan_gs_A4.yaml geoscan 00 \
             -i './data/geoscan_${NAME}/' -s -m --tag '${TAG}'"

# the container runs as root; hand the results back to the invoking user
docker run --rm -v "$REPO":/packages/pings pings:cu118 \
  chown -R "$(id -u):$(id -g)" /packages/pings/pings_experiments