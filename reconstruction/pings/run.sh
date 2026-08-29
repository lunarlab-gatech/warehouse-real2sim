#!/usr/bin/env bash
# Run PINGS on a prepared GeoScan dataset (upstream workflow, one run at a time).
#   bash run.sh <GPU_INDEX> <NAME>
# Uses config/run_geoscan_gs.yaml, whose tracker: section makes PINGS estimate
# the trajectory itself (no external poses needed). Saves map + mesh.
set -eo pipefail
GPU="${1:?usage: bash run.sh <gpu index> <name>}"
NAME="${2:?usage: bash run.sh <gpu index> <name>}"
REPO="$(cd "$(dirname "$0")" && pwd)"
# torch hub cache (LPIPS/VGG weights for offline eval etc.); override with
# PINGS_TORCH_CACHE if $HOME is on root-squashed NFS where docker cannot mkdir
CACHE="${PINGS_TORCH_CACHE:-$HOME/.cache/torch}"
mkdir -p "$CACHE"

docker run --rm --gpus "device=${GPU}" \
  -v "$REPO":/packages/pings \
  -v "$CACHE":/root/.cache/torch \
  -w /packages/pings \
  pings:cu118 \
  bash -c "git config --global --add safe.directory /packages/pings 2>/dev/null;
           python3 pings.py ./config/run_geoscan_gs.yaml geoscan 00 \
             -i './data/geoscan_${NAME}/' -s -m --tag '${NAME}'"

# the container runs as root; hand the results back to the invoking user
docker run --rm -v "$REPO":/packages/pings pings:cu118 \
  chown -R "$(id -u):$(id -g)" /packages/pings/pings_experiments
