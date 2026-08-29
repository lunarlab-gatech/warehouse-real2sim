#!/usr/bin/env bash
# Run PINGS on a prepared dataset with the HIGHER-QUALITY config
# (config/run_geoscan_gs_hq.yaml). Like run.sh, but the run tag can differ
# from the dataset name so an HQ run coexists with the stock run's output.
#   bash run_hq.sh <GPU_INDEX> <DATA_NAME> [TAG]
# DATA_NAME picks ./data/geoscan_<DATA_NAME>; TAG defaults to <DATA_NAME>_hq.
set -eo pipefail
GPU="${1:?usage: bash run_hq.sh <gpu index> <data name> [tag]}"
NAME="${2:?usage: bash run_hq.sh <gpu index> <data name> [tag]}"
TAG="${3:-${NAME}_hq}"
REPO="$(cd "$(dirname "$0")" && pwd)"
CACHE="${PINGS_TORCH_CACHE:-$HOME/.cache/torch}"
mkdir -p "$CACHE"

docker run --rm --gpus "device=${GPU}" \
  -v "$REPO":/packages/pings \
  -v "$CACHE":/root/.cache/torch \
  -w /packages/pings \
  pings:cu118 \
  bash -c "git config --global --add safe.directory /packages/pings 2>/dev/null;
           python3 pings.py ./config/run_geoscan_gs_hq.yaml geoscan 00 \
             -i './data/geoscan_${NAME}/' -s -m --tag '${TAG}'"

# the container runs as root; hand the results back to the invoking user
docker run --rm -v "$REPO":/packages/pings pings:cu118 \
  chown -R "$(id -u):$(id -g)" /packages/pings/pings_experiments