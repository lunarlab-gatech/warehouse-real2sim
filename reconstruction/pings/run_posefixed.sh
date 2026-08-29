#!/usr/bin/env bash
# Run PINGS in POSE-FIXED mode (config/run_geoscan_gs_posefixed.yaml -- no
# tracker: section, so the mapping follows the poses.txt in the dataset, e.g.
# an FL2 trajectory). Mirrors run.sh/run_hq.sh.
#   bash run_posefixed.sh <GPU_INDEX> <DATA_NAME> [TAG]
# DATA_NAME picks ./data/geoscan_<DATA_NAME> (must contain sequences/00/poses.txt);
# TAG defaults to DATA_NAME.
set -eo pipefail
GPU="${1:?usage: bash run_posefixed.sh <gpu index> <data name> [tag]}"
NAME="${2:?usage: bash run_posefixed.sh <gpu index> <data name> [tag]}"
TAG="${3:-${NAME}}"
REPO="$(cd "$(dirname "$0")" && pwd)"
CACHE="${PINGS_TORCH_CACHE:-$HOME/.cache/torch}"
mkdir -p "$CACHE"

docker run --rm --gpus "device=${GPU}" \
  -v "$REPO":/packages/pings \
  -v "$CACHE":/root/.cache/torch \
  -w /packages/pings \
  pings:cu118 \
  bash -c "git config --global --add safe.directory /packages/pings 2>/dev/null;
           python3 pings.py ./config/run_geoscan_gs_posefixed.yaml geoscan 00 \
             -i './data/geoscan_${NAME}/' -s -m --no-deskew --tag '${TAG}'"

# the container runs as root; hand the results back to the invoking user
docker run --rm -v "$REPO":/packages/pings pings:cu118 \
  chown -R "$(id -u):$(id -g)" /packages/pings/pings_experiments