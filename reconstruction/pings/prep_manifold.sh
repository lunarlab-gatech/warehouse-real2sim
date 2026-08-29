#!/usr/bin/env bash
# Prep a Manifold scene as a PINGS dataset: bags -> KITTI sequence, inside docker.
#   bash prep_manifold.sh <SCENE_DIR> <NAME>   ->  ./data/geoscan_<NAME>/sequences/00
set -eo pipefail
SCENE="$(cd "${1:?usage: bash prep_manifold.sh <scene dir> <name>}" && pwd)"
NAME="${2:?usage: bash prep_manifold.sh <scene dir> <name>}"
REPO="$(cd "$(dirname "$0")" && pwd)"

rc=0
docker run --rm \
  -v "$REPO":/packages/pings -v "$SCENE":/bagdir:ro -w /packages/pings \
  pings:cu118 \
  bash -c "
    python3 -c 'import rosbags' 2>/dev/null || pip install --quiet rosbags
    rm -rf 'data/geoscan_${NAME}/sequences/00'
    python3 scripts/convert_manifold_bags.py /bagdir 'data/geoscan_${NAME}/sequences/00'
    echo PREP_DONE" || rc=$?

# container runs as root; hand the output back even on failure
docker run --rm -v "$REPO":/packages/pings pings:cu118 \
  chown -R "$(id -u):$(id -g)" "/packages/pings/data/geoscan_${NAME}" || true
exit $rc