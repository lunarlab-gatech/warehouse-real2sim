#!/usr/bin/env bash
# Prepare a GeoScan rosbag as a PINGS dataset (runs _prep_geoscan.sh inside docker).
#   bash prep.sh <BAG_PATH> <NAME>
# Produces ./data/geoscan_<NAME>/sequences/00 with LiDAR, three cameras, and
# per-frame timestamps. No poses: PINGS tracks its own trajectory at run time.
set -eo pipefail
BAG="${1:?usage: bash prep.sh <bag path> <name>}"
NAME="${2:?usage: bash prep.sh <bag path> <name>}"
REPO="$(cd "$(dirname "$0")" && pwd)"

docker run --rm \
  -v "$REPO":/packages/pings \
  -v "$(dirname "$BAG")":/bagdir:ro \
  -w /packages/pings \
  pings:cu118 \
  bash _prep_geoscan.sh "/bagdir/$(basename "$BAG")" "./data/geoscan_${NAME}"

# the container runs as root; hand the output back to the invoking user
docker run --rm -v "$REPO":/packages/pings pings:cu118 \
  chown -R "$(id -u):$(id -g)" "/packages/pings/data/geoscan_${NAME}"
