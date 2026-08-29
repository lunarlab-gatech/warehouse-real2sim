#!/usr/bin/env bash
# Host launcher (lunarL40S): run FL2 LIO over one MANIFOLD scene (all bag parts)
# to produce the trajectory. Mirrors _fl2_box.sh; CPU-only, sequential (the FL2
# Log/ dir is shared — never run two scenes at once).
#   bash _fl2_manifold_box.sh <SceneDirName> [rate=0.5]   e.g. Warehouse
trap 'echo "FL2_ORCH_EXIT_$?"' EXIT
set -e
ROOT="${ROOT:-$HOME}"
FL2_REPO="$ROOT/projects/Fast-Livo2-GeoScan"
[ -d "$FL2_REPO" ] || { echo "ERROR: FL2 repo missing at $FL2_REPO"; exit 2; }
SCENE="${1:?need scene dir name under \$ROOT/data/rosbags/manifold (e.g. Warehouse)}"
RATE="${2:-0.5}"
echo "FL2-manifold ROOT=$ROOT scene=$SCENE rate=$RATE"

docker rm -f ali497_fl2 2>/dev/null || true
docker run --rm --name ali497_fl2 --cpus=16 --memory=32g --shm-size=16g \
  -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" \
  -v "$FL2_REPO:/root/catkin_ws/src/FAST-LIVO2" \
  -v "$ROOT/data/rosbags:/data" \
  lunarlab/fastlivo2:base \
  bash -lc "set -eo pipefail
    trap 'chown -R \$HOST_UID:\$HOST_GID /root/catkin_ws/src/FAST-LIVO2 2>/dev/null || true' EXIT
    bash /root/catkin_ws/src/FAST-LIVO2/_fl2_manifold_incontainer.sh /data/manifold/$SCENE $RATE"
