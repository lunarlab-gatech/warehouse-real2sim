#!/usr/bin/env bash
# Runs on the lunarL40S server (in tmux). One-shot: build FAST-LIVO2 in the lunarlab base
# image and run LIO on the harrison bag to produce the trajectory. The FL2 repo
# is mounted at the workspace src path so Log/pcd/lidar_poses.txt lands on the host.
trap 'echo "FL2_ORCH_EXIT_$?"' EXIT
set -e
ROOT="${ROOT:-$HOME}"
FL2_REPO="$ROOT/projects/Fast-Livo2-GeoScan"
[ -d "$FL2_REPO" ] || { echo "ERROR: FL2 repo missing at $FL2_REPO"; exit 2; }
BAG_BASENAME="${1:?need bag basename in \$ROOT/data/rosbags (e.g. street_walkthrough.bag)}"
RATE="${2:-0.5}"                                        # arg2: rosbag play rate
echo "FL2 ROOT=$ROOT bag=$BAG_BASENAME rate=$RATE"

# FL2 LIO is CPU-only (no --gpus). Etiquette: cap cpu/mem, name container ali497_*, --rm.
docker rm -f ali497_fl2 2>/dev/null || true
# HOST_UID/GID + the EXIT trap below hand FL2's bind-mounted outputs back to you on
# exit (FL2 builds/runs as root in-container), so nothing under /scratch/ali497 is
# left root-owned -- same chown-back pattern as _pings_box.sh.
docker run --rm --name ali497_fl2 --cpus=16 --memory=32g --shm-size=16g \
  -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" \
  -v "$FL2_REPO:/root/catkin_ws/src/FAST-LIVO2" \
  -v "$ROOT/data/rosbags:/data" \
  lunarlab/fastlivo2:base \
  bash -lc "set -eo pipefail
    trap 'chown -R \$HOST_UID:\$HOST_GID /root/catkin_ws/src/FAST-LIVO2 2>/dev/null || true' EXIT
    bash /root/catkin_ws/src/FAST-LIVO2/_fl2_lio_incontainer.sh /data/$BAG_BASENAME $RATE"
