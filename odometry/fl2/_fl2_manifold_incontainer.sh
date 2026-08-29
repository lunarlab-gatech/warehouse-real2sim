#!/usr/bin/env bash
# Runs INSIDE lunarlab/fastlivo2:base. Like _fl2_lio_incontainer.sh, but for a
# MANIFOLD scene: swaps in config/mid360_manifold.yaml (LIO-only is baked in)
# and plays ALL of the scene's sequential bag parts in one rosbag session.
# Produces Log/pcd/lidar_poses.txt (world<-LiDAR, "ts tx ty tz qx qy qz qw").
#   bash _fl2_manifold_incontainer.sh <BAG_DIR> [RATE]
set -eo pipefail
source /opt/ros/noetic/setup.bash
FL2=/root/catkin_ws/src/FAST-LIVO2
BAGDIR="${1:?need scene dir with BAG_*.bag, e.g. /data/manifold/Warehouse}"
RATE="${2:-0.5}"
BAGS=("$BAGDIR"/BAG_*.bag)
[ -f "${BAGS[0]}" ] || { echo "ERROR: no BAG_*.bag in $BAGDIR" >&2; exit 2; }
echo "scene: $BAGDIR (${#BAGS[@]} bag parts, rate $RATE)"

echo "=== configure (Manifold LIO-only config) ==="
cd "$FL2"
cp -f config/mid360.yaml /tmp/mid360.yaml.orig
cp -f config/mid360_manifold.yaml config/mid360.yaml
mkdir -p Log/pcd; rm -f Log/pcd/lidar_poses.txt; rm -f Log/pcd/*.pcd
grep -E 'img_en|pcd_save_en|extrinsic_T' config/mid360.yaml

echo "=== build (catkin_make) ==="
cd /root/catkin_ws
JOBS="${JOBS:-8}"
nice -n 10 catkin_make -DCMAKE_BUILD_TYPE=Release -j"$JOBS" > /tmp/build.log 2>&1 || { echo "BUILD FAILED:"; tail -40 /tmp/build.log; exit 1; }
source devel/setup.bash
echo "build OK"
cd "$FL2"

echo "=== launch FL2 (headless) + play ${#BAGS[@]} bags at rate ${RATE} ==="
roslaunch fast_livo mapping_mid360.launch rviz:=false > /tmp/fl2.log 2>&1 &
sleep 12
rosbag play --rate "$RATE" "${BAGS[@]}" 2>&1 | tail -4 || true
echo "=== drain (waiting for FL2 to clear its backlog) ==="
# FL2 buffers incoming scans and processes at its own pace, so playback
# ending does NOT mean processing ended. Wait until neither the pose file
# nor the PCD count grows for 60s (or a 40 min cap) before killing.
POSEF=Log/pcd/lidar_poses.txt
last=-1; stable=0; waited=0
while [ $stable -lt 6 ] && [ $waited -lt 2400 ]; do
  sleep 10; waited=$((waited+10))
  lines=0; if [ -f "$POSEF" ]; then lines=$(wc -l < "$POSEF"); fi
  npcd=$(ls Log/pcd/*.pcd 2>/dev/null | wc -l)
  cur=$((lines + npcd))
  if [ "$cur" -eq "$last" ]; then
    stable=$((stable+1))
  else
    stable=0; echo "  draining: $lines poses / $npcd pcds (${waited}s)"
  fi
  last=$cur
done
[ $waited -ge 2400 ] && echo "  WARNING: hit 40 min drain cap, stopping anyway"
echo "  drain complete after ${waited}s"

echo "=== stop ==="
rosnode kill -a >/dev/null 2>&1 || true
pkill -f fastlivo_mapping 2>/dev/null || true
pkill -f roslaunch 2>/dev/null || true
sleep 2; pkill -f rosmaster 2>/dev/null || true

echo "=== restore config ==="
cp -f /tmp/mid360.yaml.orig config/mid360.yaml

echo "=== result ==="
if [ -s Log/pcd/lidar_poses.txt ]; then
  echo "lidar_poses.txt lines: $(wc -l < Log/pcd/lidar_poses.txt)"
  echo "first/last:"; head -1 Log/pcd/lidar_poses.txt; tail -1 Log/pcd/lidar_poses.txt
else
  echo "NO lidar_poses.txt produced — FL2 log tail:"; tail -30 /tmp/fl2.log
  exit 3
fi
echo "FL2_DONE"
