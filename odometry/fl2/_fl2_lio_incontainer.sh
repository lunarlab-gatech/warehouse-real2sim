#!/usr/bin/env bash
# Runs INSIDE lunarlab/fastlivo2:base. Builds FAST-LIVO2, runs it in LIO-only mode
# (LiDAR + IMU, no camera) on the given Livox bag, and produces a per-scan
# trajectory at Log/pcd/lidar_poses.txt (world<-LiDAR, "ts tx ty tz qx qy qz qw").
set -eo pipefail
source /opt/ros/noetic/setup.bash
FL2=/root/catkin_ws/src/FAST-LIVO2
BAG="${1:-/data/harrison_square_livox_lidar.bag}"
RATE="${2:-0.5}"
[ -f "$BAG" ] || { echo "ERROR: bag missing at $BAG" >&2; exit 2; }

echo "=== configure LIO-only (img_en 0, no colmap) ==="
cd "$FL2"
cp -f config/mid360.yaml /tmp/mid360.yaml.orig
sed -i 's/img_en: *1/img_en: 0/' config/mid360.yaml
sed -i 's/colmap_output_en: *true/colmap_output_en: false/' config/mid360.yaml
mkdir -p Log/pcd; rm -f Log/pcd/lidar_poses.txt
grep -E 'img_en|pcd_save_en|colmap_output_en' config/mid360.yaml

echo "=== build (catkin_make) ==="
cd /root/catkin_ws
JOBS="${JOBS:-8}"  # etiquette: cap build parallelism on the shared CPU (don't grab all cores)
nice -n 10 catkin_make -DCMAKE_BUILD_TYPE=Release -j"$JOBS" > /tmp/build.log 2>&1 || { echo "BUILD FAILED:"; tail -40 /tmp/build.log; exit 1; }
source devel/setup.bash
echo "build OK"
cd "$FL2"   # back to repo root so Log/pcd + config-restore resolve (build cd'd to catkin_ws)

echo "=== launch FL2 (headless) + play bag at rate ${RATE} ==="
roslaunch fast_livo mapping_mid360.launch rviz:=false > /tmp/fl2.log 2>&1 &
sleep 12
rosbag play --rate "$RATE" "$BAG" 2>&1 | tail -4 || true
sleep 8   # let the last LIO updates flush to disk

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
