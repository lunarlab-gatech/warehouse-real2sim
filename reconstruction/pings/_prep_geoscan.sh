#!/usr/bin/env bash
# Runs INSIDE pings:cu118 (see prep.sh for the docker wrapper). Converts a
# GeoScan Livox bag into a 3-camera KITTI sequence: LiDAR + per-point timestamps
# (deskew) + right/left fisheye + RealSense, timestamp-matched, with per-frame
# camera capture times (times_camN.txt) for PINGS' time-sync correction.
# No poses are produced: PINGS estimates the trajectory with its own tracker.
# Each stage is cached independently (safe to re-run).
#   bash _prep_geoscan.sh <BAG> <DATADIR>
set -eo pipefail
cd /packages/pings
BAG="${1:?need bag path, e.g. /data/office_6_19_2026.bag}"
DATADIR="${2:?need dataset dir, e.g. ./data/geoscan_office619}"
SEQ="$DATADIR/sequences/00"

echo "=== ensure rosbags (pure-python) in container ==="
python3 -c "import rosbags" 2>/dev/null || pip install --quiet --no-input rosbags

# --- LiDAR + per-point timestamps (re-run if velodyne_ts missing -> deskew) ----
if [ "$(ls "$SEQ"/velodyne_ts/*.bin 2>/dev/null | wc -l)" -gt 0 ]; then
  echo "lidar+ts cached -> skipping LiDAR conversion"
else
  [ -f "$BAG" ] || { echo "ERROR: bag missing at $BAG" >&2; exit 2; }
  echo "=== convert LiDAR (velodyne + per-point ts + times.txt + calib) ==="
  python3 scripts/convert_geoscan_bag.py "$BAG" "$SEQ" --geoscan
fi

# --- cameras (cached; re-run all three only if missing) ------------------------
if [ -f "$SEQ/cam4.json" ] && [ "$(ls "$SEQ"/image_4/*.png 2>/dev/null | wc -l)" -gt 0 ]; then
  echo "cameras cached -> skipping camera extraction"
else
  [ -f "$BAG" ] || { echo "ERROR: bag missing at $BAG" >&2; exit 2; }
  for c in right left realsense; do
    echo "=== add_camera $c ==="
    python3 scripts/add_camera.py "$BAG" "$SEQ" --camera "$c"   # offset from the calib registry
  done
fi

echo "=== converted sequence counts ==="
for d in velodyne velodyne_ts image_2 image_3 image_4; do
  printf "  %-14s %s\n" "$d:" "$(ls "$SEQ/$d" 2>/dev/null | wc -l | tr -d ' ')"
done
for f in times.txt times_cam2.txt times_cam3.txt times_cam4.txt; do
  [ -f "$SEQ/$f" ] && printf "  %-14s %s rows\n" "$f:" "$(wc -l < "$SEQ/$f" | tr -d ' ')"
done
echo "PREP_DONE"
