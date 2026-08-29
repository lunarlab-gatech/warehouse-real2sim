#!/usr/bin/env bash
# Perception stage [C] host launcher (YOLO11x-seg + ByteTrack masks + SAM2 crops).
#   bash _perception_box.sh <seq_name> [gpu]
# Writes $SEQ/masks + $SEQ/objects per pipeline/CONTRACTS.md, where
#   SEQ=$ROOT/projects/pings/data/geoscan_<seq_name>/sequences/00
# (produced earlier by the PINGS prep step, reconstruction/pings/prep.sh). On
# success prints the PERCEPTION_DONE sentinel line the orchestrator greps for.
#
# Server etiquette (shared multi-GPU box): exposes ONLY
# the requested GPU (default 1; deliberately no auto-pick here -- pass the index),
# caps cpu/mem/shm, container named ali497_<seq_name>_perception, --rm, default
# bridge network, chowns outputs back to the host user on container exit.
#   ROOT            base dir (default /scratch/ali497)
#   GPU             device index (arg 2 wins, then $GPU, then 1)
#   IMG             docker image (default yolo_rosbag_experiment, the name already
#                   built on the lab box; built from this dir if absent)
#   CONTAINER_NAME  override container name (concurrent runs)
#   FORCE=1         redo cached cameras (forwards --force)
#   PERC_ARGS       extra perception_tracks.py args (e.g. "--cams 2,4 --conf 0.5")
# Weights cache: $ROOT/.cache/ultralytics is mounted at /weights so
# yolo11x-seg.pt / sam2.1_l.pt download exactly once across runs.
trap 'echo "ORCH_EXIT_$?"' EXIT
set -e
ROOT="${ROOT:-/scratch/ali497}"

# this module's own directory: docker build context + the /workspace mount,
# so the launcher works from wherever the repo is cloned
SELF_DIR="$(cd "$(dirname "$0")" && pwd)"

NAME="${1:?usage: _perception_box.sh <seq_name> [gpu]}"
GPU="${2:-${GPU:-1}}"
IMG="${IMG:-yolo_rosbag_experiment}"
CNAME="${CONTAINER_NAME:-ali497_${NAME}_perception}"
SEQ_HOST="$ROOT/projects/pings/data/geoscan_$NAME/sequences/00"
SEQ_CTR="/pings_data/geoscan_$NAME/sequences/00"
[ -d "$SEQ_HOST/image_2" ] || { echo "ERROR: sequence missing at $SEQ_HOST (run the PINGS prep first)"; exit 2; }
mkdir -p "$ROOT/.cache/ultralytics"

echo "NAME=$NAME GPU=$GPU IMG=$IMG SEQ=$SEQ_HOST"
echo "[gpu] requested device $GPU:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader -i "$GPU" || true

# build once if missing (ultralytics base; scripts are bind-mounted, not COPY'd,
# so code edits never require a rebuild)
docker image inspect "$IMG" >/dev/null 2>&1 \
  || docker build -t "$IMG" "$SELF_DIR"

FORCE_ARG=""
[ "${FORCE:-0}" = "1" ] && FORCE_ARG="--force"

# --entrypoint bash keeps the run shell-driven regardless of the image's
# entrypoint. Single GPU exposed => device 0 inside the container regardless
# of the host index.
docker rm -f "$CNAME" 2>/dev/null || true
docker run --rm --gpus "device=$GPU" --cpus=16 --memory=32g --shm-size=8g \
  --name "$CNAME" --entrypoint bash \
  -e OMP_NUM_THREADS=8 -e MKL_NUM_THREADS=8 \
  -e YOLO_CONFIG_DIR=/weights \
  -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" \
  -v "$SELF_DIR:/workspace" \
  -v "$ROOT/projects/pings/data:/pings_data" \
  -v "$ROOT/.cache/ultralytics:/weights" \
  "$IMG" -lc "set -eo pipefail
    trap 'chown -R \$HOST_UID:\$HOST_GID /pings_data/geoscan_$NAME /weights 2>/dev/null || true' EXIT
    python3 /workspace/perception_tracks.py --seq $SEQ_CTR --device 0 \
      --yolo /weights/yolo11x-seg.pt --sam /weights/sam2.1_l.pt \
      $FORCE_ARG ${PERC_ARGS:-}"

echo "PERCEPTION_DONE name=$NAME seq=$SEQ_HOST"
