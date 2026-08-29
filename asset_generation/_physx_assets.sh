#!/usr/bin/env bash
# PhysX-Omni asset stage [D3] (CONTRACTS.md): gobj crops -> sim-ready assets.
#   bash _physx_assets.sh <crops_dir> <out_assets_dir> <gpu> [gpu2]
#   bash _physx_assets.sh --setup            # one-time: clone+patch repo, docker build, weight download
# <crops_dir> = $SEQ/objects/crops_final (gobj_XXXX.png, must live under $ROOT so the
# bind mount sees it), <out_assets_dir> = $SCENE/assets. With [gpu2] the pending crops
# are split round-robin into two symlink shards run as two concurrent containers.
#
# The PhysX-Omni source is NOT in this repository (non-commercial S-Lab license):
# --setup clones it from upstream at the pinned commit $PHYSX_REF into $PHYSX and
# applies physx_omni_lunar.patch (attention-backend + OOM fixes; see README.md).
#
# Hot path does NO docker build / download.py: stage image + pretrain/ + hf_cache with
# --setup first (the only mode that goes online). 1vlm_demo.py hardcodes savedir
# 'ours_demo' relative to cwd and 2infer/3jsongen scan that whole dir, so each shard
# runs in its own shadow workdir ($PHYSX/_shards/<tag>/sN) of relative symlinks into
# the repo -- disjoint ours_demo caches, no cross-container races, cache stable across
# re-runs. Idempotency: a crop is skipped when <out>/<gobj>/basic.xml exists; inside a
# shard the stage scripts skip finished work themselves (allind.npy / objs count). Force:
#   rm -rf <out>/<gobj>                                          # re-copy from shard cache
#   rm -rf $ROOT/projects/PhysX-Omni/_shards/<tag>/sN/ours_demo/<gobj>   # full recompute
# Etiquette (shared box): containers ${CONTAINER_NAME:-ali497_physx}_*, --rm, explicit
# --gpus device=N only (never auto-picked), cpu/mem/shm caps, offline HF by default.
trap 'echo "ORCH_EXIT_$?"' EXIT
set -e
shopt -s nullglob
ROOT="${ROOT:-/scratch/ali497}"
PHYSX="$ROOT/projects/PhysX-Omni"
PHYSX_REF="${PHYSX_REF:-5ba54ee3d0e11c8690fd414d2343d47e514930dd}"
IMAGE="${IMAGE:-physx_omni_image}"
CPREFIX="${CONTAINER_NAME:-ali497_physx}"

# this module's own directory: holds the Dockerfile (build context) and the patch
SELF_DIR="$(cd "$(dirname "$0")" && pwd)"

DRUN=(docker run --rm --cpus="${PHYSX_CPUS:-16}" --memory="${PHYSX_MEM:-64g}" --shm-size="${PHYSX_SHM:-16g}"
      -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)"
      -v "$ROOT:/data_volume" -v "$PHYSX/hf_cache:/root/.cache/huggingface")

if [ "${1:-}" = "--setup" ]; then  # explicit clone + patch + build + download, kept OUT of the hot path
  if [ ! -d "$PHYSX/.git" ]; then
    echo "[setup] cloning PhysX-Omni @ $PHYSX_REF -> $PHYSX"
    git clone https://github.com/physx-omni/PhysX-Omni.git "$PHYSX"
    git -C "$PHYSX" checkout "$PHYSX_REF"
  fi
  if git -C "$PHYSX" apply --reverse --check "$SELF_DIR/physx_omni_lunar.patch" 2>/dev/null; then
    echo "[setup] physx_omni_lunar.patch already applied"
  else
    git -C "$PHYSX" apply "$SELF_DIR/physx_omni_lunar.patch"
    echo "[setup] applied physx_omni_lunar.patch"
  fi
  docker build -t "$IMAGE" "$SELF_DIR"
  docker rm -f "${CPREFIX}_setup" 2>/dev/null || true
  "${DRUN[@]}" --name "${CPREFIX}_setup" -w /data_volume/projects/PhysX-Omni "$IMAGE" /bin/bash -c "set -e
    trap 'chown -R \$HOST_UID:\$HOST_GID /data_volume/projects/PhysX-Omni/pretrain /root/.cache/huggingface 2>/dev/null || true' EXIT
    pip install huggingface_hub && python download.py
    python -c \"from transformers import AutoProcessor; AutoProcessor.from_pretrained('Qwen/Qwen2.5-VL-7B-Instruct')\""
  echo "SETUP_DONE"; exit 0
fi

CROPS="${1:?usage: _physx_assets.sh <crops_dir> <out_assets_dir> <gpu> [gpu2] | --setup}"
OUT="${2:?need <out_assets_dir>}"; GPU="${3:?need <gpu> index (never auto-picked on the shared box)}"; GPU2="${4:-}"
[ -d "$CROPS" ] || { echo "ERROR: crops_dir missing: $CROPS"; exit 2; }
CROPS="$(cd "$CROPS" && pwd)"; mkdir -p "$OUT"; OUT="$(cd "$OUT" && pwd)"
case "$CROPS" in "$ROOT"/*) ;; *) echo "ERROR: crops_dir must be under ROOT=$ROOT (bind mount)"; exit 2;; esac
TAG="$(basename "$(dirname "$OUT")")_$(basename "$OUT")"   # e.g. street_assets -> per-scene shard cache
WORK="$PHYSX/_shards/$TAG"
echo "CROPS=$CROPS OUT=$OUT GPU=$GPU GPU2=${GPU2:-none} IMAGE=$IMAGE WORK=$WORK"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader -i "$GPU" 2>/dev/null || true

# ---- collect pending crops (skip = already harvested into <out>) ----
PENDING=(); SKIP_N=0
for png in "$CROPS"/*.png; do
  base="$(basename "$png" .png)"
  if [ -f "$OUT/$base/basic.xml" ]; then echo "SKIP $base (cached in $OUT)"; SKIP_N=$((SKIP_N+1))
  else PENDING+=("$base"); fi
done

NSH=1; if [ -n "$GPU2" ]; then NSH=2; fi
declare -a S0=() S1=()
prep_shard() {  # shadow workdir of relative symlinks so 'ours_demo' is per-shard
  mkdir -p "$1/crops" "$1/ours_demo"; rm -f "$1"/crops/*.png
  for item in 1vlm_demo.py 2infer_geo.py 3jsongen_update.py decoder_each.py dataset trellis pretrain mjcf_source; do
    ln -sfn "../../../$item" "$1/$item"
  done
}
if [ "${#PENDING[@]}" -gt 0 ]; then
  prep_shard "$WORK/s0"; if [ "$NSH" = 2 ]; then prep_shard "$WORK/s1"; fi
  RELC0="$(python3 -c 'import os,sys; print(os.path.relpath(sys.argv[1], sys.argv[2]))' "$CROPS" "$WORK/s0/crops")"
  RELC1="$(python3 -c 'import os,sys; print(os.path.relpath(sys.argv[1], sys.argv[2]))' "$CROPS" "$WORK/s1/crops")"
  i=0
  for base in "${PENDING[@]}"; do  # round-robin split
    if [ $((i % NSH)) = 0 ]; then ln -sfn "$RELC0/$base.png" "$WORK/s0/crops/$base.png"; S0+=("$base")
    else ln -sfn "$RELC1/$base.png" "$WORK/s1/crops/$base.png"; S1+=("$base"); fi
    i=$((i+1))
  done
fi

run_shard() {  # $1 shard dir (host), $2 gpu index, $3 container suffix
  local cdir="/data_volume${1#"$ROOT"}" cname="${CPREFIX}_$3"
  docker rm -f "$cname" 2>/dev/null || true
  "${DRUN[@]}" --name "$cname" --gpus "device=$2" -w "$cdir" \
    -e HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" -e TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}" \
    "$IMAGE" /bin/bash -c "set -eo pipefail
      trap 'chown -R \$HOST_UID:\$HOST_GID $cdir 2>/dev/null || true' EXIT
      python 1vlm_demo.py --imagepath ./crops --modelpath pretrain
      python 2infer_geo.py --outputpath ./ours_demo
      python 3jsongen_update.py --basepath ./ours_demo"
}

RC0=0; RC1=0
if [ "${#S0[@]}" -gt 0 ] && [ "${#S1[@]}" -gt 0 ]; then
  run_shard "$WORK/s0" "$GPU"  s0 >"$WORK/s0.log" 2>&1 & P0=$!
  run_shard "$WORK/s1" "$GPU2" s1 >"$WORK/s1.log" 2>&1 & P1=$!
  echo "[shards] s0(gpu$GPU)=${#S0[@]} s1(gpu$GPU2)=${#S1[@]} crops; logs: $WORK/s0.log $WORK/s1.log"
  wait "$P0" || RC0=$?; wait "$P1" || RC1=$?
  echo "[shards] s0 exit=$RC0 s1 exit=$RC1"
elif [ "${#S0[@]}" -gt 0 ]; then
  run_shard "$WORK/s0" "$GPU" s0 || RC0=$?
fi

OK_N=0; FAIL_N=0
harvest() { local d="$1"; shift  # copy finished assets -> <out>/<gobj>/ verbatim
  for base in "$@"; do
    if [ -f "$d/ours_demo/$base/basic.xml" ]; then
      mkdir -p "$OUT/$base"; rsync -a "$d/ours_demo/$base/" "$OUT/$base/"
      rm -f "$OUT/$base/basic_scaled.urdf"  # stale physics from a prior fix pass -> force refit
      echo "OK $base"; OK_N=$((OK_N+1))
    else echo "FAIL $base (no basic.xml in $d/ours_demo/$base)"; FAIL_N=$((FAIL_N+1)); fi
  done
}
if [ "${#S0[@]}" -gt 0 ]; then harvest "$WORK/s0" "${S0[@]}"; fi
if [ "${#S1[@]}" -gt 0 ]; then harvest "$WORK/s1" "${S1[@]}"; fi
echo "ASSETS_DONE ok=$OK_N fail=$FAIL_N skip=$SKIP_N out=$OUT"
if [ "$FAIL_N" -gt 0 ]; then exit 1; fi
