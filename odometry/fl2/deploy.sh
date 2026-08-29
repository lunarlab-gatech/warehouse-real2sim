#!/usr/bin/env bash
# Deploy the FL2 launcher overlay onto a Fast-Livo2-GeoScan checkout.
#   bash deploy.sh [target_dir]     (default: $ROOT/projects/Fast-Livo2-GeoScan,
#                                    ROOT defaults to $HOME — the layout _fl2_box.sh expects)
#
# The launchers in this directory are kept here rather than committed to the
# lab's Fast-Livo2-GeoScan fork; the in-container scripts must sit INSIDE the
# FL2 checkout at run time (the host launchers execute them through the repo
# bind-mount), so this script clones the fork at a pinned commit if it is
# missing and copies the overlay files in. Re-run any time to refresh the
# overlay after pulling changes to this repo. Override the pin with $FL2_REF.
set -e
SELF_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="${ROOT:-$HOME}"
FL2_REF="${FL2_REF:-a6c8c7c865bf21205d41938980c313041f5514df}"
DEST="${1:-$ROOT/projects/Fast-Livo2-GeoScan}"

if [ ! -d "$DEST/.git" ]; then
  echo "[deploy] cloning Fast-Livo2-GeoScan @ $FL2_REF -> $DEST"
  git clone https://github.com/lunarlab-gatech/Fast-Livo2-GeoScan.git "$DEST"
  git -C "$DEST" checkout "$FL2_REF"
fi

cp "$SELF_DIR/Dockerfile" "$SELF_DIR/.dockerignore" \
   "$SELF_DIR/_fl2_box.sh" "$SELF_DIR/_fl2_lio_incontainer.sh" \
   "$SELF_DIR/_fl2_manifold_box.sh" "$SELF_DIR/_fl2_manifold_incontainer.sh" "$DEST/"
mkdir -p "$DEST/config"
cp "$SELF_DIR/config/mid360_manifold.yaml" "$DEST/config/"

echo "FL2_OVERLAY_DEPLOYED dest=$DEST ref=$FL2_REF"
echo "Next: build the image (docker build -t lunarlab/fastlivo2:base \"$DEST\"), then run bags with _fl2_box.sh / _fl2_manifold_box.sh from $DEST."
