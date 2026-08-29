#!/usr/bin/env bash
# Host: re-run the KITTI smoke test WITHOUT rebuilding the image (PhysX-Omni
# parity). Use after _setup_and_run.sh has built pings:cu118 once. Edits to
# _run.sh / configs take effect immediately since the repo is bind-mounted.
set -e
cd "$HOME/projects/pings"

docker rm -f pings_run 2>/dev/null || true
docker run --rm --name pings_run --gpus all --shm-size=128g \
  -v "$HOME/projects/pings:/packages/pings" \
  -v "$HOME/.cache/torch:/root/.cache/torch" \
  pings:cu118 bash -lc 'bash /packages/pings/_run.sh'
