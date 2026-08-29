# Object extraction

Finds object instances in the camera streams of a prepped sequence. YOLO11x-seg performs segmentation, ByteTrack keeps instances consistent across frames, and SAM2 refines the best view of each instance into a clean crop. The crops feed the asset_generation module.

## Files

- `perception_tracks.py` — the pipeline stage. Reads a KITTI-style sequence directory (the output of the PINGS prep step in `reconstruction/pings/`) and writes, per sequence: uint16 instance-mask PNGs for every frame of every camera (one global instance id counter across all cameras), per-camera track JSONs, SAM2-refined white-background best-view crops (`objects/crops_raw/inst_<id>.png`), and a stats report for validation. Idempotent per camera; pass `--force` to redo. The full artifact spec is documented in the script's docstring.
- `_perception_box.sh` — the host launcher for the shared multi-GPU box: single-GPU exposure, cpu/mem caps, the ultralytics weights cache mounted so YOLO/SAM weights download exactly once, and outputs chowned back to the host user. Prints a `PERCEPTION_DONE` sentinel on success. The script bind-mounts this directory into the container, so it works from wherever the repo is cloned.
- `Dockerfile` — the ultralytics base image, nothing baked in. `perception_tracks.py` is bind-mounted, so editing it never requires a rebuild.

## Usage

```
bash _perception_box.sh <seq_name> [gpu]
```

`ROOT` (default `/scratch/ali497`) sets the base directory for prepped data and the weights cache. Extra stage arguments pass through `PERC_ARGS`.

## Class policy

Animals and people are always masked and never asset-eligible (see `MASK_ONLY_CLASSES` in `perception_tracks.py`): they get removed from the reconstruction but no asset is generated for them.
