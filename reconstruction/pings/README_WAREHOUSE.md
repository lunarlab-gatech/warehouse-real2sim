# PINGS changes

This directory contains the full source code of PINGS, as well as a few specific changes to the source code that are listed in this readme.

## Modified files

- `utils/mapper.py`: object-mask pixels are excluded from all supervision (RGB, depth, normals, distortion), and sky pixels are excluded from the RGB loss (lines 1221-1236, 1257-1270, 1290-1291)
- `utils/tracker.py` and `utils/config.py`: the registration validity gate (`min_valid_points`, `min_valid_ratio`) are configurable via `tracker.min_valid_points` and `tracker.min_valid_ratio` rather than hardcoded
- `gaussian_splatting/utils/cameras.py`: `CamImage` gains an optional object-mask pyramid alongside the rgb, depth, and sky-mask pyramids, plus safe defaults for pose-only cameras
- `dataset/slam_dataset.py`: passes the per-camera object mask from the frame data to `CamImage`
- `inspect_pings.py`: added evaluation system, per-camera PSNR/SSIM/LPIPS reporting, and fixed evaluation seeds
- `.gitignore`

## New files

Dataloader:
- `dataset/dataloaders/geoscan.py`: dataloader for the GeoScan and Manifold devices

Data preparation:
- `prep.sh` and `_prep_geoscan.sh`: prepares a GeoScan rosbag as a KITTI-format sequence inside Docker
- `prep_manifold.sh`: preparation process for converting multi-rosbag sequences from the Manifold device, calls `scripts/convert_manifold_bags.py`.
- `scripts/convert_geoscan_bag.py`, `scripts/convert_manifold_bags.py`: converts the bags to a sequence that is legible to PINGS
- `scripts/geoscan_calib.py`, `scripts/manifold_calib.py`: calibration definitions for the two rigs

Running:

These example shell files begin the PINGS runs
- `run.sh`
- `run_a4.sh`
- `Dockerfile.lunar4090`: builds the `pings:cu118` image that the launchers use

Configs:

Example yaml file configurations for different runs / scenes
- `config/run_geoscan_gs.yaml`
- `config/run_geoscan_gs_A1.yaml` through `A4.yaml`"
- `config/run_manifold_gs.yaml`