# Odometry

Produces a per-scan trajectory from the LiDAR and IMU streams. Two backends are maintained as their own repositories under this organization, and each carries its own Docker build, launchers, and rig configs:

- [Fast-Livo2-GeoScan](https://github.com/lunarlab-gatech/Fast-Livo2-GeoScan) — FAST-LIVO2 run in LIO-only mode. The launchers and configs for it live in this repo under `fl2/` and are deployed as an overlay onto a checkout of the fork: `bash fl2/deploy.sh` clones the fork at a pinned commit (if missing) and copies the files in. The launchers (`_fl2_box.sh`, `_fl2_manifold_box.sh`) then run a bag or a multi-bag Manifold scene headless in Docker and produce the per-scan trajectory at `Log/pcd/lidar_poses.txt` (TUM format). The Manifold rig config is `fl2/config/mid360_manifold.yaml`.
- [lio_sam_mid360](https://github.com/lunarlab-gatech/lio_sam_mid360) — LIO-SAM with loop closure, used when drift-corrected trajectories matter. `tools/liosam_box.sh` runs it headless on a merged ROS 2 bag (see `preprocessing/merge_manifold_ros1_to_ros2.py`) and saves the loop-corrected keyframe map via the save_map service.

A ScanContext loop-closure variant of FAST-LIVO2 (SC-PGO) also exists, but its source currently lives only in a separate clone on the lab server and is not yet published.

This directory holds the pose-format conversion scripts that sit between the backends and the rest of the pipeline.

## Files

- `liosam_to_poses.py` — densifies a LIO-SAM trajectory into per-scan poses for pose-fixed reconstruction. Inputs: the loop-corrected keyframe poses (`transformations.pcd` from the save_map service), a recorded bag with `/lio_sam/mapping/odometry_incremental` (the jump-free stream — the plain odometry stream jumps at loop closures), and the sequence's `times.txt`. Each scan pose is transferred from both neighboring keyframes and blended, so the output stays anchored to the corrected trajectory. Writes KITTI 3x4 rows and self-validates with inter-scan speed and anchor-consistency checks. Needs numpy, scipy, and rosbags.

```
python3 liosam_to_poses.py <transformations.pcd> <odom_bag_dir> <times.txt> <out_poses.txt>
```

- `convert_fl2_to_balm.py` — converts FAST-LIVO2 output (world-frame `Log/pcd/*.pcd` clouds plus the TUM trajectory) into the input layout BALM2's benchmark_realworld node expects: per-scan body-frame `full*.pcd` clouds and `alidarPose.csv`. Matching is nearest-timestamp with a configurable tolerance. Needs numpy only.

```
python3 convert_fl2_to_balm.py --pcd-dir <Log/pcd> --poses <lidar_poses.txt> --out <balm_input_dir>
```

The BALM output is also what `reconstruction/nksr/accumulate_fl2_with_sensor.py` consumes to build the NKSR input cloud.
