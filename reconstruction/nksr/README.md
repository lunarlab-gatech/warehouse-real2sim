# NKSR meshing

Reconstructs a surface mesh directly from the accumulated LiDAR point cloud using NVIDIA's Neural Kernel Surface Reconstruction (NKSR). This backend produces geometry only, no color.

The NKSR source is not in this repository. The Dockerfile clones [nv-tlabs/nksr](https://github.com/nv-tlabs/nksr) at build time, pinned to commit `14d7567`, and bakes the compiled package into the image. NKSR is distributed under the NVIDIA Source Code License (non-commercial research use).

## Files

- `Dockerfile` — builds the self-contained `nksr:dev` image: conda env, PyTorch with CUDA, and the compiled NKSR package. The GPU architecture is a build argument (`TORCH_CUDA_ARCH_LIST`, default 8.6 for RTX 30xx).
- `accumulate_fl2_with_sensor.py` — builds the reconstruction input. Reads BALM's body-frame clouds plus the FL2 trajectory, transforms each scan to world frame, tags every point with its sensor position (NKSR's LiDAR normal-estimation path needs this), optionally voxel-downsamples, and writes a binary PLY with fields x, y, z, sensor_x, sensor_y, sensor_z.
- `run_nksr.py` — the in-container driver. Reads a YAML config (paths, detail level, chunk size, voxel size, trim), runs chunked reconstruction with sensor-based normal estimation, and writes the mesh as binary PLY.

## Usage

Build the image once:

```
docker build -t nksr:dev .
```

Prepare the input cloud, then run the reconstruction with the export folder mounted at `/work`:

```
docker run --gpus all --rm --user "$(id -u):$(id -g)" \
  -v "$EXPORT_DIR":/work nksr:dev \
  python /opt/nksr/run_nksr.py /work/nksr_config.yaml
```

## Upstream dependency

The input for `accumulate_fl2_with_sensor.py` comes from the odometry module: `odometry/convert_fl2_to_balm.py` converts FAST-LIVO2 output into the BALM format (per-scan body-frame clouds plus `alidarPose.csv`) that this script reads.
