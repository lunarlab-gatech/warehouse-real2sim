#!/usr/bin/env python3
# NKSR reconstruction driver. Reads a YAML config, loads an oriented-or-sensored
# point cloud (PLY), runs Neural Kernel Surface Reconstruction, writes a mesh PLY.
# Invoked as: python /opt/nksr/run_nksr.py /work/nksr_config.yaml
#
# Config keys (all optional except input/output):
#   input:          path to input .ply (needs sensor_x/y/z OR nx/ny/nz)
#   output:         path to output mesh .ply
#   detail_level:   0.0-1.0, higher = more detail (default 0.5); null = model scale
#   chunk_size:     spatial chunk in meters for large scenes, <=0 disables (default 51.2)
#   voxel_size:     override voxel size (invalidates detail_level) (default null)
#   mise_iter:      dual-marching-cubes refinement iterations (default 1)
#   normal_knn:     kNN for sensor-based normal estimation (default 64)
#   normal_max_angle: reject points whose normal-to-sensor angle exceeds this (default 85)
#   trim:           trim mesh faces in low-confidence/unsupported regions (default true)
import sys
import numpy as np
import torch
import yaml
from plyfile import PlyData, PlyElement


def load_cloud(path):
    ply = PlyData.read(path)
    v = ply["vertex"].data
    names = v.dtype.names
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
    sensor = normal = None
    if all(k in names for k in ("sensor_x", "sensor_y", "sensor_z")):
        sensor = np.stack([v["sensor_x"], v["sensor_y"], v["sensor_z"]], axis=1).astype(np.float32)
    if all(k in names for k in ("nx", "ny", "nz")):
        normal = np.stack([v["nx"], v["ny"], v["nz"]], axis=1).astype(np.float32)
    return xyz, sensor, normal


def save_mesh(path, v, f):
    vert = np.array([tuple(p) for p in v], dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    face = np.array([(list(map(int, tri)),) for tri in f],
                    dtype=[("vertex_indices", "i4", (3,))])
    PlyData([PlyElement.describe(vert, "vertex"),
             PlyElement.describe(face, "face")], text=False).write(path)


def main():
    cfg = yaml.safe_load(open(sys.argv[1]))
    dev = torch.device("cuda:0")
    print(f"[nksr] torch {torch.__version__}  cuda_available={torch.cuda.is_available()}", flush=True)
    if not torch.cuda.is_available():
        sys.exit("[nksr] FATAL: CUDA not available (driver/runtime mismatch)")
    print(f"[nksr] GPU: {torch.cuda.get_device_name(0)}", flush=True)

    import nksr
    xyz_np, sensor_np, normal_np = load_cloud(cfg["input"])
    print(f"[nksr] loaded {len(xyz_np):,} points  sensor={'yes' if sensor_np is not None else 'no'}"
          f"  normal={'yes' if normal_np is not None else 'no'}", flush=True)

    xyz = torch.from_numpy(xyz_np).to(dev)
    sensor = torch.from_numpy(sensor_np).to(dev) if sensor_np is not None else None
    normal = torch.from_numpy(normal_np).to(dev) if normal_np is not None else None

    reconstructor = nksr.Reconstructor(dev)
    reconstructor.chunk_tmp_device = torch.device("cpu")

    preprocess_fn = None
    if sensor is not None and normal is None:
        preprocess_fn = nksr.get_estimate_normal_preprocess_fn(
            int(cfg.get("normal_knn", 64)), float(cfg.get("normal_max_angle", 85.0)))

    field = reconstructor.reconstruct(
        xyz, normal=normal, sensor=sensor,
        detail_level=cfg.get("detail_level", 0.5),
        voxel_size=cfg.get("voxel_size", None),
        chunk_size=float(cfg.get("chunk_size", 51.2)),
        approx_kernel_grad=True, solver_tol=1e-4, fused_mode=True,
        preprocess_fn=preprocess_fn,
    )
    if field is None:
        sys.exit("[nksr] FATAL: reconstruction returned no field")

    # trim=True drops faces in low-confidence / unsupported regions (reduces
    # hallucinated surface in holes); it is built into dual-mesh extraction.
    mesh = field.extract_dual_mesh(
        mise_iter=int(cfg.get("mise_iter", 1)), trim=bool(cfg.get("trim", True)))

    def to_np(x):
        return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
    v, f = to_np(mesh.v), to_np(mesh.f)
    print(f"[nksr] mesh: {len(v):,} verts  {len(f):,} faces", flush=True)
    save_mesh(cfg["output"], v, f)
    print(f"[nksr] wrote {cfg['output']}", flush=True)


if __name__ == "__main__":
    main()
