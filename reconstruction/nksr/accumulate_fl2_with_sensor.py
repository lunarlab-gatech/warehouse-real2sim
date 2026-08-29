#!/usr/bin/env python3
# Rebuild the FL2 point cloud carrying per-point SENSOR positions, for NKSR.
# NKSR's LiDAR path estimates oriented normals from the sensor origin of each
# point (get_estimate_normal_preprocess_fn), which avoids the slow/ambiguous
# MST normal orientation a bare merged cloud would need.
#
# Reuses BALM's body-frame clouds (full*.pcd) + FL2 poses (alidarPose.csv, the
# UNoptimized FL2 trajectory): for scan k, sensor = pose_k translation and
# world = pose_k * body_points. Writes a binary PLY with fields
# x,y,z,sensor_x,sensor_y,sensor_z (the layout NKSR's waymo loader expects).
#
#   python3 accumulate_fl2_with_sensor.py <data_dir> <poses.csv> <out.ply> [--voxel 0.02]
import argparse
from pathlib import Path

import numpy as np


def read_pcd_xyz(path):
    with open(path, "rb") as f:
        fields = []
        while True:
            line = f.readline().decode("ascii", "replace")
            t = line.split()
            if t and t[0] == "FIELDS":
                fields = t[1:]
            if t and t[0] == "POINTS":
                n = int(t[1])
            if t and t[0] == "DATA":
                mode = t[1].strip()
                break
        if mode == "binary":
            # assume float32 for all fields (XYZI etc.)
            nf = len(fields)
            arr = np.frombuffer(f.read(n * nf * 4), np.float32).reshape(n, nf)
            return arr[:, :3].astype(np.float64)
        else:
            arr = np.loadtxt(f, dtype=np.float64, max_rows=n).reshape(n, -1)
            return arr[:, :3]


def load_poses_csv(path):
    """BALM 4-lines-per-pose CSV -> N x 4 x 4."""
    rows = [[float(x) for x in l.replace(",", " ").split()] for l in open(path) if l.strip()]
    a = np.array(rows).reshape(-1, 4, 4)
    return a


def write_ply(path, xyz, sensor):
    n = len(xyz)
    hdr = ("ply\nformat binary_little_endian 1.0\n"
           f"element vertex {n}\n"
           "property float x\nproperty float y\nproperty float z\n"
           "property float sensor_x\nproperty float sensor_y\nproperty float sensor_z\n"
           "end_header\n")
    buf = np.empty((n, 6), np.float32)
    buf[:, :3] = xyz
    buf[:, 3:] = sensor
    with open(path, "wb") as f:
        f.write(hdr.encode("ascii"))
        buf.tofile(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir", help="dir with full0.pcd..fullN.pcd")
    ap.add_argument("poses", help="alidarPose.csv (FL2 poses)")
    ap.add_argument("out")
    ap.add_argument("--voxel", type=float, default=0.02)
    a = ap.parse_args()

    T = load_poses_csv(a.poses)
    n = len(T)
    clouds = sorted(Path(a.data_dir).glob("full*.pcd"),
                    key=lambda p: int(p.stem[4:]))
    assert len(clouds) == n, f"{len(clouds)} clouds != {n} poses"

    xyz_all, sensor_all = [], []
    for k, c in enumerate(clouds):
        body = read_pcd_xyz(c)
        world = (T[k, :3, :3] @ body.T).T + T[k, :3, 3]
        xyz_all.append(world.astype(np.float32))
        sensor_all.append(np.broadcast_to(T[k, :3, 3].astype(np.float32), world.shape).copy())
        if (k + 1) % 200 == 0:
            print(f"  ... {k+1}/{n}", flush=True)
    xyz = np.concatenate(xyz_all)
    sensor = np.concatenate(sensor_all)
    print(f"merged {len(xyz):,} points")

    if a.voxel > 0:
        keys = np.floor(xyz.astype(np.float64) / a.voxel).astype(np.int64)
        _, idx = np.unique(keys, axis=0, return_index=True)
        idx = np.sort(idx)
        xyz, sensor = xyz[idx], sensor[idx]
        print(f"after {a.voxel} m voxel: {len(xyz):,} points")

    write_ply(a.out, xyz, sensor)
    print(f"DONE: {a.out} ({Path(a.out).stat().st_size/1e9:.2f} GB)")


if __name__ == "__main__":
    main()
