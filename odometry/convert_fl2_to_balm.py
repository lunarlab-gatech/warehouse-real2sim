#!/usr/bin/env python3
"""
convert_fl2_to_balm.py

Converts FAST-LIVO2 output into the input format expected by BALM2's
benchmark_realworld node.

Input (FAST-LIVO2 side):
  Log/pcd/<timestamp>.pcd   World-frame point clouds. One file per
                            pcd_save.interval scans; the filename is the
                            timestamp of the LAST scan folded into that file.
  Log/pcd/lidar_poses.txt   TUM format, one line per scan:
                            timestamp x y z qx qy qz qw

Output (BALM2 side, written into --out):
  full0.pcd .. fullN.pcd    Each cloud un-projected into the LiDAR body frame
                            of its matched pose:  p_body = R^T (p_world - t)
  alidarPose.csv            4 comma-separated lines per pose:
                              R00,R01,R02,tx
                              R10,R11,R12,ty
                              R20,R21,R22,tz
                              0,0,0,timestamp
                            This matches read_pose() in benchmark_realworld.cpp,
                            which fills a column-major Eigen Matrix4d linearly
                            and then transposes it -- i.e. CSV lines == matrix
                            rows of the world-from-body transform.

Matching: each PCD is paired with the pose whose timestamp is nearest to the
PCD filename (default tolerance 0.05 s). Poses without a PCD (4 of every 5
when interval=5) are silently dropped. Because BALM re-applies the exact pose
we un-project with, the initial world map inside BALM is identical to the
FAST-LIVO2 map regardless of which nearby pose anchors each cloud.

Usage (inside the fastlivo2 container, or anywhere with python3 + numpy):
  python3 convert_fl2_to_balm.py \
      --pcd-dir /data/fl2/Log/pcd \
      --poses   /data/fl2/Log/pcd/lidar_poses.txt \
      --out     /root/catkin_ws/src/BALM/datas/benchmark_realworld
"""
import argparse
import glob
import math
import os
import sys

import numpy as np

# ---------------------------------------------------------------- poses ----

def quat_to_R(qx, qy, qz, qw):
    """Rotation matrix from a (qx, qy, qz, qw) quaternion, TUM ordering."""
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw),     2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw),     1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw),     2 * (qy * qz + qx * qw),     1 - 2 * (qx * qx + qy * qy)],
    ])


def load_tum_poses(path):
    """Parse 'timestamp x y z qx qy qz qw' lines -> (times[N], R[N,3,3], t[N,3])."""
    times, Rs, ts = [], [], []
    with open(path) as f:
        for ln, line in enumerate(f, 1):
            parts = line.split()
            if not parts or parts[0].startswith("#"):
                continue
            if len(parts) != 8:
                sys.exit(f"{path}:{ln}: expected 8 values "
                         f"(t x y z qx qy qz qw), got {len(parts)}")
            v = [float(p) for p in parts]
            qn = math.sqrt(sum(q * q for q in v[4:8]))
            if abs(qn - 1.0) > 0.01:
                sys.exit(f"{path}:{ln}: quaternion norm {qn:.4f} != 1 -- "
                         f"is this really TUM 't x y z qx qy qz qw'?")
            times.append(v[0])
            ts.append(v[1:4])
            Rs.append(quat_to_R(*v[4:8]))
    if not times:
        sys.exit(f"{path}: no poses parsed")
    return np.array(times), np.array(Rs), np.array(ts)

# ------------------------------------------------------------------ pcd ----

PCD_TYPE = {("F", 4): "f4", ("F", 8): "f8",
            ("U", 1): "u1", ("U", 2): "u2", ("U", 4): "u4", ("U", 8): "u8",
            ("I", 1): "i1", ("I", 2): "i2", ("I", 4): "i4", ("I", 8): "i8"}


def read_pcd(path):
    """Read an ascii or binary (uncompressed) PCD -> (fields, sizes, types, counts, arr)."""
    with open(path, "rb") as f:
        header = {}
        data_fmt = None
        while True:
            raw = f.readline()
            if not raw:
                sys.exit(f"{path}: unexpected end of file in header")
            line = raw.decode("ascii", "replace").strip()
            if not line or line.startswith("#"):
                continue
            key, _, rest = line.partition(" ")
            header[key] = rest.split()
            if key == "DATA":
                data_fmt = rest.strip()
                break

        fields = header["FIELDS"]
        sizes = [int(s) for s in header["SIZE"]]
        types = header["TYPE"]
        counts = [int(c) for c in header.get("COUNT", ["1"] * len(fields))]
        if "POINTS" in header:
            npts = int(header["POINTS"][0])
        else:
            npts = int(header["WIDTH"][0]) * int(header["HEIGHT"][0])

        dt = []
        for name, s, t, c in zip(fields, sizes, types, counts):
            base = PCD_TYPE[(t, s)]
            dt.append((name, base, (c,)) if c > 1 else (name, base))
        dtype = np.dtype(dt)

        if data_fmt == "binary":
            need = npts * dtype.itemsize
            buf = f.read(need)
            if len(buf) < need:
                sys.exit(f"{path}: truncated binary data "
                         f"({len(buf)} of {need} bytes)")
            arr = np.frombuffer(buf, dtype=dtype, count=npts).copy()
        elif data_fmt == "ascii":
            raw = np.loadtxt(f, dtype="f8", ndmin=2)
            arr = np.zeros(npts, dtype=dtype)
            col = 0
            for name, c in zip(fields, counts):
                if c == 1:
                    arr[name] = raw[:npts, col]
                else:
                    arr[name] = raw[:npts, col:col + c]
                col += c
        else:
            sys.exit(f"{path}: DATA '{data_fmt}' not supported "
                     f"(expected 'binary' or 'ascii'; 'binary_compressed' "
                     f"would need pcl_convert_pcd_ascii_binary first)")
    return fields, sizes, types, counts, arr


def write_pcd_binary(path, fields, sizes, types, counts, arr):
    n = arr.shape[0]
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        f"FIELDS {' '.join(fields)}\n"
        f"SIZE {' '.join(map(str, sizes))}\n"
        f"TYPE {' '.join(types)}\n"
        f"COUNT {' '.join(map(str, counts))}\n"
        f"WIDTH {n}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {n}\nDATA binary\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(arr.tobytes())

# ----------------------------------------------------------------- main ----

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pcd-dir", required=True,
                    help="directory containing <timestamp>.pcd files")
    ap.add_argument("--poses", required=True,
                    help="TUM lidar_poses.txt from FAST-LIVO2")
    ap.add_argument("--out", required=True,
                    help="BALM datas/benchmark_realworld directory to write into")
    ap.add_argument("--tol", type=float, default=0.05,
                    help="max |pcd timestamp - pose timestamp| in seconds (default 0.05)")
    a = ap.parse_args()

    times, Rs, ts = load_tum_poses(a.poses)
    print(f"poses: {len(times)}  spanning {times[0]:.3f} .. {times[-1]:.3f} "
          f"({times[-1] - times[0]:.1f} s, ~{(len(times) - 1) / max(times[-1] - times[0], 1e-9):.1f} Hz)")

    pcds = []
    for p in glob.glob(os.path.join(a.pcd_dir, "*.pcd")):
        stem = os.path.splitext(os.path.basename(p))[0]
        try:
            pcds.append((float(stem), p))
        except ValueError:
            print(f"  skipping non-timestamp filename: {os.path.basename(p)}")
    pcds.sort()
    print(f"pcd files: {len(pcds)}")
    if not pcds:
        sys.exit("no <timestamp>.pcd files found in --pcd-dir")

    os.makedirs(a.out, exist_ok=True)
    removed = 0
    for old in glob.glob(os.path.join(a.out, "full*.pcd")):
        os.remove(old)
        removed += 1
    if removed:
        print(f"cleared {removed} stale full*.pcd from {a.out}")

    matched, skipped = [], []
    for ts_pcd, path in pcds:
        i = int(np.argmin(np.abs(times - ts_pcd)))
        dt = abs(times[i] - ts_pcd)
        (matched if dt <= a.tol else skipped).append((ts_pcd, path, i, dt))
    if skipped:
        print(f"WARNING: {len(skipped)} pcd(s) had no pose within {a.tol}s -- skipped:")
        for ts_pcd, path, i, dt in skipped[:10]:
            print(f"  {os.path.basename(path)}  nearest pose off by {dt:.3f}s")

    csv_path = os.path.join(a.out, "alidarPose.csv")
    total_pts = 0
    with open(csv_path, "w") as csv:
        n = 0
        for ts_pcd, path, i, dt in matched:
            R, t, tp = Rs[i], ts[i], times[i]
            fields, sizes, types_, counts, arr = read_pcd(path)
            for c in ("x", "y", "z"):
                if c not in fields:
                    sys.exit(f"{path}: field '{c}' missing (fields: {fields})")
            if arr.shape[0] == 0:
                print(f"  skipping empty cloud: {os.path.basename(path)}")
                continue

            P = np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype("f8")
            B = (P - t) @ R                       # p_body = R^T (p_world - t)
            arr["x"] = B[:, 0].astype(arr["x"].dtype)
            arr["y"] = B[:, 1].astype(arr["y"].dtype)
            arr["z"] = B[:, 2].astype(arr["z"].dtype)

            write_pcd_binary(os.path.join(a.out, f"full{n}.pcd"),
                             fields, sizes, types_, counts, arr)
            for r in range(3):
                csv.write(f"{R[r, 0]:.12g},{R[r, 1]:.12g},{R[r, 2]:.12g},{t[r]:.12g}\n")
            csv.write(f"0,0,0,{tp:.6f}\n")

            total_pts += arr.shape[0]
            if n == 0 or n == len(matched) - 1:
                rng = np.linalg.norm(B, axis=1)
                print(f"  full{n}.pcd  {arr.shape[0]} pts  body-frame range "
                      f"min/med/max = {rng.min():.2f}/{np.median(rng):.2f}/{rng.max():.2f} m  "
                      f"(pose dt {dt * 1000:.1f} ms)")
            n += 1

    print(f"wrote {n} clouds ({total_pts} points total) + {csv_path}")
    print(f"BALM should now report:  The size of poses: {n}")


if __name__ == "__main__":
    main()
