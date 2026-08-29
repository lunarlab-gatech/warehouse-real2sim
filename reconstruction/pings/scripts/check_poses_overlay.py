#!/usr/bin/env python3
# Pose-alignment sanity gate: transform the first N velodyne scans by their
# poses.txt poses (world<-LiDAR) into a single world cloud and render a top-down
# image (+ an ASCII .ply). Correct poses -> crisp coherent walls/ground/curbs;
# a wrong frame/quaternion/index -> a smeared blur. Run BEFORE the long PINGS run.
# NumPy + cv2 only.
#
#   python3 check_poses_overlay.py <seq_dir> <out_prefix> [--n 40] [--res 0.05]
import argparse
import glob
import os

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("seq_dir", help="sequence dir with velodyne/ + poses.txt")
    ap.add_argument("out_prefix", help="output path prefix (writes _topdown.png + .ply)")
    ap.add_argument("--n", type=int, default=40, help="number of scans to accumulate")
    ap.add_argument("--res", type=float, default=0.05, help="top-down meters/pixel")
    a = ap.parse_args()

    poses = np.loadtxt(os.path.join(a.seq_dir, "poses.txt"), dtype=np.float64)
    poses = poses.reshape(-1, 3, 4)
    bins = sorted(glob.glob(os.path.join(a.seq_dir, "velodyne/") + "*.bin"))
    n = min(a.n, len(bins), len(poses))
    print(f"accumulating {n} scans (of {len(bins)} bins, {len(poses)} poses)")

    allp, allc = [], []
    for i in range(n):
        pts = np.fromfile(bins[i], dtype=np.float32).reshape(-1, 4)[:, :3].astype(np.float64)
        R, t = poses[i][:, :3], poses[i][:, 3]
        w = (R @ pts.T).T + t                       # world<-lidar
        allp.append(w)
        allc.append(np.full(len(w), i))             # frame index, for the .ply
    P = np.concatenate(allp); C = np.concatenate(allc)
    print(f"total points: {len(P)}   x[{P[:,0].min():.1f},{P[:,0].max():.1f}] "
          f"y[{P[:,1].min():.1f},{P[:,1].max():.1f}] z[{P[:,2].min():.1f},{P[:,2].max():.1f}]")

    # --- top-down PNG, colored by height (z) ---
    xy = P[:, :2]
    lo = np.percentile(xy, 1, axis=0); hi = np.percentile(xy, 99, axis=0)
    W = max(2, int((hi[0] - lo[0]) / a.res)); H = max(2, int((hi[1] - lo[1]) / a.res))
    W = min(W, 2000); H = min(H, 2000)
    u = np.clip(((xy[:, 0] - lo[0]) / (hi[0] - lo[0]) * (W - 1)), 0, W - 1).astype(int)
    v = np.clip(((xy[:, 1] - lo[1]) / (hi[1] - lo[1]) * (H - 1)), 0, H - 1).astype(int)
    zlo, zhi = np.percentile(P[:, 2], [2, 98])
    zc = np.clip((P[:, 2] - zlo) / max(zhi - zlo, 1e-6), 0, 1)
    img = np.zeros((H, W), np.float32); cnt = np.zeros((H, W), np.float32)
    np.add.at(img, (v, u), zc); np.add.at(cnt, (v, u), 1)
    img = np.where(cnt > 0, img / np.maximum(cnt, 1), 0)
    col = cv2.applyColorMap((img * 255).astype(np.uint8), cv2.COLORMAP_JET)
    col[cnt == 0] = 0
    out_png = a.out_prefix + "_topdown.png"
    cv2.imwrite(out_png, cv2.flip(col, 0))          # flip so +y is up
    print(f"wrote {out_png}  ({W}x{H} @ {a.res} m/px)")

    # --- ASCII .ply colored by frame index (rainbow) for CloudCompare ---
    fc = (cv2.applyColorMap(((C / max(n - 1, 1)) * 255).astype(np.uint8).reshape(-1, 1),
                            cv2.COLORMAP_HSV).reshape(-1, 3))[:, ::-1]  # ->RGB
    out_ply = a.out_prefix + ".ply"
    step = max(1, len(P) // 1_500_000)              # cap ply size
    Ps, Cs = P[::step], fc[::step]
    with open(out_ply, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(Ps)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for (x, y, z), (r, g, b) in zip(Ps, Cs):
            f.write(f"{x:.3f} {y:.3f} {z:.3f} {int(r)} {int(g)} {int(b)}\n")
    print(f"wrote {out_ply}  ({len(Ps)} pts, frame-colored)")


if __name__ == "__main__":
    main()
