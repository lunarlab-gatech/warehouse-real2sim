#!/usr/bin/env python3
# Convert a GeoScan ROS bag (Livox Mid-360 + cameras) into a KITTI-format
# sequence directory for PINGS' `geoscan` dataloader.
#
# Two modes:
#   (default, labrandom_undist) LiDAR + the already-rectified right camera:
#       velodyne/*.bin  +  image_2/*.png  +  calib.txt
#   (--harrison) LiDAR ONLY + times.txt + calib.txt (rectified-right K from the
#       shared calib registry). The cameras (incl. the right fisheye, which is
#       RAW/compressed here) are added separately and timestamp-matched by
#       add_camera.py. This keeps a free-running RealSense correctly aligned.
#
# Output (under <out> = .../sequences/00):
#   velodyne/000000.bin   float32 N x4 (x,y,z,reflectivity), lidar frame
#   times.txt             one float-seconds LiDAR header stamp per scan (harrison)
#   image_2/000000.png    right camera (default mode only)
#   calib.txt             P2 = [K|0],  Tr = [R_cam_lidar | t_cam_lidar]
#
#   python3 convert_geoscan_bag.py <bag> <out_dir> [--geoscan] [--max-frames N]
import argparse
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from rosbags.highlevel import AnyReader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import geoscan_calib as gc

# --- default-mode calibration (labrandom: pre-rectified right camera) ----------
# p_cam = RCL @ p_lidar + PCL   (Fast-LIVO2 config/mid360.yaml, camera <- lidar)
RCL = [-0.008655, -0.999961, 0.001772,
        0.447172, -0.005455, -0.894432,
        0.894406, -0.006949, 0.447201]
PCL = [-0.051469, -0.127918, -0.010792]
# camera_pinhole_mid360.yaml (rectified right camera, zero distortion)
FX, FY, CX, CY = 280.46565011173544, 280.0831120891398, 639.5, 511.5

LIDAR_TOPIC = "/livox/lidar"
IMAGE_TOPIC = "/right_camera/image"
# Drop the Livox near-blind zone / (0,0,0) dummy points. The Mid-360 is reliable
# down to ~0.1 m; 0.25 keeps desk-distance returns that 0.5 used to discard.
# Overridable with --blind-m (set from main()).
BLIND_M = 0.25


def _calib_str(K, RCLv, PCLv):
    P = f"{K[0,0]} 0 {K[0,2]} 0 0 {K[1,1]} {K[1,2]} 0 0 0 1 0"
    Tr = " ".join(str(v) for v in [
        RCLv[0], RCLv[1], RCLv[2], PCLv[0],
        RCLv[3], RCLv[4], RCLv[5], PCLv[1],
        RCLv[6], RCLv[7], RCLv[8], PCLv[2]])
    return P, Tr


def write_calib(path, harrison):
    if harrison:  # rectified right-fisheye K from the registry + right extrinsic
        Knew, _, _ = gc.rectify("right")
        c = gc.CAMERAS["right"]
        P, Tr = _calib_str(Knew, c["RCL"], c["PCL"])
    else:
        K = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]])
        P, Tr = _calib_str(K, RCL, PCL)
    with open(path, "w") as f:
        for k in ("P0", "P1", "P2", "P3"):
            f.write(f"{k}: {P}\n")
        f.write(f"Tr: {Tr}\n")


def _lidar_to_bin(msg, with_ts=False):
    pts = msg.points
    n = len(pts)
    if not with_ts:
        arr = np.fromiter(
            (v for p in pts for v in (p.x, p.y, p.z, float(p.reflectivity))),
            dtype=np.float32, count=n * 4).reshape(-1, 4)
        d = np.linalg.norm(arr[:, :3], axis=1)
        return arr[d > BLIND_M]
    # also pull per-point offset_time (ns within the scan) for deskew; one pass,
    # float64 so the ~1e8 ns values keep precision before normalizing to [0,1].
    a = np.fromiter(
        (v for p in pts for v in (p.x, p.y, p.z, float(p.reflectivity), float(p.offset_time))),
        dtype=np.float64, count=n * 5).reshape(-1, 5)
    mask = np.linalg.norm(a[:, :3], axis=1) > BLIND_M
    xyz = a[mask, :4].astype(np.float32)
    ot = a[mask, 4]
    span = ot.max() - ot.min()
    ts = ((ot - ot.min()) / span if span > 0 else np.zeros(int(mask.sum()))).astype(np.float32)
    return xyz, ts


def _hdr_time(msg):
    s = msg.header.stamp
    return float(s.sec) + float(s.nanosec) * 1e-9


def main():
    global BLIND_M
    ap = argparse.ArgumentParser()
    ap.add_argument("bag")
    ap.add_argument("out", help="output sequence dir, e.g. .../sequences/00")
    ap.add_argument("--geoscan", "--harrison", dest="geoscan", action="store_true",
                    help="GeoScan-rig (Livox CustomMsg): LiDAR-only + times.txt; cameras via add_camera.py")
    ap.add_argument("--max-frames", type=int, default=0, help="0 = all")
    ap.add_argument("--blind-m", type=float, default=BLIND_M,
                    help="drop LiDAR returns closer than this (m)")
    a = ap.parse_args()
    BLIND_M = a.blind_m

    out = Path(a.out)
    velo = out / "velodyne"
    velo.mkdir(parents=True, exist_ok=True)
    write_calib(out / "calib.txt", a.geoscan)

    if a.geoscan:
        velo_ts = out / "velodyne_ts"  # per-point normalized offset_time, for deskew
        velo_ts.mkdir(parents=True, exist_ok=True)
        li = 0
        scan_times = []
        with AnyReader([Path(a.bag)]) as r:
            cons = [c for c in r.connections if c.topic == LIDAR_TOPIC]
            if not cons:
                sys.exit(f"ERROR: {LIDAR_TOPIC} not in bag")
            for con, ts, raw in r.messages(connections=cons):
                if a.max_frames and li >= a.max_frames:
                    break
                m = r.deserialize(raw, con.msgtype)
                xyz, pts_ts = _lidar_to_bin(m, with_ts=True)
                xyz.tofile(str(velo / f"{li:06d}.bin"))
                pts_ts.tofile(str(velo_ts / f"{li:06d}.bin"))
                scan_times.append(_hdr_time(m))
                li += 1
                if li % 200 == 0:
                    print(f"  ... {li} scans", flush=True)
        np.savetxt(out / "times.txt", np.asarray(scan_times, np.float64), fmt="%.9f")
        print(f"DONE (harrison): {li} scans + per-point ts + times.txt + calib.txt -> {out}", flush=True)
        return

    # --- default mode: LiDAR + pre-rectified right camera, positional pairing ---
    imgd = out / "image_2"
    imgd.mkdir(parents=True, exist_ok=True)
    li = ii = 0
    with AnyReader([Path(a.bag)]) as r:
        cons = [c for c in r.connections if c.topic in (LIDAR_TOPIC, IMAGE_TOPIC)]
        for con, ts, raw in r.messages(connections=cons):
            if a.max_frames and li >= a.max_frames and ii >= a.max_frames:
                break
            if con.topic == LIDAR_TOPIC:
                if a.max_frames and li >= a.max_frames:
                    continue
                _lidar_to_bin(r.deserialize(raw, con.msgtype)).tofile(str(velo / f"{li:06d}.bin"))
                li += 1
            elif con.topic == IMAGE_TOPIC:
                if a.max_frames and ii >= a.max_frames:
                    continue
                m = r.deserialize(raw, con.msgtype)
                bgr = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, 3)
                Image.fromarray(bgr[:, :, ::-1]).save(str(imgd / f"{ii:06d}.png"))
                ii += 1
            if (li + ii) % 200 == 0:
                print(f"  ... {li} scans, {ii} images", flush=True)

    n = min(li, ii)
    for i in range(n, li):
        os.remove(velo / f"{i:06d}.bin")
    for i in range(n, ii):
        os.remove(imgd / f"{i:06d}.png")
    print(f"DONE: {li} scans, {ii} images -> {n} usable frames at {out}", flush=True)


if __name__ == "__main__":
    main()
