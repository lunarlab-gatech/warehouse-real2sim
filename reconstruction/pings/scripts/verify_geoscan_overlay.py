#!/usr/bin/env python3
# Local de-risk BEFORE a multi-hour GPU run: project a LiDAR scan onto each
# rectified GeoScan camera (right / left / realsense) and save overlays so the
# extrinsics + equidistant rectification can be eyeballed. If projected LiDAR
# points land on the matching image structures, the calibration is good.
#
# Also emits, for the RealSense, a "raw" overlay (project with the fisheye model
# onto the un-rectified image) to tell whether its large equidistant coeffs are
# correct or whether the SDK already rectified that stream.
#
#   python3 verify_geoscan_overlay.py <bag> [out_dir]      (env SKIP=seconds)
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from rosbags.highlevel import AnyReader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import geoscan_calib as gc
from add_camera import decode_bgr

BAG = sys.argv[1]
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "/tmp/geoscan_verify")
OUT.mkdir(parents=True, exist_ok=True)
SKIP = float(os.environ.get("SKIP", "5.0"))   # seconds into the bag to sample
NSCAN = 8
LIDAR = "/livox/lidar"
TOPIC2CAM = {gc.CAMERAS[c]["topic"]: c for c in gc.CAMERAS}


def hdr(m):
    s = m.header.stamp
    return float(s.sec) + float(s.nanosec) * 1e-9


def draw_points(img, uv, depth, r=2):
    out = img.copy()
    dn = np.clip((depth - 0.5) / (15.0 - 0.5), 0, 1)
    col = cv2.applyColorMap((dn * 255).astype(np.uint8).reshape(-1, 1), cv2.COLORMAP_JET).reshape(-1, 3)
    for (u, v), c in zip(uv, col):
        cv2.circle(out, (int(u), int(v)), r, (int(c[0]), int(c[1]), int(c[2])), -1)
    return out


def overlay_pinhole(pts, K, T, img):
    """Project lidar pts (Nx3) into a RECTIFIED pinhole image."""
    pc = (T @ np.c_[pts, np.ones(len(pts))].T).T[:, :3]
    z = pc[:, 2]
    front = z > 0.3
    pc, z = pc[front], z[front]
    uv = (K @ pc.T).T
    uv = uv[:, :2] / uv[:, 2:3]
    H, W = img.shape[:2]
    m = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H) & (z < 40)
    return draw_points(img, uv[m], z[m]), int(m.sum())


def overlay_fisheye_raw(pts, cam, img):
    """Project lidar pts onto the RAW (distorted) image via the equidistant model."""
    c = gc.CAMERAS[cam]
    T = gc.T_cam_lidar(cam)
    pc = (T @ np.c_[pts, np.ones(len(pts))].T).T[:, :3]
    z = pc[:, 2]
    front = z > 0.3
    pc, z = pc[front], z[front]
    uv, _ = cv2.fisheye.projectPoints(pc.reshape(-1, 1, 3), np.zeros(3), np.zeros(3),
                                      gc.K_orig(cam), gc.D_vec(cam))
    uv = uv.reshape(-1, 2)
    H, W = img.shape[:2]
    m = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H) & (z < 40)
    return draw_points(img, uv[m], z[m]), int(m.sum())


def main():
    scans = []                       # (t, Nx3)
    cams = {c: [] for c in gc.CAMERAS}  # name -> [(t, bgr)]
    t0 = None
    want = [LIDAR] + list(TOPIC2CAM)
    with AnyReader([Path(BAG)]) as r:
        cons = [con for con in r.connections if con.topic in want]
        for con, ts, raw in r.messages(connections=cons):
            m = r.deserialize(raw, con.msgtype)
            t = hdr(m)
            if t0 is None:
                t0 = t
            if t < t0 + SKIP:
                continue
            if t > t0 + SKIP + NSCAN * 0.12 + 0.5:
                break
            if con.topic == LIDAR:
                if len(scans) < NSCAN:
                    pts = np.array([(p.x, p.y, p.z) for p in m.points], float)
                    d = np.linalg.norm(pts, axis=1)
                    scans.append((t, pts[d > 0.5]))
            else:
                name = TOPIC2CAM[con.topic]
                cams[name].append((t, decode_bgr(m, gc.CAMERAS[name]["compressed"])))

    if not scans:
        sys.exit("no lidar scans collected; lower SKIP")
    ts_s, pts = scans[len(scans) // 2]   # a middle scan in the window
    print(f"sample scan t={ts_s:.3f}  ({len(pts)} pts)   cam frames: "
          + ", ".join(f"{k}={len(v)}" for k, v in cams.items()))

    for cam in gc.CAMERAS:
        if not cams[cam]:
            print(f"[{cam}] NO camera frames in window")
            continue
        cts = np.array([t for t, _ in cams[cam]])
        for off in (0.0, 0.1):
            j = int(np.abs(cts - (ts_s + off)).argmin())
            t_c, bgr = cams[cam][j]
            Knew, m1, m2 = gc.rectify(cam)
            rect = cv2.remap(bgr, m1, m2, interpolation=cv2.INTER_LINEAR)
            ov, npix = overlay_pinhole(pts, Knew, gc.T_cam_lidar(cam), rect)
            tag = f"{cam}_off{off:.2f}_dt{(t_c-ts_s)*1e3:+.0f}ms_n{npix}"
            cv2.imwrite(str(OUT / f"{tag}.png"), ov)
            if off == 0.0:
                cv2.imwrite(str(OUT / f"{cam}_rect.png"), rect)
            print(f"[{cam}] off={off}: matched dt={(t_c-ts_s)*1e3:+.0f}ms, {npix} pts on image -> {tag}.png")
        # RealSense: also test the raw (un-rectified) projection
        if cam == "realsense":
            t_c, bgr = cams[cam][int(np.abs(cts - ts_s).argmin())]
            ovr, npr = overlay_fisheye_raw(pts, cam, bgr)
            cv2.imwrite(str(OUT / "realsense_RAW_overlay.png"), ovr)
            cv2.imwrite(str(OUT / "realsense_RAW.png"), bgr)
            print(f"[realsense] RAW fisheye-projection overlay: {npr} pts -> realsense_RAW_overlay.png")

    print(f"\nwrote overlays to {OUT}")


if __name__ == "__main__":
    main()
