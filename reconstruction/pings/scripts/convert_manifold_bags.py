#!/usr/bin/env python3
# Convert a handheld-scanner scene (a directory of sequential BAG_*.bag parts +
# calib_online_final.yaml) into the KITTI-format sequence PINGS' geoscan
# dataloader reads. Combines convert_geoscan_bag.py --geoscan (LiDAR) and
# add_camera.py (timestamp-matched rectified cameras) for the Manifold rig,
# reading ALL bag parts of the scene as one time-ordered stream.
#
# Output (under <out> = .../sequences/00):
#   velodyne/*.bin        float32 N x4 (x,y,z,reflectivity), lidar frame
#   velodyne_ts/*.bin     per-point normalized Livox offset_time (deskew)
#   times.txt             LiDAR header stamp (s) per scan
#   calib.txt             P2=[K_rect_middle|0], Tr=[R_cam_lidar|t] (middle cam)
#   image_2/ 3/ 4/        rectified middle / left / right camera PNGs
#   times_cam{2,3,4}.txt  matched image capture time per scan (sensor_ts sync)
#   cam{3,4}.json         extra-camera rectified K + T_cam_lidar
#
#   python3 convert_manifold_bags.py <scene_dir> <out_seq_dir> \
#       [--stage all|lidar|cam] [--camera middle|left|right] [--max-frames N]
import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from rosbags.highlevel import AnyReader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import manifold_calib as mc

LIDAR_TOPIC = "/livox/lidar"
BLIND_M = 0.25  # drop the Livox near-blind zone (matches config min_range_m)


def scene_bags(scene_dir):
    bags = sorted(Path(scene_dir).glob("BAG_*.bag"))
    if not bags:
        sys.exit(f"ERROR: no BAG_*.bag in {scene_dir}")
    return bags


def hdr_time(msg):
    s = msg.header.stamp
    return float(s.sec) + float(s.nanosec) * 1e-9


def lidar_to_bin(msg, blind_m):
    # x,y,z,reflectivity + per-point offset_time (ns) normalized to [0,1]
    pts = msg.points
    n = len(pts)
    a = np.fromiter(
        (v for p in pts for v in (p.x, p.y, p.z, float(p.reflectivity), float(p.offset_time))),
        dtype=np.float64, count=n * 5).reshape(-1, 5)
    mask = np.linalg.norm(a[:, :3], axis=1) > blind_m
    xyz = a[mask, :4].astype(np.float32)
    ot = a[mask, 4]
    span = ot.max() - ot.min() if len(ot) else 0.0
    ts = ((ot - ot.min()) / span if span > 0 else np.zeros(int(mask.sum()))).astype(np.float32)
    return xyz, ts


def write_calib(path, T_middle):
    Knew, _, _ = mc.rectify("middle")
    P = f"{Knew[0,0]} 0 {Knew[0,2]} 0 0 {Knew[1,1]} {Knew[1,2]} 0 0 0 1 0"
    R, t = T_middle[:3, :3].reshape(-1), T_middle[:3, 3]
    Tr = " ".join(str(v) for v in [
        R[0], R[1], R[2], t[0], R[3], R[4], R[5], t[1], R[6], R[7], R[8], t[2]])
    with open(path, "w") as f:
        for k in ("P0", "P1", "P2", "P3"):
            f.write(f"{k}: {P}\n")
        f.write(f"Tr: {Tr}\n")


def convert_lidar(bags, out, T_ext, max_frames, blind_m):
    velo = out / "velodyne"
    velo_ts = out / "velodyne_ts"
    velo.mkdir(parents=True, exist_ok=True)
    velo_ts.mkdir(parents=True, exist_ok=True)
    write_calib(out / "calib.txt", T_ext["middle"])
    li = 0
    scan_times = []
    with AnyReader(bags) as r:
        cons = [c for c in r.connections if c.topic == LIDAR_TOPIC]
        if not cons:
            sys.exit(f"ERROR: {LIDAR_TOPIC} not in bags")
        for con, ts, raw in r.messages(connections=cons):
            if max_frames and li >= max_frames:
                break
            m = r.deserialize(raw, con.msgtype)
            xyz, pts_ts = lidar_to_bin(m, blind_m)
            xyz.tofile(str(velo / f"{li:06d}.bin"))
            pts_ts.tofile(str(velo_ts / f"{li:06d}.bin"))
            scan_times.append(hdr_time(m))
            li += 1
            if li % 500 == 0:
                print(f"  ... {li} scans", flush=True)
    np.savetxt(out / "times.txt", np.asarray(scan_times, np.float64), fmt="%.9f")
    print(f"LIDAR_DONE: {li} scans + per-point ts + times.txt + calib.txt -> {out}", flush=True)


def convert_camera(bags, out, cam, T_ext, max_dt=0.06):
    c = mc.CAMERAS[cam]
    offset = c["img_time_offset"]
    times_file = out / "times.txt"
    if not times_file.exists():
        sys.exit(f"ERROR: {times_file} missing (run --stage lidar first)")
    scan_t = np.loadtxt(times_file, dtype=np.float64).reshape(-1)
    n = len(scan_t)
    # image whose CAPTURE time (= header + offset) is nearest each scan
    target_t = scan_t - offset

    imgN = out / f"image_{c['out_index']}"
    imgN.mkdir(parents=True, exist_ok=True)
    Knew, m1, m2 = mc.rectify(cam)
    if c["out_index"] > 2:
        import json
        camjson = {"K": Knew.tolist(), "T_cam_lidar": T_ext[cam].tolist(),
                   "width": c["W"], "height": c["H"], "dir": imgN.name}
        with open(out / f"cam{c['out_index']}.json", "w") as f:
            json.dump(camjson, f, indent=2)

    with AnyReader(bags) as r:
        cons = [con for con in r.connections if con.topic == c["topic"]]
        if not cons:
            sys.exit(f"ERROR: topic {c['topic']} not found in bags")

        # pass 1: camera header stamps (stop once past the last scan time)
        t_stop = float(target_t.max()) + 0.5
        cam_t = []
        for con, ts, raw in r.messages(connections=cons):
            tt = hdr_time(r.deserialize(raw, con.msgtype))
            cam_t.append(tt)
            if tt > t_stop:
                break
        cam_t = np.asarray(cam_t, np.float64)
        print(f"[{cam}] {len(cam_t)} images on {c['topic']}, matching to {n} scans "
              f"(offset {offset:+.3f}s)", flush=True)

        match = np.abs(cam_t[None, :] - target_t[:, None]).argmin(axis=1)
        dt = np.abs(cam_t[match] - target_t)
        bad = int((dt > max_dt).sum())
        print(f"[{cam}] match dt: median {np.median(dt)*1e3:.1f}ms  max {dt.max()*1e3:.1f}ms"
              f"  ({bad}/{n} over {max_dt*1e3:.0f}ms)", flush=True)
        np.savetxt(out / f"times_cam{c['out_index']}.txt", cam_t[match] + offset, fmt="%.9f")

        wanted = {}  # cam_msg_index -> [output frame indices]
        for fi, ci in enumerate(match):
            wanted.setdefault(int(ci), []).append(fi)

        # pass 2: decode only matched messages, rectify, write
        last_ci = max(wanted) if wanted else -1
        written = 0
        for ci, (con, ts, raw) in enumerate(r.messages(connections=cons)):
            if ci > last_ci:
                break
            outs = wanted.get(ci)
            if not outs:
                continue
            m = r.deserialize(raw, con.msgtype)
            bgr = cv2.imdecode(np.frombuffer(m.data, np.uint8), cv2.IMREAD_COLOR)
            und = cv2.remap(bgr, m1, m2, interpolation=cv2.INTER_LINEAR)
            for fi in outs:
                cv2.imwrite(str(imgN / f"{fi:06d}.png"), und)
                written += 1
            if written % 500 < len(outs):
                print(f"  [{cam}] ... {written}/{n}", flush=True)
    print(f"CAM_DONE [{cam}]: {written} rectified images -> {imgN}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene_dir", help="dir with BAG_*.bag parts + calib_online_final.yaml")
    ap.add_argument("out", help="output sequence dir, e.g. ./data/geoscan_warehouse/sequences/00")
    ap.add_argument("--stage", choices=["all", "lidar", "cam"], default="all")
    ap.add_argument("--camera", choices=list(mc.CAMERAS.keys()),
                    help="with --stage cam: convert only this camera")
    ap.add_argument("--max-frames", type=int, default=0, help="0 = all")
    ap.add_argument("--blind-m", type=float, default=BLIND_M)
    a = ap.parse_args()

    bags = scene_bags(a.scene_dir)
    calib_yaml = Path(a.scene_dir) / "calib_online_final.yaml"
    if not calib_yaml.exists():
        sys.exit(f"ERROR: {calib_yaml} missing")
    T_ext = mc.load_extrinsics(calib_yaml)
    out = Path(a.out)
    print(f"scene: {a.scene_dir} ({len(bags)} bags) -> {out}", flush=True)

    if a.stage in ("all", "lidar"):
        convert_lidar(bags, out, T_ext, a.max_frames, a.blind_m)
    if a.stage == "cam":
        cams = [a.camera] if a.camera else list(mc.CAMERAS.keys())
        for cam in cams:
            convert_camera(bags, out, cam, T_ext)
    elif a.stage == "all":
        for cam in mc.CAMERAS.keys():
            convert_camera(bags, out, cam, T_ext)


if __name__ == "__main__":
    main()
