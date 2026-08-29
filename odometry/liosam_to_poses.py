#!/usr/bin/env python3
# Densify a LIO-SAM trajectory to per-scan poses for PINGS' pose-fixed mapping.
#
# Inputs:
#   transformations.pcd   loop-CORRECTED keyframe poses from the save_map service
#                         (PCL PointTypePose: x,y,z,intensity,roll,pitch,yaw + double
#                         time; R = Rz(yaw)@Ry(pitch)@Rx(roll), pcl::getTransformation)
#   odometry bag          recorded /lio_sam/mapping/odometry_incremental (nav_msgs/
#                         Odometry) — jump-free relative motion (the plain mapping/
#                         odometry stream JUMPS at loop closures; do not use it here)
#   times.txt             the PINGS sequence's per-scan stamps
#
# Method: DOUBLE-ANCHORED transfer. For scan stamp t between corrected keyframes
# i,i+1: transfer the incremental-odometry relative motion onto both anchors,
#   A = T_kf_i  @ inv(T_incr(t_kf_i))  @ T_incr(t)
#   B = T_kf_i1 @ inv(T_incr(t_kf_i1)) @ T_incr(t)
# and blend slerp/lerp by u=(t-t_i)/(t_i1-t_i). Exact at keyframes, continuous,
# immune to loop-closure jumps, keeps gait-frequency motion that pure keyframe
# interpolation would smear. One-sided anchoring outside the keyframe span.
#
#   python3 liosam_to_poses.py <transformations.pcd> <odom_bag_dir> <times.txt> <out_poses.txt>
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def read_pcd_typepose(path):
    """Parse a PCL PCD (ascii or binary) with PointTypePose fields."""
    with open(path, "rb") as f:
        header = {}
        while True:
            line = f.readline().decode("ascii", "replace").strip()
            key = line.split(" ")[0].upper()
            header[key] = line.split(" ")[1:]
            if key == "DATA":
                data_mode = header["DATA"][0]
                break
        names = header["FIELDS"]
        sizes = [int(s) for s in header["SIZE"]]
        types = header["TYPE"]
        counts = [int(c) for c in header.get("COUNT", ["1"] * len(names))]
        n = int(header["POINTS"][0])
        np_map = {("F", 4): "f4", ("F", 8): "f8", ("U", 1): "u1", ("U", 4): "u4",
                  ("I", 1): "i1", ("I", 4): "i4"}
        dtype = np.dtype([(nm, np_map[(tp, sz)], ct) if ct > 1 else (nm, np_map[(tp, sz)])
                          for nm, sz, tp, ct in zip(names, sizes, types, counts)])
        if data_mode == "binary":
            arr = np.frombuffer(f.read(dtype.itemsize * n), dtype=dtype, count=n)
        elif data_mode == "ascii":
            rows = np.loadtxt(f, dtype=np.float64, max_rows=n).reshape(n, -1)
            arr = np.zeros(n, dtype=dtype)
            col = 0
            for nm, ct in zip(names, counts):
                arr[nm] = rows[:, col] if ct == 1 else rows[:, col:col + ct]
                col += ct
        else:
            sys.exit(f"ERROR: unsupported PCD DATA mode {data_mode}")
    return arr, names


def pose_mats(t, xyz, quat):
    """N stamps + positions + quats (xyzw) -> N x 4 x 4."""
    T = np.tile(np.eye(4), (len(t), 1, 1))
    T[:, :3, :3] = Rotation.from_quat(quat).as_matrix()
    T[:, :3, 3] = xyz
    return T


def interp_pose(t_arr, T_arr, t):
    """slerp/lerp a pose stream at time t; constant-velocity extrapolation at ends."""
    if t <= t_arr[0]:
        return T_arr[0]
    if t >= t_arr[-1]:
        if len(t_arr) >= 2:
            dt = t_arr[-1] - t_arr[-2]
            if dt > 0 and (t - t_arr[-1]) < 1.0:
                delta = np.linalg.inv(T_arr[-2]) @ T_arr[-1]
                frac = (t - t_arr[-1]) / dt
                dR = Rotation.from_matrix(delta[:3, :3]).as_rotvec() * frac
                out = T_arr[-1].copy()
                out[:3, :3] = T_arr[-1][:3, :3] @ Rotation.from_rotvec(dR).as_matrix()
                out[:3, 3] = T_arr[-1][:3, 3] + T_arr[-1][:3, :3] @ (delta[:3, 3] * frac)
                return out
        return T_arr[-1]
    i = np.searchsorted(t_arr, t) - 1
    u = (t - t_arr[i]) / (t_arr[i + 1] - t_arr[i])
    return blend(T_arr[i], T_arr[i + 1], u)


def blend(TA, TB, u):
    out = np.eye(4)
    rots = Rotation.from_matrix(np.stack([TA[:3, :3], TB[:3, :3]]))
    out[:3, :3] = Slerp([0.0, 1.0], rots)([u]).as_matrix()[0]
    out[:3, 3] = (1 - u) * TA[:3, 3] + u * TB[:3, 3]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kf_pcd", help="transformations.pcd (corrected keyframes)")
    ap.add_argument("odom_bag", help="recorded ros2 bag dir with /lio_sam/mapping/odometry_incremental")
    ap.add_argument("times", help="sequence times.txt")
    ap.add_argument("out", help="output poses.txt (KITTI 3x4, world<-lidar)")
    a = ap.parse_args()

    # --- corrected keyframes -------------------------------------------------
    kf, fields = read_pcd_typepose(a.kf_pcd)
    if "time" not in fields:
        sys.exit(f"ERROR: no time field in {a.kf_pcd} (fields: {fields})")
    order = np.argsort(kf["time"])
    kf = kf[order]
    t_kf = np.asarray(kf["time"], np.float64)
    R_kf = Rotation.from_euler("ZYX", np.stack([kf["yaw"], kf["pitch"], kf["roll"]], 1))
    T_kf = np.tile(np.eye(4), (len(kf), 1, 1))
    T_kf[:, :3, :3] = R_kf.as_matrix()
    T_kf[:, :3, 3] = np.stack([kf["x"], kf["y"], kf["z"]], 1)
    print(f"keyframes: {len(kf)}  span [{t_kf[0]:.3f}, {t_kf[-1]:.3f}] ({t_kf[-1]-t_kf[0]:.1f}s)")

    # --- incremental odometry stream ----------------------------------------
    from rosbags.highlevel import AnyReader
    from rosbags.typesys import Stores, get_typestore
    t_od, xyz_od, q_od = [], [], []
    # bags recorded by Humble's rosbag2 carry no embedded type definitions
    with AnyReader([Path(a.odom_bag)],
                   default_typestore=get_typestore(Stores.ROS2_HUMBLE)) as r:
        cons = [c for c in r.connections if c.topic == "/lio_sam/mapping/odometry_incremental"]
        if not cons:
            sys.exit("ERROR: /lio_sam/mapping/odometry_incremental not in recorded bag")
        for con, ts, raw in r.messages(connections=cons):
            m = r.deserialize(raw, con.msgtype)
            t_od.append(m.header.stamp.sec + m.header.stamp.nanosec * 1e-9)
            p, q = m.pose.pose.position, m.pose.pose.orientation
            xyz_od.append([p.x, p.y, p.z])
            q_od.append([q.x, q.y, q.z, q.w])
    t_od = np.asarray(t_od)
    order = np.argsort(t_od)
    t_od = t_od[order]
    T_od = pose_mats(t_od, np.asarray(xyz_od)[order], np.asarray(q_od)[order])
    print(f"incremental odometry: {len(t_od)} samples, median dt {np.median(np.diff(t_od))*1e3:.0f}ms")

    scan_t = np.loadtxt(a.times, np.float64).reshape(-1)
    n = len(scan_t)

    # incremental-odometry pose evaluated at each keyframe stamp (cache)
    T_od_at_kf = np.stack([interp_pose(t_od, T_od, t) for t in t_kf])
    Tinv_od_at_kf = np.linalg.inv(T_od_at_kf)

    rows = np.zeros((n, 12))
    for si, t in enumerate(scan_t):
        T_inc = interp_pose(t_od, T_od, t)
        if t <= t_kf[0]:
            T = T_kf[0] @ (Tinv_od_at_kf[0] @ T_inc)
        elif t >= t_kf[-1]:
            T = T_kf[-1] @ (Tinv_od_at_kf[-1] @ T_inc)
        else:
            i = np.searchsorted(t_kf, t) - 1
            A = T_kf[i] @ (Tinv_od_at_kf[i] @ T_inc)
            B = T_kf[i + 1] @ (Tinv_od_at_kf[i + 1] @ T_inc)
            T = blend(A, B, (t - t_kf[i]) / (t_kf[i + 1] - t_kf[i]))
        rows[si] = T[:3, :4].reshape(-1)

    # --- validation ----------------------------------------------------------
    tr = rows[:, [3, 7, 11]]
    step = np.linalg.norm(np.diff(tr, axis=0), axis=1)
    dtv = np.diff(scan_t)
    v = step / np.clip(dtv, 1e-6, None)
    print(f"inter-scan speed: median {np.median(v):.2f} m/s  p99 {np.percentile(v,99):.2f}  max {v.max():.2f}")
    if v.max() > 3.0:
        print("WARNING: speed bound exceeded (>3 m/s) — inspect densification")
    # anchor exactness: densified pose at each keyframe stamp vs the keyframe
    kf_in = (t_kf >= scan_t[0]) & (t_kf <= scan_t[-1])
    errs = []
    for tk, Tk in zip(t_kf[kf_in], T_kf[kf_in]):
        si = np.abs(scan_t - tk).argmin()
        if abs(scan_t[si] - tk) < 0.02:
            errs.append(np.linalg.norm(rows[si][[3, 7, 11]] - Tk[:3, 3]))
    if errs:
        print(f"anchor consistency at {len(errs)} keyframe-coincident scans: "
              f"median {np.median(errs)*100:.1f}cm  max {np.max(errs)*100:.1f}cm")
    np.savetxt(a.out, rows, fmt="%.9f")
    print(f"DONE: wrote {n} poses (KITTI 3x4, world<-lidar) -> {a.out}")


if __name__ == "__main__":
    main()
