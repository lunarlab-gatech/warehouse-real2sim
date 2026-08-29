#!/usr/bin/env python3
# Single source of truth for the handheld scanner rig (Livox Mid-360 + three
# 10 Hz fisheye cameras), shared by convert_manifold_bags.py. Mirrors
# geoscan_calib.py so the conversion follows the exact same conventions.
#
# Intrinsics: extrinsic_intrinsic/mid360_cam.yaml (EquidistantCamera =
# Kannala-Brandt, cv2.fisheye, 1600x1296, active uncommented block).
# Extrinsics: per-scene calib_online_final.yaml next to the bags -- the
# online-refined camera<-lidar transforms (p_cam = Tcl @ p_lidar, FAST-LIVO2
# convention). Tcl_N indices follow mid360.yaml's cam_id -> topic mapping:
#   Tcl_0 = /left_camera, Tcl_1 = /middle_camera, Tcl_2 = /right_camera.
#
# img_time_offset is 0.0: the rig's own mid360.yaml declares
# time_offset.img_time_offset: 0.00 (unlike the GeoScan-S1 drivers).
import re

import numpy as np

try:
    import cv2
except Exception:  # cv2 only needed for the rectify helper
    cv2 = None


# name -> calibration. out_index N => writes image_N/ (+ camN.json for N>2).
# middle is the forward camera -> main cam (image_2, calib.txt). max_hfov_deg
# bounds the rectified pinhole to a centered-principal-point 100-degree HFOV,
# the same tame rectification the GeoScan fisheyes use (native HFOV ~125 deg).
CAMERAS = {
    "middle": dict(
        topic="/middle_camera/image/compressed", compressed=True, out_index=2,
        tcl_key="Tcl_1", W=1600, H=1296, max_hfov_deg=100.0, img_time_offset=0.0,
        fx=733.789123, fy=733.939516, cx=840.876695, cy=687.524124,
        D=[-0.016567, 0.000967, -0.004411, 0.001281],
    ),
    "left": dict(
        topic="/left_camera/image/compressed", compressed=True, out_index=3,
        tcl_key="Tcl_0", W=1600, H=1296, max_hfov_deg=100.0, img_time_offset=0.0,
        fx=732.459529, fy=732.519632, cx=756.096002, cy=589.584926,
        D=[-0.014926, -0.002410, -0.000832, 0.000032],
    ),
    "right": dict(
        topic="/right_camera/image/compressed", compressed=True, out_index=4,
        tcl_key="Tcl_2", W=1600, H=1296, max_hfov_deg=100.0, img_time_offset=0.0,
        fx=732.337365, fy=732.375844, cx=736.903159, cy=618.467428,
        D=[-0.015897, -0.000977, -0.002257, 0.000415],
    ),
}


def K_orig(cam):
    c = CAMERAS[cam]
    return np.array([[c["fx"], 0, c["cx"]], [0, c["fy"], c["cy"]], [0, 0, 1]], float)


def D_vec(cam):
    return np.array(CAMERAS[cam]["D"], float)


def load_extrinsics(calib_final_yaml):
    """Parse a scene's calib_online_final.yaml -> {cam_name: 4x4 T_cam_lidar}.

    The file is a tiny hand-rolled YAML: `Tcl_N: [16 row-major floats]` blocks
    (plus Til, imu<-lidar, unused here). Validates each rotation block."""
    text = open(calib_final_yaml).read()
    mats = {}
    for m in re.finditer(r"(Tcl_\d|Til):\s*\[([^\]]+)\]", text):
        vals = np.array([float(v) for v in m.group(2).replace(",", " ").split()])
        if vals.size != 16:
            raise ValueError(f"{m.group(1)} in {calib_final_yaml} has {vals.size} values, want 16")
        mats[m.group(1)] = vals.reshape(4, 4)
    out = {}
    for name, c in CAMERAS.items():
        key = c["tcl_key"]
        if key not in mats:
            raise ValueError(f"{key} missing from {calib_final_yaml}")
        T = mats[key]
        R = T[:3, :3]
        if not (np.allclose(R @ R.T, np.eye(3), atol=1e-3) and abs(np.linalg.det(R) - 1) < 1e-3):
            raise ValueError(f"{key} rotation block is not a valid rotation")
        out[name] = T
    return out


def rectify(cam):
    """Return (Knew, map1, map2): equidistant fisheye -> bounded pinhole.

    Same policy as geoscan_calib.rectify for the fisheyes: a symmetric pinhole
    with CENTERED principal point whose horizontal FOV is max_hfov_deg."""
    assert cv2 is not None, "cv2 required for rectify()"
    c = CAMERAS[cam]
    K, D, size = K_orig(cam), D_vec(cam), (c["W"], c["H"])
    f_new = (c["W"] / 2.0) / np.tan(np.radians(c["max_hfov_deg"]) / 2.0)
    Knew = np.array([[f_new, 0.0, (c["W"] - 1) / 2.0],
                     [0.0, f_new, (c["H"] - 1) / 2.0],
                     [0.0, 0.0, 1.0]])
    m1, m2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), Knew, size, cv2.CV_16SC2)
    return Knew, m1, m2
