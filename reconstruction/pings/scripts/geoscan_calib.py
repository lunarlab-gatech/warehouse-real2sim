#!/usr/bin/env python3
# Single source of truth for GeoScan-S1 camera calibration, shared by
# convert_geoscan_bag.py, add_camera.py, and verify_geoscan_overlay.py so the
# numbers can never drift between conversion and verification.
#
# All three cameras were calibrated with Kalibr's EQUIDISTANT (Kannala-Brandt)
# fisheye model. Extrinsics are camera <- lidar:  p_cam = RCL @ p_lidar + PCL
# (FAST-LIVO2 "multi_calib_result.txt" convention).
#
# Sources:
#   right : geoscan_cam_calibration/right_fisheye_camera/{data-camchain,multi_calib_result}
#   left  : geoscan_cam_calibration/left_fisheye_camera/{data-camchain,multi_calib_result}
#   rs    : geoscan_cam_calibration/realsense_camera/{data-camchain,multi_calib_result}
import numpy as np

try:
    import cv2
except Exception:  # cv2 only needed for the rectify helpers
    cv2 = None


# name -> calibration. out_index N => writes image_N/ (+ camN.json for N>2).
# max_hfov_deg (fisheyes only): bound the rectified pinhole to this horizontal FOV
# with a CENTERED principal point. Unbounded rectification of these ~140-degree
# fisheyes stretches 38-48% of the pixels into unusable smear and pushes the
# principal point to x=1030 of 1280 (a quarter of the image becomes black border).
CAMERAS = {
    "right": dict(
        topic="/right_camera/image/compressed", compressed=True, out_index=2,
        W=1280, H=1024, max_hfov_deg=100.0, img_time_offset=-0.1,
        fx=467.44275018622574, fy=466.805186815233,
        cx=658.829249055717,   cy=519.6802678124923,
        D=[-0.026931814656773856, 0.0037434928646581226,
           -0.000586074477876758, -0.00018162294789360667],
        RCL=[-0.008655, -0.999961, 0.001772,
              0.447172, -0.005455, -0.894432,
              0.894406, -0.006949, 0.447201],
        PCL=[-0.051469, -0.127918, -0.010792],
    ),
    "left": dict(
        topic="/left_camera/image/compressed", compressed=True, out_index=3,
        W=1280, H=1024, max_hfov_deg=100.0, img_time_offset=-0.1,
        fx=471.8452356803565, fy=471.2540487136626,
        cx=646.742959553862,  cy=457.6210192168221,
        D=[-0.034062864120001424, 0.013596626836851117,
           -0.006674771756419758, 0.0010751998833440292],
        RCL=[-0.015597, -0.999834, 0.009422,
              0.450406, -0.015438, -0.892691,
              0.892688, -0.009679, 0.450572],
        PCL=[0.047041, -0.128563, -0.008073],
    ),
    "realsense": dict(
        topic="/camera/color/image_raw", compressed=False, out_index=4,
        W=1280, H=720, img_time_offset=-0.05,
        fx=868.1137481375785, fy=867.7965691684615,
        cx=644.0291010828827, cy=369.68502498929234,
        D=[0.4240986039693088, 0.2714224784303239,
           -2.052081423157285, 3.1937347770136015],
        RCL=[-0.002887, -0.999996, -0.000450,
              0.453158, -0.000907, -0.891430,
              0.891426, -0.002777, 0.453158],
        PCL=[0.021341, -0.208215, -0.106697],
    ),
}


def K_orig(cam):
    c = CAMERAS[cam]
    return np.array([[c["fx"], 0, c["cx"]], [0, c["fy"], c["cy"]], [0, 0, 1]], float)


def D_vec(cam):
    return np.array(CAMERAS[cam]["D"], float)


def T_cam_lidar(cam):
    """4x4 homogeneous camera<-lidar transform."""
    c = CAMERAS[cam]
    T = np.eye(4)
    T[:3, :3] = np.array(c["RCL"], float).reshape(3, 3)
    T[:3, 3] = c["PCL"]
    return T


def rectify(cam, balance=0.0):
    """Return (Knew, map1, map2) to undistort the equidistant fisheye to a pinhole.

    Cameras with a max_hfov_deg entry get a BOUNDED rectification: a symmetric
    pinhole (centered principal point) whose horizontal FOV is exactly that bound.
    This keeps the rectilinear stretch factor tame across the whole image instead
    of letting cv2's estimator chase the full fisheye FOV. Cameras without the
    entry (the narrow RealSense) keep the cv2 estimate; balance=0 crops to the
    valid region (no black borders)."""
    assert cv2 is not None, "cv2 required for rectify()"
    c = CAMERAS[cam]
    K, D, size = K_orig(cam), D_vec(cam), (c["W"], c["H"])
    max_hfov = c.get("max_hfov_deg")
    if max_hfov is not None:
        f_new = (c["W"] / 2.0) / np.tan(np.radians(max_hfov) / 2.0)
        Knew = np.array([[f_new, 0.0, (c["W"] - 1) / 2.0],
                         [0.0, f_new, (c["H"] - 1) / 2.0],
                         [0.0, 0.0, 1.0]])
    else:
        Knew = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K, D, size, np.eye(3), balance=balance)
    m1, m2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), Knew, size, cv2.CV_16SC2)
    return Knew, m1, m2
