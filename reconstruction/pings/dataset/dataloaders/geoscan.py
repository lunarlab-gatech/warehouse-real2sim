# GeoScan (Livox Mid-360 + cameras) dataloader for PINGS.
#
# Reads a KITTI-format sequence produced by scripts/convert_geoscan_bag.py and
# scripts/add_camera.py:
#   <data_dir>/sequences/<seq>/velodyne/*.bin      float32 N x4 (x,y,z,intensity)
#   <data_dir>/sequences/<seq>/velodyne_ts/*.bin   per-point normalized Livox offset_time (deskew)
#   <data_dir>/sequences/<seq>/times.txt           one LiDAR header stamp (s) per scan
#   <data_dir>/sequences/<seq>/image_2/*.png       right camera (rectified pinhole)
#   <data_dir>/sequences/<seq>/calib.txt           P2=[K_right|0], Tr=[R_cam_lidar|t]
#   <data_dir>/sequences/<seq>/image_N/*.png       extra cameras (N>=3), rectified
#   <data_dir>/sequences/<seq>/camN.json           extra-camera K + LiDAR->cam extrinsic
#   <data_dir>/sequences/<seq>/times_camN.txt      per-frame matched camera stamp (s), optional;
#                                                  enables PINGS' per-camera time-sync correction
#   <data_dir>/sequences/<seq>/poses.txt           optional world<-lidar poses (eval reference in
#                                                  tracker mode; mapping poses in pose-fixed mode)
#
# Mirrors kitti.py (colorize LiDAR points in the loader, provide per-camera depth
# maps), but: reads image size from the actual image, supports MULTIPLE cameras,
# uses real Livox per-point timestamps, and does not invert poses through Tr.
import glob
import json
import os

import cv2
import numpy as np
import open3d as o3d


class GeoScanDataset:
    def __init__(self, data_dir, sequence, *_, **__):

        self.contains_image = True
        self.sequence_id = str(sequence).zfill(2)
        self.seq_dir = os.path.join(data_dir, "sequences", self.sequence_id)

        self.velodyne_dir = os.path.join(self.seq_dir, "velodyne/")
        self.scan_files = sorted(glob.glob(self.velodyne_dir + "*.bin"))
        scan_count = len(self.scan_files)
        if scan_count == 0:
            raise ValueError(f"No .bin scans found in {self.velodyne_dir}")

        # per-point normalized timestamps for deskew (velodyne_ts/*.bin, written by
        # convert_geoscan_bag.py --geoscan). Present => real Livox offset_time;
        # absent => fall back to the (wrong-for-Livox) azimuth guess + no deskew.
        self.ts_dir = os.path.join(self.seq_dir, "velodyne_ts/")
        self.ts_files = sorted(glob.glob(self.ts_dir + "*.bin"))
        self.has_real_ts = len(self.ts_files) == scan_count

        self.sem_available = False
        self.load_img = False  # set True by SLAMDataset when use_image
        self.mono_depth_for_high_z = False  # expected by mapper; monodepth is deprecated

        # The Mid-360 sees 360 degrees but the cameras cover only part of it, so we
        # map ALL LiDAR points (uncolorized points are simply gray in the radiance
        # field). This is the one deliberate deviation from kitti.py's colorized-only
        # behavior; flip to True to restore it.
        self.use_only_colorized_points = False

        self.K_mats = {}
        self.T_c_l_mats = {}
        self.cam_widths = {}
        self.cam_heights = {}
        self.intrinsics_o3d = {}
        self.img_files = {}
        self.camera_names = []

        # --- camera 2 (main, rectified pinhole) from calib.txt (KITTI format) ---
        # POLICY: a prepped camera must align 1:1 with the scans. We never
        # silently drop a sensor -- a count mismatch means prep is broken, so
        # fail the load instead of mapping with a quietly degraded sensor set.
        self.calibration = self.read_calib_file(os.path.join(self.seq_dir, "calib.txt"))
        calib_data = self._load_calib()
        cam2_imgs = sorted(glob.glob(os.path.join(self.seq_dir, "image_2/") + "*.png"))
        if os.path.isdir(os.path.join(self.seq_dir, "image_2")) and len(cam2_imgs) != scan_count:
            raise ValueError(f"[geoscan] cam2 has {len(cam2_imgs)} images for {scan_count} scans"
                             " -- prepped cameras must match 1:1; re-run prep")
        if len(cam2_imgs) == scan_count:
            self.img_files["cam2"] = cam2_imgs
            self._add_cam("cam2", calib_data["K_cam2"], calib_data["T_cam2_velo"], cam2_imgs[0])

        # --- extra cameras (cam3 = left fisheye, cam4 = realsense, ...) ----------
        # Each extra camera N is described by cam<N>.json (rectified K + lidar->cam
        # extrinsic) with image_<N>/*.png, timestamp-matched to the scans by
        # add_camera.py. A point is kept if it lands in ANY camera.
        for n in range(3, 9):
            cam_json = os.path.join(self.seq_dir, f"cam{n}.json")
            if not os.path.exists(cam_json):
                continue
            cam_imgs = sorted(glob.glob(os.path.join(self.seq_dir, f"image_{n}/") + "*.png"))
            if len(cam_imgs) != scan_count:
                raise ValueError(f"[geoscan] cam{n} has {len(cam_imgs)} images for {scan_count}"
                                 " scans -- prepped cameras must match 1:1; re-run prep")
            cj = json.load(open(cam_json))
            self.img_files[f"cam{n}"] = cam_imgs
            self._add_cam(f"cam{n}", np.asarray(cj["K"], dtype=float),
                          np.asarray(cj["T_cam_lidar"], dtype=float), cam_imgs[0])

        self.main_cam_name = "cam2" if "cam2" in self.camera_names else (self.camera_names[0] if self.camera_names else None)
        self.left_cam_name = self.main_cam_name  # kept for kitti-compat
        self.image_available = len(self.camera_names) > 0

        # --- per-frame sensor timestamps (times.txt + times_camN.txt) ------------
        # When every active camera has a times_cam<N>.txt (written by add_camera.py),
        # emit frame_data["sensor_ts"] so PINGS' per-camera pose slerp correction
        # (slam_dataset.get_cur_cam_ref_ts_ratio) can compensate residual camera/LiDAR
        # timing differences. All stamps must share one clock.
        self.main_lidar_name = "lidar"
        self.lidar_ts = None
        self.cam_ts = {}
        times_file = os.path.join(self.seq_dir, "times.txt")
        if os.path.exists(times_file):
            lt = np.loadtxt(times_file, dtype=np.float64).reshape(-1)
            if len(lt) == scan_count:
                self.lidar_ts = lt
        if self.lidar_ts is not None:
            for cam in self.camera_names:
                cam_ts_file = os.path.join(self.seq_dir, f"times_{cam}.txt")
                if os.path.exists(cam_ts_file):
                    ct = np.loadtxt(cam_ts_file, dtype=np.float64).reshape(-1)
                    if len(ct) != scan_count:
                        raise ValueError(f"[geoscan] {os.path.basename(cam_ts_file)} has {len(ct)}"
                                         f" rows for {scan_count} scans -- re-run prep")
                    self.cam_ts[cam] = ct
        self.sensor_ts_available = (
            self.lidar_ts is not None
            and len(self.cam_ts) == len(self.camera_names)
            and len(self.camera_names) > 0
        )
        if self.sensor_ts_available:
            print(f"[geoscan] sensor_ts on: times.txt + {len(self.cam_ts)} camera timestamp files")

        # SLAMDataset force-disables deskew if the loader has a 'deskew_off' attr
        # (value-agnostic). So only set it when we lack real per-point timestamps.
        if not self.has_real_ts:
            self.deskew_off = True

        # --- provided poses: sequences/<seq>/poses.txt = KITTI 3x4 rows, world<-lidar.
        # In tracker mode they serve ONLY as the trajectory-evaluation reference; in
        # pose-fixed mode (no tracker: section in the config) they drive the mapping.
        poses_file = os.path.join(self.seq_dir, "poses.txt")
        if os.path.exists(poses_file):
            gp = self.load_poses(poses_file)
            if len(gp) == scan_count:
                self.gt_poses = gp
                print(f"[geoscan] loaded {len(gp)} provided poses (world<-lidar) from poses.txt")
            else:
                print(f"[geoscan] poses.txt has {len(gp)} != {scan_count} scans -- ignoring")

    def _add_cam(self, name, K, T_c_l, sample_img):
        K = np.asarray(K, dtype=float)[:3, :3]
        first = cv2.imread(sample_img)
        H, W = first.shape[0], first.shape[1]
        self.K_mats[name] = K
        self.T_c_l_mats[name] = np.asarray(T_c_l, dtype=float)
        self.cam_widths[name] = W
        self.cam_heights[name] = H
        intr = o3d.camera.PinholeCameraIntrinsic()
        intr.set_intrinsics(height=H, width=W, fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2])
        self.intrinsics_o3d[name] = intr
        self.camera_names.append(name)

    def __getitem__(self, idx):
        points = self.scans(idx)
        point_ts = self.get_point_ts_for(idx, points)

        if self.load_img and self.image_available:
            points_rgb = np.ones_like(points)  # N,4; col 3 is the colour mask (0 = coloured)
            img_dict = {}
            depth_dict = {}
            for cam in self.camera_names:
                img = self.read_img(self.img_files[cam][idx])
                points_rgb, depth_map = self.project_points_to_cam(
                    points, points_rgb, img, self.T_c_l_mats[cam], self.K_mats[cam])
                img_dict[cam] = img
                depth_dict[cam] = depth_map

            if self.use_only_colorized_points:
                with_rgb_mask = (points_rgb[:, 3] == 0)  # coloured by at least one camera
                points = points[with_rgb_mask]
                points_rgb = points_rgb[with_rgb_mask]
                point_ts = point_ts[with_rgb_mask]

            points = np.hstack((points[:, :3], points_rgb[:, :3]))
            frame_data = {"points": points, "point_ts": point_ts, "img": img_dict, "depth": depth_dict}
        else:
            frame_data = {"points": points, "point_ts": point_ts}

        if self.sensor_ts_available:
            sensor_ts = {self.main_lidar_name: float(self.lidar_ts[idx])}
            for cam, ct in self.cam_ts.items():
                sensor_ts[cam] = float(ct[idx])
            frame_data["sensor_ts"] = sensor_ts

        return frame_data

    def __len__(self):
        return len(self.scan_files)

    def scans(self, idx):
        return self.read_point_cloud(self.scan_files[idx])

    def read_point_cloud(self, scan_file: str):
        return np.fromfile(scan_file, dtype=np.float32).reshape((-1, 4))[:, :4].astype(np.float64)

    def read_img(self, img_file: str):
        img = cv2.imread(img_file)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def get_point_ts_for(self, idx, points):
        # real normalized [0,1] per-point time (Livox offset_time) if available,
        # else the azimuth guess (only valid for spinning LiDAR, not Livox).
        if self.has_real_ts:
            ts = np.fromfile(self.ts_files[idx], dtype=np.float32).astype(np.float64)
            if len(ts) == len(points):
                return ts
        return self.get_timestamps(points)

    @staticmethod
    def get_timestamps(points):
        x = points[:, 0]
        y = points[:, 1]
        yaw = -np.arctan2(y, x)
        return 0.5 * (yaw / np.pi + 1.0)

    @staticmethod
    def load_poses(poses_file):
        # poses.txt: N x 12 (KITTI 3x4 rows), ALREADY world<-lidar. Unlike kitti.py
        # we do NOT apply inv(Tr)@p@Tr -- the provided trajectory is in the LiDAR
        # frame, which is exactly what SLAMDataset's gt_poses expects.
        p = np.loadtxt(poses_file, dtype=np.float64)
        if p.ndim == 1:
            p = p.reshape(1, -1)
        n = p.shape[0]
        T = np.tile(np.eye(4, dtype=np.float64), (n, 1, 1))
        T[:, :3, :4] = p.reshape(n, 3, 4)
        return T

    @staticmethod
    def read_calib_file(file_path: str) -> dict:
        calib_dict = {}
        with open(file_path, "r") as calib_file:
            for line in calib_file.readlines():
                tokens = line.split(" ")
                if tokens[0] == "calib_time:":
                    continue
                if len(tokens) > 0:
                    values = np.array([float(t) for t in tokens[1:]], dtype=np.float32)
                    calib_dict[tokens[0][:-1]] = values
        return calib_dict

    def project_points_to_cam(self, points, points_rgb, img, T_c_l, K_mat):
        points[:, 3] = 1
        points_cam = np.matmul(T_c_l, points.T).T[:, :3]
        u, v, depth = self.persepective_cam2image(points_cam.T, K_mat)
        u = u.astype(np.int32)
        v = v.astype(np.int32)
        img_height, img_width, _ = np.shape(img)
        depth_map = np.zeros((img_height, img_width, 1))
        mask = np.logical_and(np.logical_and(np.logical_and(u >= 0, u < img_width), v >= 0), v < img_height)
        mask = np.logical_and(np.logical_and(mask, depth > 0.3), depth < 100.0)
        v_valid = v[mask]
        u_valid = u[mask]
        depth_map[v_valid, u_valid, 0] = depth[mask]
        points_rgb[mask, :3] = img[v_valid, u_valid].astype(np.float64) / 255.0
        points_rgb[mask, 3] = 0
        return points_rgb, depth_map

    def persepective_cam2image(self, points, K_mat):
        ndim = points.ndim
        if ndim == 2:
            points = np.expand_dims(points, 0)
        points_proj = np.matmul(K_mat[:3, :3].reshape([1, 3, 3]), points)
        depth = points_proj[:, 2, :]
        depth[depth == 0] = -1e-6
        u = np.round(points_proj[:, 0, :] / np.abs(depth)).astype(int)
        v = np.round(points_proj[:, 1, :] / np.abs(depth)).astype(int)
        if ndim == 2:
            u = u[0]; v = v[0]; depth = depth[0]
        return u, v, depth

    def _load_calib(self):
        data = {}
        filedata = self.calibration
        P_rect_20 = np.reshape(filedata["P2"], (3, 4))
        T2 = np.eye(4)
        T2[0, 3] = P_rect_20[0, 3] / P_rect_20[0, 0]
        T_cam0_velo = np.reshape(filedata["Tr"], (3, 4))
        T_cam0_velo = np.vstack([T_cam0_velo, [0, 0, 0, 1]])
        data["T_cam2_velo"] = T2.dot(T_cam0_velo)
        data["K_cam2"] = P_rect_20[0:3, 0:3]
        return data
