# MIT License
#
# Copyright (c) 2022 Ignacio Vizzo, Tiziano Guadagnino, Benedikt Mersch, Cyrill
# Stachniss.
# 2024 Yue Pan
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to mse, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE msE OR OTHER DEALINGS IN THE
# SOFTWARE.
import glob
import os

import cv2
import numpy as np

import open3d as o3d

import yaml

from datetime import datetime

# R3Live Livox dataset in kitti-like format
# https://github.com/ziv-lin/r3live_dataset

# camera parameters
# r3live_vio:
#    image_width: 1280
#    image_height: 1024
#    camera_intrinsic:
#       [863.4241, 0.0, 640.6808,
#       0.0,  863.4171, 518.3392,
#       0.0, 0.0, 1.0 ] 
#    camera_dist_coeffs: [-0.1080, 0.1050, -1.2872e-04, 5.7923e-05, -0.0222]  #k1, k2, p1, p2, k3
#    # Fine extrinsic value. form camera-LiDAR calibration.
#    camera_ext_R:
#          [-0.00113207, -0.0158688, 0.999873,
#             -0.9999999,  -0.000486594, -0.00113994,
#             0.000504622,  -0.999874,  -0.0158682]
#    # camera_ext_t: [0.050166, 0.0474116, -0.0312415] 
#    camera_ext_t: [0,0,0] 

class R3LiveDataset:
    def __init__(self, data_dir, *_, **__):
        
        self.contains_image = True
        self.load_img = False # default
        self.use_only_colorized_points = False

        self.livox_dir = os.path.join(data_dir, "livox_points", "data/")
        self.scan_files = sorted(glob.glob(self.livox_dir + "*.bin"))
        scan_count = len(self.scan_files)
        self.scan_ts = self.read_timestamps(os.path.join(data_dir, "livox_points", "timestamps.txt"))

        # camera
        self.img_dir = os.path.join(data_dir, "camera_image_color_compressed", "data/")
        self.img_files = sorted(glob.glob(self.img_dir + "*.png"))
        self.img_ts = self.read_timestamps(os.path.join(data_dir, "camera_image_color_compressed", "timestamps.txt"))

        # synchronize lidar and camera
        self.img_ts_sync, img_idx_sync = self.associate_img_to_lidar(self.scan_ts, self.img_ts)
        self.img_files = [self.img_files[i] for i in img_idx_sync] # get the synchronized files
        img_count = len(self.img_files)

        self.main_cam_name = "cam"

        H, W = 1024, 1280

        self.K_mats = {}
        self.T_c_l_mats = {}
        self.dist_coeffs = {}
       
        self.fx = 863.4241
        self.fy = 863.4171
        self.cx = 640.6808
        self.cy = 518.3392
        K_mat = np.eye(3)
        K_mat[0,0] = self.fx
        K_mat[0,2] = self.cx
        K_mat[1,1] = self.fy
        K_mat[1,2] = self.cy

        self.intrinsic = o3d.camera.PinholeCameraIntrinsic()
        self.intrinsic.set_intrinsics(height=H,
                                      width=W,
                                      fx=self.fx,
                                      fy=self.fy,
                                      cx=self.cx,
                                      cy=self.cy)

        self.K_mats[self.main_cam_name] = K_mat

        self.dist_coeffs[self.main_cam_name] = np.array([-0.1080, 0.1050, -1.2872e-04, 5.7923e-05, -0.0222])

        T_l_c = np.eye(4)
        T_l_c[:3,:3] = np.array([[-0.00113207, -0.0158688, 0.999873],
                                [-0.9999999,  -0.000486594, -0.00113994],
                                [0.000504622,  -0.999874,  -0.0158682]])

        T_l_c[:3,3] = np.array([0.050166, 0.0474116, -0.0312415]) 

        T_c_l = np.linalg.inv(T_l_c)

        self.T_c_l_mats[self.main_cam_name] = T_c_l

        self.cam_heights = {self.main_cam_name: H}
        self.cam_widths = {self.main_cam_name: W}

        self.extrinsic = T_c_l

        self.mono_depth_for_high_z: bool = False

        # no gt pose available
        # gt_poses_file = os.path.join(data_dir, "gt.txt")
        # if os.path.exists(gt_poses_file):
        #     self.gt_poses, self.scan_timestamps = self.load_tum_format_gt_poses(gt_poses_file) 
        #     self.gt_poses = np.array(self.gt_poses)

    def __getitem__(self, idx):
        
        points = self.scans(idx)

        # livox timestamp
        point_count = np.shape(points)[0] # 24000 for livox
        point_ts = np.arange(point_count)*1.0/point_count

        if self.load_img: # default on

            cur_img_file = self.img_files[idx]

            img_file_split = cur_img_file.split("/")
            img_file_split[-2] = "data_undistorted" # data --> data_undistorted
            cur_img_file_distorted = "/".join(img_file_split)

            if os.path.exists(cur_img_file_distorted):
                undistort_on = False # if already exists the undistorted file, then use it
                cur_img_file = cur_img_file_distorted
            else:
                undistort_on = True # otherwise, do the distortion and save the file

            img_cam = self.read_img(cur_img_file, undistort_on, self.K_mats[self.main_cam_name], self.dist_coeffs[self.main_cam_name]) 
        
            points_color = np.ones_like(points)

            # project to the image plane to get the corresponding color
            points_color, depth_map = self.project_points_to_cam(points, points_color, img_cam, 
                self.T_c_l_mats[self.main_cam_name], self.K_mats[self.main_cam_name])

            if self.use_only_colorized_points:
                with_rgb_mask = (points_color[:, 3] == 0)
                points = points[with_rgb_mask]
                points_color = points_color[with_rgb_mask]
                point_ts = point_ts[with_rgb_mask]

            # we skip the intensity here for now (and also the color mask)
            points = np.hstack((points[:,:3], points_color[:,:3]))

            img_dict = {self.main_cam_name: img_cam}
            depth_img_dict = {self.main_cam_name: depth_map}

            frame_data = {"points": points, "point_ts": point_ts, "img": img_dict, "depth": depth_img_dict}
        else:
            frame_data = {"points": points, "point_ts": point_ts}

        return frame_data

    def __len__(self):
        return len(self.scan_files)
    
    def read_timestamps(self, file_path):
        # read timestamps are parse to the ROS time format
        timestamps = []
        with open(file_path, 'r') as file:
            for line in file:
                iso_str = line[:-1]
                dt_str, ns_str = iso_str.split('.')
                ns_str = ns_str[:9].ljust(9, '0')
                dt = datetime.strptime(dt_str, "%Y-%m-%dT%H:%M:%S")
                epoch = datetime(1970, 1, 1)
                delta = dt - epoch
                sec = int(delta.total_seconds())
                nsec = int(ns_str)
                timestamp_sec = sec + nsec * 1e-9

                timestamps.append(timestamp_sec)
        timestamps = np.array(timestamps)
        return timestamps
    
    def associate_img_to_lidar(self, lidar_ts, img_ts):
        # for each lidar ts, find the closest img_ts
        img_ts_associated = []
        img_associated_idx = []
        for i in range(lidar_ts.shape[0]):
            cur_lidar_ts = lidar_ts[i]
            j = np.argmin(np.abs(img_ts - cur_lidar_ts))
            img_ts_associated.append(img_ts[j])
            img_associated_idx.append(j)
        img_ts_associated = np.array(img_ts_associated)
        img_associated_idx = np.array(img_associated_idx, dtype=np.int32)
        # print(img_associated_idx)
        return img_ts_associated, img_associated_idx        

    def scans(self, idx):
        points = self.read_point_cloud(self.scan_files[idx])
        return points

    # TODO: figure out why it the point with t does not work with the vbr converter
    def read_point_cloud(self, scan_file: str):
        points = np.fromfile(scan_file, dtype=np.float32).reshape((-1, 4))[:, :4].astype(np.float64)
        return points # N, 4
    
    def read_img(self, img_file: str, undistort_on: bool = False, K_mat = None, dist_coeffs = None):
        
        img = cv2.imread(img_file)

        # apply undistortion:
        if undistort_on and K_mat is not None and dist_coeffs is not None:
            img = cv2.undistort(img, K_mat, dist_coeffs)
            img_file_split = img_file.split("/")
            img_file_split[-2] = "data_undistorted" # data --> data_undistorted
            out_img_file = "/".join(img_file_split)

            cv2.imwrite(out_img_file, img)
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img


    def project_points_to_cam(self, points, points_rgb, img, T_c_l, K_mat):
        
        # points as np.numpy (N,4)
        points[:,3] = 1 # homo coordinate

        # transfrom velodyne points to camera coordinate
        points_cam = np.matmul(T_c_l, points.T).T # N, 4
        points_cam = points_cam[:,:3] # N, 3

        # project to image space
        u, v, depth= self.persepective_cam2image(points_cam.T, K_mat) 
        u = u.astype(np.int32)
        v = v.astype(np.int32)

        img_height, img_width, _ = np.shape(img)

        # prepare depth map for visualization
        depth_map = np.zeros((img_height, img_width, 1))
        #
        mask = np.logical_and(np.logical_and(np.logical_and(u>=0, u<img_width), v>=0), v<img_height)
        
        # visualize points within 30 meters
        min_depth = 1.0
        max_depth = 100.0
        mask = np.logical_and(np.logical_and(mask, depth>min_depth), depth<max_depth)
        
        v_valid = v[mask]
        u_valid = u[mask]

        depth_map[v_valid,u_valid,0] = depth[mask]

        # print(np.shape(points_rgb))

        points_rgb[mask, :3] = img[v_valid,u_valid].astype(np.float64)/255.0 # 0-1
        points_rgb[mask, 3] = 0 # has color

        return points_rgb, depth_map
    
    def persepective_cam2image(self, points, K_mat):
        ndim = points.ndim
        if ndim == 2:
            points = np.expand_dims(points, 0)
        points_proj = np.matmul(K_mat[:3,:3].reshape([1,3,3]), points)
        depth = points_proj[:,2,:]
        depth[depth==0] = -1e-6
        u = np.round(points_proj[:,0,:]/np.abs(depth)).astype(int)
        v = np.round(points_proj[:,1,:]/np.abs(depth)).astype(int)

        if ndim==2:
            u = u[0]; v=v[0]; depth=depth[0]
        return u, v, depth
