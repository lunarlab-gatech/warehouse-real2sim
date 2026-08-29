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

import open3d as o3d
import cv2
import numpy as np

import yaml

from datetime import datetime

from utils.tools import get_time

# Data loader for the in-house car dataset collected in Bonn (to-be-released after anynomization)
# This is the dataset used for the main experiments in the paper

class IPBCarDataset:
    def __init__(self, data_dir, cam_name: str, *_, **__):
        
        self.contains_image = True
        self.load_img = False # default

        self.use_only_colorized_points = False
        
        self.use_only_lidar_h = True
        self.lidar_list = ["lidar_h"] 
        self.main_lidar_name = self.lidar_list[0]

        if cam_name == "both_lidars":
            self.use_only_lidar_h = False # use lidar_h or both (lidar_h + lidar_v)
            self.lidar_list.append("lidar_v") 

        self.min_lidar_radius_m = 0.5

        self.lidar_h_topic_name = "horizontal" 
        self.lidar_v_topic_name = "vertical" 
        
        self.cam_left_topic_name = "left"  
        self.cam_right_topic_name = "right" 
        self.cam_front_topic_name = "front" 
        self.cam_rear_topic_name = "rear"

        cam_list_all = [self.cam_front_topic_name, self.cam_left_topic_name, self.cam_rear_topic_name, self.cam_right_topic_name]

        if cam_name in cam_list_all: 
            self.main_cam_only = True
            self.main_cam_name = cam_name
            self.cam_list = [self.main_cam_name]
            print("Use {} camera only".format(cam_name))
        else: 
            # use all the cameras
            self.main_cam_only = False
            self.main_cam_name = self.cam_front_topic_name
            self.cam_list = cam_list_all
            print("Use all the cameras")

        self.img_files = {}
        self.img_ts = {}
        self.K_mats = {}
        self.dist_coeffs = {}
        self.T_c_l_mats = {}
        self.T_l_lm_mats = [] # inter-lidar transformation, from the main LiDAR (horizontal) to other LiDARs (vertical)
        self.ts_ref_ratio_diffs = [0.0]

        self.cam_widths = {}
        self.cam_heights = {}

        self.cam_valid_v_ratios_minmax = {
            "front": [0.04, 1.0],
            "left": [0.0, 1.0],
            "right": [0.0, 1.0],
            "rear": [0.12, 1.0]
        }

        # horizontal lidar
        self.lidar_horizontal_dir = os.path.join(data_dir, "lidar_{}_points".format(self.lidar_h_topic_name), "data/")
        self.lidar_horizontal_files = sorted(glob.glob(self.lidar_horizontal_dir + "*.ply")) # we use bin here, can not be directly visualized but would be much smaller
        self.lidar_horizontal_ts = self.read_timestamps(os.path.join(data_dir, "lidar_{}_points".format(self.lidar_h_topic_name), "timestamps.txt"))

        # vertical lidar
        if not self.use_only_lidar_h:
            self.lidar_vertical_dir = os.path.join(data_dir, "lidar_{}_points".format(self.lidar_v_topic_name), "data/")
            self.lidar_vertical_files = sorted(glob.glob(self.lidar_vertical_dir + "*.ply"))
            self.lidar_vertical_ts = self.read_timestamps(os.path.join(data_dir, "lidar_{}_points".format(self.lidar_v_topic_name), "timestamps.txt"))

        # img_size: 2064x1024
        H, W = 1024, 2064 

        self.img_already_undistorted = False
        
        for cam_name in self.cam_list:
            cur_cam_dir = os.path.join(data_dir, "camera_{}".format(cam_name), "data/")
            cur_img_files = sorted(glob.glob(cur_cam_dir + "*.png"))
            
            # create folder if not yet there
            cur_cam_undistorted_dir = os.path.join(data_dir, "camera_{}".format(cam_name), "data_undistorted/")
            os.makedirs(cur_cam_undistorted_dir, 0o755, exist_ok=True)

            if len(cur_img_files) == 0:
                cur_img_files = sorted(glob.glob(cur_cam_undistorted_dir + "*.png"))
            
            assert len(cur_img_files) > 0, "No image data in data or data_undistorted folder."
                
            self.img_files[cam_name] = cur_img_files

            cur_img_ts = self.read_timestamps(os.path.join(data_dir, "camera_{}".format(cam_name), "timestamps.txt"))
            self.img_ts[cam_name] = cur_img_ts
            
            self.cam_widths[cam_name] = W
            self.cam_heights[cam_name] = H

        # read calib
        self.calibration_dict = self.read_calib_file(os.path.join(data_dir, "calibration", "results.yaml"))

        # read reference poses (by Louis)

        # poses_file = os.path.join(data_dir, "poses.txt") # TODO: this is the globally bundle adjustment pose (but not with TLS constraints yet)
        poses_file = os.path.join(data_dir, "poses_pin_slam.txt")
        if os.path.exists(poses_file):
            self.gt_poses = self.read_kitti_format_poses(poses_file)
            # self.gt_poses = np.load(os.path.join(data_dir, "poses", "latest.npy"))
            # print("gt poses for {} frames".format(np.shape(self.gt_poses)[0]))

        # print(self.gt_poses) 
        # NOTE: for the UTC coordinate, we need to substract the larger part from them (other wise float would have precision lose)
        # we use a Bonn local reference coordinate : 3.65e5, 5.62e6, 1e2
        # self.gt_poses = np.load(os.path.join(data_dir, "poses", "gt_poses_joint.npy"))
        # bonn_ref_xyz_utc = np.array([3.65e5, 5.62e6, 1e2])
        # self.gt_poses[:,:3,3] -= bonn_ref_xyz_utc

        # main cam parameters
        self.intrinsic = o3d.camera.PinholeCameraIntrinsic()

        # NOTE for the rear camera, from about h=920 is the ego car's tail

        self.intrinsic.set_intrinsics(
                                    height=H,
                                    width=W,
                                    fx=self.K_mats[self.main_cam_name][0,0],
                                    fy=self.K_mats[self.main_cam_name][1,1],
                                    cx=self.K_mats[self.main_cam_name][0,2],
                                    cy=self.K_mats[self.main_cam_name][1,2])

        self.extrinsic = self.T_c_l_mats[self.main_cam_name] # T_c_l

        self.mono_depth_for_high_z: bool = False # complete the low Z part 


    def __getitem__(self, idx):
        
        frame_data = {}

        # tic_read_pc = get_time()
        sensor_ts_dict = {}

        h_lidar_ref_ts = self.lidar_horizontal_ts[idx] # unit: s
        sensor_ts_dict["lidar_h"] = h_lidar_ref_ts
        # print("H Lidar ts: {}".format(h_lidar_ref_ts))

        points, point_ts = self.read_point_cloud_ply(self.lidar_horizontal_files[idx]) # lidar_h_points

        # toc_read_pc = get_time()
        # print("pc reading time (ms):" , (toc_read_pc -tic_read_pc)*1e3)

        valid_mask = ~np.all(np.abs(points[:,:3]) < self.min_lidar_radius_m, axis=1) 
        points = points[valid_mask]
        point_ts = point_ts[valid_mask] # unit: s

        # print(point_ts)
        
        point_lidar_idx = np.zeros_like(point_ts) # all zero

        lidar_h_point_count = np.shape(points)[0]
        # from 0 to lidar_h_point_count-1: lidar_h
        # from lidar_h_point_count to end: lidar_v

        # FIXME: it's not correct to firstly combine the two point clouds are then apply undistortion
        # you need to apply a T_lv_lh to the points from the vertical liadr during the undistortion
        # Now we only use the single horizontal lidar
        if not self.use_only_lidar_h:
            
            v_lidar_ref_ts = self.lidar_vertical_ts[idx]
            sensor_ts_dict["lidar_v"] = v_lidar_ref_ts
            # print("V Lidar ts: {}".format(v_lidar_ref_ts))

            lidar_v_points, lidar_v_points_ts = self.read_point_cloud_ply(self.lidar_vertical_files[idx])
            # print(np.shape(lidar_v_points)[0])

            lidar_v_points_homo = np.hstack((lidar_v_points[:,:3], np.ones((np.shape(lidar_v_points)[0], 1))))

            lidar_v_points_h_frame = lidar_v_points_homo @ self.T_lv_lh.T
            lidar_v_points[:,:3] = lidar_v_points_h_frame[:,:3]

            valid_mask = ~np.all(np.abs(lidar_v_points[:,:3]) < self.min_lidar_radius_m, axis=1) 
            lidar_v_points = lidar_v_points[valid_mask]
            lidar_v_points_ts = lidar_v_points_ts[valid_mask]

            # this is actually not a very big number, consider it later FIXME
            # lidar_v_points_ts += (v_lidar_ref_ts - h_lidar_ref_ts) # convert to the same reference time of the horizontal lidar

            added_ref_ts_ratio = (h_lidar_ref_ts - v_lidar_ref_ts) / 0.1
            self.ts_ref_ratio_diffs[0] = added_ref_ts_ratio
            # print(added_ref_ts_ratio)

            lidar_v_point_lidar_idx = np.ones_like(lidar_v_points_ts)

            points = np.concatenate((points, lidar_v_points), axis=0) # K, 4
            point_ts = np.concatenate((point_ts, lidar_v_points_ts), axis=0)
            point_lidar_idx = np.concatenate((point_lidar_idx, lidar_v_point_lidar_idx), axis=0)

        if self.load_img:

            img_dict = {}
            depth_img_dict = {}

            points_rgb = -1.0 * np.ones_like(points) # N,4, last channel for the mask # set to invalid (indicated by negative value) at first

            for cam_name in self.cam_list:
                
                # tic_0 = get_time()
                # slow, but would be hard to speed up

                sensor_ts_dict[cam_name] = self.img_ts[cam_name][idx]
                # print("{} ts: {}".format(cam_name, self.img_ts[cam_name][idx]))

                cur_img_file = self.img_files[cam_name][idx]

                img_file_split = cur_img_file.split("/")
                img_file_split[-2] = "data_undistorted" # data --> data_undistorted
                cur_img_file_distorted = "/".join(img_file_split)

                if os.path.exists(cur_img_file_distorted):
                    undistort_on = False # if already exists the undistorted file, then use it
                    cur_img_file = cur_img_file_distorted
                else:
                    undistort_on = True # otherwise, do the distortion and save the file

                # print(cur_img_file)

                img_cam = self.read_img(cur_img_file, undistort_on, self.K_mats[cam_name], self.dist_coeffs[cam_name]) 
                
                # toc_1 = get_time()
                
                # FIXME: not used now
                # points_rgb, depth_map = self.project_points_to_cam(points, points_rgb, img_cam, self.T_c_l_mats[cam_name], self.K_mats[cam_name])

                # toc_1 = get_time()

                # depth_img_dict[cam_name] = depth_map # H, W, 1
                
                img_dict[cam_name] = img_cam # H, W, 3
                
                # print("img reading time (ms):" , (toc_0 - tic_0)*1e3)
                # print("pc colorize time (ms):" , (toc_1 - toc_0)*1e3)

            # FIXME
            # if self.use_only_colorized_points:
            #     with_rgb_mask = (points_rgb[:, 3] == 0)
            #     points = points[with_rgb_mask]
            #     points_rgb = points_rgb[with_rgb_mask]
            #     point_ts = point_ts[with_rgb_mask]
            #     point_lidar_idx = point_lidar_idx[with_rgb_mask]

            # we skip the intensity here for now (and also the color mask)
            points = np.hstack((points[:,:3], points_rgb[:,:3]))

            # print(point_ts) # correct
            frame_data["img"] = img_dict 
            # frame_data["depth"] = depth_img_dict  

        frame_data.update({"points": points, "point_ts": point_ts, "point_lidar_idx": point_lidar_idx})

        return frame_data

    def __len__(self):
        return len(self.lidar_horizontal_files)
    
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

    # # read bin format
    # def read_point_cloud(self, scan_file: str):
    #     points = np.fromfile(scan_file, dtype=np.float32).reshape((-1, 4)).astype(np.float64)
    #     return points[:, :4] # N, 4
    
    # read ply format
    def read_point_cloud_ply(self, scan_file: str):

        # tic_read_pc = get_time()

        pc_load = o3d.t.io.read_point_cloud(scan_file)
        pc_load = {k: v.numpy() for k, v in pc_load.point.items()}

        keys = list(pc_load.keys())
        # print("available attributes:", keys)
        
        points = pc_load["positions"]

        assert "t" in keys, "The point cloud ply file must have the t field for point wise timestamp"

        ts = pc_load["t"] # already in seconds

        point_count = np.shape(points)[0]
        points = np.concatenate((points, np.ones((point_count, 1))), axis=1) # N, 4
        
        # toc_read_pc = get_time()
        # print("pc reading time ply (ms): {:.2f}".format((toc_read_pc -tic_read_pc)*1e3))

        return points, ts

    def read_img(self, img_file: str, undistort_on: bool = False, K_mat = None, dist_coeffs = None):
        
        tic_0 = get_time()
        img = cv2.imread(img_file)
        toc_0 = get_time()

        # print("img reading time (ms):" , (toc_0 -tic_0)*1e3)

        # apply undistortion:
        # tic_undistort = get_time()
        if undistort_on and K_mat is not None and dist_coeffs is not None:
            img = cv2.undistort(img, K_mat, dist_coeffs)
            img_file_split = img_file.split("/")
            img_file_split[-2] = "data_undistorted" # data --> data_undistorted # ugly fix here
            out_img_file = "/".join(img_file_split)
            # print(out_img_file)

            cv2.imwrite(out_img_file, img)

        # toc_undistort = get_time() 

        # print("Image undistortion time (ms):", (toc_undistort-tic_undistort)*1e3) # 12 ms / each 
        # for 4 imgs, takes about 50ms, better to do this offline
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img

    def read_calib_file(self, yaml_file_path: str) -> dict:

        calib_dict = {}
        with open(yaml_file_path, 'r') as file:
            calib_dict = yaml.safe_load(file)

            lidar_h_calib = calib_dict["lidarhorizontalpoints"]
            lidar_v_calib = calib_dict["lidarverticalpoints"]
            T_cf_lh = np.array(lidar_h_calib["extrinsics"])
            T_cf_lv = np.array(lidar_v_calib["extrinsics"])
            self.T_lv_lh = np.linalg.inv(T_cf_lv) @ T_cf_lh
            if not self.use_only_lidar_h:
                self.T_l_lm_mats.append(self.T_lv_lh)

            for cam_name in self.cam_list:
                cur_cam_calib_name = "camera{}image_raw".format(cam_name)
                cur_camera_calib = calib_dict[cur_cam_calib_name]
                self.K_mats[cam_name] = np.array(cur_camera_calib["K"])
                self.dist_coeffs[cam_name] = np.array(cur_camera_calib["distortion_coeff"])
                self.T_c_l_mats[cam_name] = np.linalg.inv(np.array(cur_camera_calib["extrinsics"])) @ T_cf_lh 

        return calib_dict
    
    def project_points_to_cam(self, points, points_rgb, img, T_c_l, K_mat):
        
        tic_0 = get_time()

        # points as np.numpy (N,4)
        points[:,3] = 1 # homo coordinate

        # transfrom velodyne points to camera coordinate
        points_cam = np.matmul(T_c_l, points.T).T # N, 4
        points_cam = points_cam[:,:3] # N, 3

        toc_0 = get_time() # fast

        # project to image space
        u, v, depth = self.persepective_cam2image(points_cam.T, K_mat) 
        u = u.astype(np.int32)
        v = v.astype(np.int32)

        img_height, img_width, _ = np.shape(img)

        toc_1 = get_time() # relatively fast

        # prepare depth map for visualization
        depth_map = np.zeros((img_height, img_width, 1)) # H, W, 1
        mask = np.logical_and(np.logical_and(np.logical_and(u>=0, u<img_width), v>=0), v<img_height)
        
        # visualize points within 30 meters
        min_depth = 1.0
        max_depth = 100.0
        mask = np.logical_and(np.logical_and(mask, depth>min_depth), depth<max_depth)
        
        v_valid = v[mask]
        u_valid = u[mask]

        depth_map[v_valid,u_valid, 0] = depth[mask]

        # print(np.shape(points_rgb))

        points_rgb[mask, :3] = img[v_valid,u_valid].astype(np.float64)/255.0 # 0-1
        points_rgb[mask, 3] = 0 # has color

        toc_2 = get_time() # slow

        # print("colorize time 1 (ms):" , (toc_0 -tic_0)*1e3)
        # print("colorize time 2 (ms):" , (toc_1 -toc_0)*1e3)
        # print("colorize time 3 (ms):" , (toc_2 -toc_1)*1e3)

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


    def read_kitti_format_poses(self, filename: str):
        """
        read pose file (with the kitti format)
        returns -> np.array, transformation before calibration transformation
        if the format is incorrect, return None
        """
        poses = []
        with open(filename, 'r') as file:            
            for line in file:
                values = line.strip().split()
                if len(values) < 12: # FIXME: > 12 means maybe it's a 4x4 matrix
                    print('Not a kitti format pose file')
                    return None

                values = [float(value) for value in values]
                pose = np.zeros((4, 4))
                pose[0, 0:4] = values[0:4]
                pose[1, 0:4] = values[4:8]
                pose[2, 0:4] = values[8:12]
                pose[3, 3] = 1.0
                poses.append(pose)
        
        return np.array(poses)