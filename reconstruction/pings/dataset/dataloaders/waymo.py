import glob
import os
from pathlib import Path
from typing import List

import cv2
import json
import numpy as np
import open3d as o3d

# https://waymo.com/open/

class WaymoDataset:
    def __init__(self, data_dir, *_, **__):
        
        self.contains_image = True
        self.use_only_lidar_top = True 
        self.load_img = False # default
        self.use_only_colorized_points = True

        self.lidar_top_topic_name = "lidar_TOP" 
        self.lidar_front_topic_name = "lidar_FRONT" 
        self.lidar_rear_topic_name = "lidar_REAR"
        self.lidar_side_left_topic_name = "lidar_SIDE_LEFT"
        self.lidar_side_right_topic_name = "lidar_SIDE_RIGHT"

        self.cam_front_topic_name = "FRONT"  
        self.cam_front_left_topic_name = "FRONT_LEFT" 
        self.cam_front_right_topic_name = "FRONT_RIGHT"
        self.cam_side_left_topic_name = "SIDE_LEFT"
        self.cam_side_right_topic_name = "SIDE_RIGHT"

        self.main_cam_only: bool = True
        self.main_cam_name = self.cam_front_topic_name

        if self.main_cam_only:
            self.cam_names = [self.main_cam_name]
        else:
            self.cam_names = [self.cam_front_topic_name, self.cam_front_left_topic_name, self.cam_front_right_topic_name, self.cam_side_left_topic_name, self.cam_side_right_topic_name]

        self.K_mats = {}
        self.T_c_l_mats = {}

        self.lidar_dir = os.path.join(data_dir, "lidars")
        # top center lidar
        self.lidar_top_dir = os.path.join(self.lidar_dir, self.lidar_top_topic_name)
        self.lidar_top_files = sorted(glob.glob(self.lidar_top_dir + "/*.pcd"))
        # front blind-area lidar 
        self.lidar_front_dir = os.path.join(self.lidar_dir, self.lidar_front_topic_name)
        self.lidar_front_files = sorted(glob.glob(self.lidar_front_dir + "/*.pcd"))
        # rear blind-area lidar 
        self.lidar_rear_dir = os.path.join(self.lidar_dir, self.lidar_rear_topic_name)
        self.lidar_rear_files = sorted(glob.glob(self.lidar_rear_dir + "/*.pcd"))
        # side left blind-area lidar 
        self.lidar_side_left_dir = os.path.join(self.lidar_dir, self.lidar_side_left_topic_name)
        self.lidar_side_left_files = sorted(glob.glob(self.lidar_side_left_dir + "/*.pcd"))
        # side right blind-area lidar 
        self.lidar_side_right_dir = os.path.join(self.lidar_dir, self.lidar_side_right_topic_name)
        self.lidar_side_right_files = sorted(glob.glob(self.lidar_side_right_dir + "/*.pcd"))

        self.img_dir = os.path.join(data_dir, "images_ud") # already undistorted?
        if not os.path.exists(self.img_dir):
            self.img_dir = os.path.join(data_dir, "images") 

        # camera front
        self.img_front_dir = os.path.join(self.img_dir, self.cam_front_topic_name)
        self.img_front_files = sorted(glob.glob(self.img_front_dir + "/*.jpg"))
        # camera front left
        self.img_front_left_dir = os.path.join(self.img_dir, self.cam_front_left_topic_name)
        self.img_front_left_files = sorted(glob.glob(self.img_front_left_dir + "/*.jpg"))
        # camera front right
        self.img_front_right_dir = os.path.join(self.img_dir, self.cam_front_right_topic_name)
        self.img_front_right_files = sorted(glob.glob(self.img_front_right_dir + "/*.jpg"))
        # camera side left
        self.img_side_left_dir = os.path.join(self.img_dir, self.cam_side_left_topic_name)
        self.img_side_left_files = sorted(glob.glob(self.img_side_left_dir + "/*.jpg"))
        # camera side right
        self.img_side_right_dir = os.path.join(self.img_dir, self.cam_side_right_topic_name)
        self.img_side_right_files = sorted(glob.glob(self.img_side_right_dir + "/*.jpg"))

        # add dynamic masks
        self.img_mask_dir = os.path.join(data_dir, "masks") 
        self.img_front_mask_dir = os.path.join(self.img_mask_dir, self.cam_front_topic_name)
        self.img_front_mask_files = sorted(glob.glob(self.img_front_mask_dir + "/*.png"))

        self.img_front_left_mask_dir = os.path.join(self.img_mask_dir, self.cam_front_left_topic_name)
        self.img_front_left_mask_files = sorted(glob.glob(self.img_front_left_mask_dir + "/*.png"))

        self.img_front_right_mask_dir = os.path.join(self.img_mask_dir, self.cam_front_right_topic_name)
        self.img_front_right_mask_files = sorted(glob.glob(self.img_front_right_mask_dir + "/*.png"))

        self.img_side_left_mask_dir = os.path.join(self.img_mask_dir, self.cam_side_left_topic_name)
        self.img_side_left_mask_files = sorted(glob.glob(self.img_side_left_mask_dir + "/*.png"))

        self.img_side_right_mask_dir = os.path.join(self.img_mask_dir, self.cam_side_right_topic_name)
        self.img_side_right_mask_files = sorted(glob.glob(self.img_side_right_mask_dir + "/*.png"))

        # load calib and gt poses
        self.read_transform(os.path.join(data_dir, "transform.json"))

    def __getitem__(self, idx):
        
        points = self.read_point_cloud(self.lidar_top_files[idx]) # in body frame

        if not self.use_only_lidar_top: # but all in body frame
            points_front = self.read_point_cloud(self.lidar_front_files[idx])
            points_rear = self.read_point_cloud(self.lidar_rear_files[idx])
            points_left = self.read_point_cloud(self.lidar_side_left_files[idx])
            points_right = self.read_point_cloud(self.lidar_side_right_files[idx])

            points = np.concatenate((points, points_front), axis=0) 
            points = np.concatenate((points, points_rear), axis=0) 
            points = np.concatenate((points, points_left), axis=0) 
            points = np.concatenate((points, points_right), axis=0) 

        # convert to center top lidar frame
        points_homo = np.hstack((points[:,:3], np.ones((np.shape(points)[0], 1))))

        points_homo_lidar_frame = points_homo @ np.linalg.inv(self.lidar_top_extrinsic).T # T_l_b.T

        points_homo[:,:3] = points_homo_lidar_frame[:,:3]

        points = points_homo

        if not self.load_img:
            frame_data = {"points": points}
            return frame_data

        # load img
        
        # follow IPB car loader
        img_dict = {}
        depth_img_dict = {}

        if self.main_cam_only:
            img_front = self.read_img(self.img_front_files[idx])
            img_dict = {self.cam_front_topic_name: img_front}
        else:
            img_front = self.read_img(self.img_front_files[idx])
            img_front_left = self.read_img(self.img_front_left_files[idx])
            img_front_right = self.read_img(self.img_front_right_files[idx])
            img_side_left = self.read_img(self.img_side_left_files[idx])
            img_side_right = self.read_img(self.img_side_right_files[idx])

            img_dict = {self.cam_side_left_topic_name: img_side_left,
                        self.cam_front_left_topic_name: img_front_left, 
                        self.cam_front_topic_name: img_front,
                        self.cam_front_right_topic_name: img_front_right,
                        self.cam_side_right_topic_name: img_side_right}

        points_rgb = np.ones_like(points) # N,4, last channel for the mask
        
        for cam_name in list(img_dict.keys()):
            points_rgb, depth_map = self.project_points_to_cam(points, points_rgb, img_dict[cam_name], 
                                                    self.T_c_l_mats[cam_name], self.K_mats[cam_name])

            # img_dict[cam_name] = np.concatenate((img_dict[cam_name], np.expand_dims(depth_map, axis=-1)), axis=-1) # 4 channels

        if self.use_only_colorized_points:
            with_rgb_mask = (points_rgb[:, 3] == 0)
            points = points[with_rgb_mask]
            points_rgb = points_rgb[with_rgb_mask]

        # # # we skip the intensity here for now (and also the color mask)
        points = np.hstack((points[:,:3], points_rgb[:,:3]))

        frame_data = {"points": points, "img": img_dict, "depth": depth_img_dict}

        return frame_data

    def __len__(self):
        return len(self.lidar_top_files)
    
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

    def read_point_cloud(self, scan_file: str):
        points = np.asarray(o3d.io.read_point_cloud(scan_file).points, dtype=np.float64)
        return points.astype(np.float64)
    
    def read_img(self, img_file: str):
        img = cv2.imread(img_file)
        # print(img.shape)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = np.array(img) # as np array
        
        return img

    def read_transform(self, json_file_path: str) -> dict:
        # transforms_dict = {}
        # self.gt_poses = []

        with open(json_file_path, 'r') as infile: # load intrinsic json file
            transforms_dict = json.load(infile)

            sensor_params = transforms_dict["sensor_params"]
            lidar_top_params = sensor_params["lidar_TOP"]
            self.lidar_top_extrinsic = np.array(lidar_top_params["extrinsic"]) # T_b_l

            for cam_name in self.cam_names:
                cam_params = sensor_params[cam_name]
                self.K_mats[cam_name] = np.array(cam_params["camera_intrinsic"])
                cam_extrinsic = np.array(cam_params["extrinsic"]) # T_b_c
                self.T_c_l_mats[cam_name] = np.linalg.inv(cam_extrinsic) @ self.lidar_top_extrinsic # T_c_l

        main_cam_K_mat = self.K_mats[self.main_cam_name]
        main_cam_extrinsic = self.T_c_l_mats[self.main_cam_name]

        self.intrinsic = o3d.camera.PinholeCameraIntrinsic()
        self.intrinsic.set_intrinsics(height=1280,
                                        width=1920,
                                        fx=main_cam_K_mat[0,0],
                                        fy=main_cam_K_mat[1,1],
                                        cx=main_cam_K_mat[0,2],
                                        cy=main_cam_K_mat[1,2])
        self.extrinsic = main_cam_extrinsic

        self.mono_depth_for_high_z: bool = True

        # gt poses still have some problem 
        #     frame_trans = transforms_dict["frames"]
        #     for frame_meta in frame_trans:
        ## you need to select the camera or lidar, and figure out what's the transformation is refered to
        #         T_w_b = np.array(frame_meta["transform_matrix"])
        #         T_w_l = T_w_b @ self.lidar_top_extrinsic
        #         self.gt_poses.append(T_w_l)

        # self.gt_poses = np.array(self.gt_poses)

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
