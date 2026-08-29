#!/usr/bin/env python3
# @file      slam_dataset.py
# @author    Yue Pan     [yue.pan@igg.uni-bonn.de]
# Copyright (c) 2024 Yue Pan, all rights reserved

import csv
import math
import os
import sys
from pathlib import Path
from typing import List

import cv2
import datetime as dt
import matplotlib.cm as cm
import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
import wandb
from numpy.linalg import inv
from rich import print
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image

# import vdbfusion # for debugging, comparison with the baseline

from dataset.dataloaders import dataset_factory
from eval.eval_traj_utils import absolute_error, plot_trajectories, relative_error
from utils.config import Config
from utils.tools import (
    colorize_depth_maps,
    deskewing,
    get_time,
    plot_timing_detail,
    tranmat_close_to_identity,
    transform_torch,
    voxel_down_sample_torch,
    project_points_to_cam_torch,
    rotmat_to_degree_np,
    slerp_pose,
    sem_kitti_color_map
)

from gaussian_splatting.utils.cameras import CamImage

class SLAMDataset():
    def __init__(self, config: Config) -> None:

        super().__init__()

        self.config = config
        self.silence = config.silence
        self.dtype = config.dtype
        self.device = config.device
        self.run_path = config.run_path

        max_frame_number: int = 1000000

        self.monodepth_on: bool = False
        if config.gs_on:
            self.monodepth_on = config.monodepth_on

        self.poses_ts = None # timestamp for each reference pose, also as np.array
        self.gt_poses = None
        self.calib = {"Tr": np.eye(4), "T_l_c": np.eye(4)} # as T_lidar<-body (cam)
        # "Tr" is used for KITTI, as the reference pose is not in LiDAR frame

        self.T_l_lm_list = None # inter-LiDAR extrinsic (when there are multiple lidars)
        
        self.K_mats = None 
        self.cam_widths = None
        self.cam_heights = None
        self.T_c_l_mats = None # camera-LiDAR extrinsic

        self.is_rgbd: bool = False # by default, lidar dataset

        self.cam_valid_v_ratios_minmax = None # mask for valid vertical FOV

        self.loader = dataset_factory(
            dataloader=config.data_loader_name, # a specific dataset or data format
            data_dir=Path(config.pc_path),
            sequence=config.data_loader_seq,
            topic=config.data_loader_seq,
            cam_name=config.data_loader_seq,
            lidar_name=config.data_loader_seq,
        )
        config.end_frame = min(len(self.loader), config.end_frame)
        used_frame_count = int((config.end_frame - config.begin_frame) / config.step_frame)
        self.total_pc_count = used_frame_count
        max_frame_number = self.total_pc_count
        if hasattr(self.loader, 'gt_poses'):
            self.gt_poses = self.loader.gt_poses[config.begin_frame:config.end_frame:config.step_frame]
            self.gt_pose_provided = True

            # write_kitti_format_poses(os.path.join(self.run_path, "gt_poses"), self.gt_poses)
            # write_traj_as_o3d(self.gt_poses, os.path.join(self.run_path, "gt_poses.ply"))
        else:
            self.gt_pose_provided = False
        if hasattr(self.loader, 'calibration'):
            self.calib["Tr"][:3, :4] = self.loader.calibration["Tr"].reshape(3, 4)
        if hasattr(self.loader, "K_mats"): # as dictionary
            self.K_mats = self.loader.K_mats
            self.cam_names = list(self.K_mats.keys())
        if hasattr(self.loader, "T_c_l_mats"):
            self.T_c_l_mats = self.loader.T_c_l_mats # as dictionary
            # print(self.T_c_l_mats)
        if hasattr(self.loader, "cam_widths"):
            self.cam_widths = self.loader.cam_widths
            self.cam_heights = self.loader.cam_heights
        if hasattr(self.loader, "T_l_lm_mats"):
            if len(self.loader.T_l_lm_mats) > 0:
                self.T_l_lm_list = self.loader.T_l_lm_mats # as list
        if hasattr(self.loader, "intrinsics_o3d"):
            if len(self.loader.intrinsics_o3d) > 0:
                self.intrinsics_o3d = self.loader.intrinsics_o3d
        if hasattr(self.loader, "is_rgbd"):
            self.is_rgbd = self.loader.is_rgbd
        if hasattr(self.loader, "deskew_off"):
            self.config.deskew = False
        if hasattr(self.loader, "contains_image"):
            if not self.loader.contains_image and self.config.gs_on:
                print("Warning: the dataset does not contain images, degenerate to lidar-only SLAM (PIN-SLAM)")
                self.config.gs_on = False
        if hasattr(self.loader, "cam_valid_v_ratios_minmax"):
            self.cam_valid_v_ratios_minmax = self.loader.cam_valid_v_ratios_minmax

        if config.color_channel == 3:
            self.loader.load_img = True
            if self.monodepth_on:
                # NOTE: For dataset where the LiDAR only cover a very small portion of the FOV when projected to the image (like KITTI), 
                # we can use the monodepth estimator to predict the depth and align it with the LiDAR measurements using least square fitting
                # However, this is not among the main scope of this project
                print("Monocular depth estimation is on, load Metric3D model")
                # load metric3d model for mono depth estimation
                model_name = 'metric3d_vit_small' # faster, 80 ms
                # model_name = 'metric3d_vit_large' # slower, 450 ms

                # use torch hub for model loading
                self.metric3d = torch.hub.load('yvanyin/metric3d', model_name, pretrain=True).to(self.device).eval()
                # NOTE: you need to install install mmcv and timm for using metric3d
                # install xformers for faster inference
        
        # use pre-allocated numpy array
        self.odom_poses = None
        if config.track_on:
            self.odom_poses = np.broadcast_to(np.eye(4), (max_frame_number, 4, 4)).copy() # T_wi

        self.pgo_poses = None
        if config.pgo_on:
            self.pgo_poses = np.broadcast_to(np.eye(4), (max_frame_number, 4, 4)).copy() # T_wi

        self.travel_dist = np.zeros(max_frame_number) 
        self.accu_travel_dist: float = 0.0
        self.accu_travel_dist_for_keyframe: float = 0.0
        self.accu_travel_degree_for_keyframe: float = 0.0
        
        self.time_table = []

        self.processed_frame: int = 0
        self.shift_ts: float = 0.0
        self.lose_track: bool = False  # the odometry lose track or not (for robustness)
        self.consecutive_lose_track_frame: int = 0
        self.color_available: bool = False
        self.intensity_available: bool = False
        self.color_scale: float = 255.0
        self.last_pose_ref = np.eye(4)
        self.last_odom_tran = np.eye(4)
        self.cur_pose_ref = np.eye(4)
        # count the consecutive stop frame of the robot
        self.stop_count: int = 0
        self.stop_status = False

        self.loop_reg_failed_count: int = 0

        if self.config.kitti_correction_on:
            self.last_odom_tran[0, 3] = (
                self.config.max_range * 1e-2
            )  # inital guess for booting on x aixs
            self.color_scale = 1.0
            self.config.deskew_ref_ratio = 0.5

        self.last_odom_tran_torch = torch.tensor(self.last_odom_tran, device=self.device, dtype=self.dtype)

        # current frame point cloud (for visualization)
        self.cur_frame_o3d = o3d.geometry.PointCloud()
        # current frame bounding box in the world coordinate system
        self.cur_bbx = o3d.geometry.AxisAlignedBoundingBox()
        # merged downsampled point cloud (for visualization)
        self.map_down_o3d = o3d.geometry.PointCloud()
        # map bounding box in the world coordinate system
        self.map_bbx = o3d.geometry.AxisAlignedBoundingBox()
        # current frame mono depth predicted point cloud
        self.cur_frame_mono_depth_o3d = o3d.geometry.PointCloud()

        self.static_mask = None

        # adaptive resolution
        self.crop_max_range = self.config.max_range
        self.train_voxel_m = self.config.vox_down_m
        self.source_voxel_m = self.config.source_vox_down_m

        # initialize data for temp frame
        self.init_temp_data()

        self.ts_ref_ratio_diffs = None


    def init_temp_data(self):
        
        # current frame's data
        self.cur_point_cloud_torch = None
        self.cur_point_ts_torch = None
        self.cur_sem_labels_torch = None
        self.cur_sem_labels_full = None
        self.cur_point_normals = None
        self.cur_point_lidar_idx_torch = None # now only used for deskewing multiple-LiDARs

        self.cur_point_cloud_mono_depth = None # point cloud results from image mono (metric) depth estimation
        self.cur_point_normals_mono_depth = None
        
        # source data for registration
        self.cur_source_points = None
        self.cur_source_normals = None
        self.cur_source_colors = None

        # img data
        self.cur_cam_img = None # dict of CamImage

        # imu data
        self.cur_frame_imus = None

        # sensor timestamp
        self.cur_sensor_ts = None


    # read frame with specific data loader (partially borrow from kiss-icp: https://github.com/PRBonn/kiss-icp)
    def read_frame_with_loader(self, frame_id, init_pose: bool = True, 
        use_image: bool = True, monodepth_on: bool = False):
        
        if init_pose:
            self.set_ref_pose(frame_id)

        frame_id_in_folder = self.config.begin_frame + frame_id * self.config.step_frame
        frame_data = self.loader[frame_id_in_folder]

        points = None
        point_ts = None
        point_lidar_idx = None
        img_dict = None
        depth_dict = None
        imus = None

        # this might still be slow
        tic_0 = get_time()

        if hasattr(self.loader, 'ts_ref_ratio_diffs'): # the ts difference between other lidar and the main lidar (dirty fix)
            self.ts_ref_ratio_diffs = self.loader.ts_ref_ratio_diffs

        if isinstance(frame_data, dict):
            dict_keys = list(frame_data.keys())
            # if not self.silence:
            #     print("Available data source:", dict_keys)
            if "points" in dict_keys:
                points = frame_data["points"] # may also contain intensity or color
            if "point_ts" in dict_keys:
                point_ts = frame_data["point_ts"]
            if "point_lidar_idx" in dict_keys: # the point belong to which lidar, now we support the multi-lidar system
                point_lidar_idx = frame_data["point_lidar_idx"]
            if "imus" in dict_keys:
                self.cur_frame_imus = frame_data["imus"]
                # not implemented in PINGS
            if "sensor_ts" in dict_keys:
                self.cur_sensor_ts = frame_data["sensor_ts"]
            if "img" in dict_keys and use_image: # we support multiple cameras
                img_dict: dict = frame_data["img"]
                cam_list = list(img_dict.keys())
                self.cur_cam_img = {}
                
                # have depth img, if not, we will project LiDAR measurements to the image to get depth
                if "depth" in dict_keys: 
                    depth_dict: dict = frame_data["depth"]
                else:
                    depth_dict = None

                if "sky" in dict_keys: # have sky mask
                    sky_dict: dict = frame_data["sky"]
                else:
                    sky_dict = None

                if "obj_mask" in dict_keys: # have object mask (cookie cutter, no supervision on object pixels)
                    obj_mask_dict: dict = frame_data["obj_mask"]
                else:
                    obj_mask_dict = None

                pred_pcd_merged = o3d.geometry.PointCloud()

                for cam_name in cam_list:
                    
                    tic_load_cam = get_time()

                    cur_img_rgb_np = img_dict[cam_name] # 3 channel (rgb only) # uint8 [0, 255]

                    cur_img_depth_np = None
                    cur_img_depth = None
                    if depth_dict is not None:
                        cur_img_depth_np = depth_dict[cam_name] # H, W, 1

                        cur_img_depth = torch.tensor(cur_img_depth_np, dtype=self.dtype, device=self.device) # unit: m
                        cur_img_depth = cur_img_depth.permute(2,0,1) # 1, H, W
                        # cur_img_rgb_np = cur_img_np[:,:,:3].astype(np.uint8) # [0,255]

                        cur_img_depth_np = np.squeeze(cur_img_depth_np) # H, W

                    sky_mask = None # optional sky mask (sky:1, non-sky:0)
                    if sky_dict is not None:
                        sky_mask = torch.tensor(sky_dict[cam_name], dtype=torch.bool, device=self.device) #
                        sky_mask = sky_mask.permute(2,0,1) # 1, H, W

                    obj_mask = None # optional object mask (object:1, background:0)
                    if obj_mask_dict is not None and cam_name in obj_mask_dict:
                        obj_mask = torch.tensor(obj_mask_dict[cam_name], dtype=torch.bool, device=self.device) #
                        obj_mask = obj_mask.permute(2,0,1) # 1, H, W
                    
                    # cur_img_rgb_torch = torch.tensor(cur_img_rgb_np, dtype=self.dtype, device=self.device) 
                    cur_img_rgb_torch = torch.from_numpy(cur_img_rgb_np).float().to(self.device)
                    cur_img_rgb_torch = cur_img_rgb_torch.permute(2,0,1) # 3, H, W
                    cur_img_rgb_torch /= 255.0 # convert RGB channel to [0,1]
                    # print(cur_img[3])

                    H, W = cur_img_rgb_torch.shape[1], cur_img_rgb_torch.shape[2]

                    pred_normal = None # optional normal image 

                    toc_load_cam = get_time()
                    

                    # NOTE: For dataset where the LiDAR only cover a very small portion of the FOV when projected to the image (like KITTI), 
                    # we can use the monodepth estimator to predict the depth and align it with the LiDAR measurements using least square fitting
                    # However, this is not among the main scope of this project
                    # TODO: support multi-camera, add to another function
                    if monodepth_on:
                        
                        use_mono_depth_for_gs_init = True
                        if self.is_rgbd:
                            use_mono_depth_for_gs_init = False

                        mono_depth_input_rgb = cur_img_rgb_torch
                        
                        # down-sample input image to save computation (otherwise it will take more than 100ms)
                        if H*W > 8e5:
                            mono_depth_input_rgb = F.interpolate(cur_img_rgb_torch.unsqueeze(0), scale_factor=0.5, mode='bilinear', align_corners=False).squeeze(0)
                            # For kitti, if downsized, computational time can be decrease to 20ms on my GPU

                        tic_metric3d = get_time()

                        with torch.no_grad():
                            # pred_depth, confidence, output_dict = self.metric3d.inference({'input': rgb})
                            pred_depth, confidence, output_dict = self.metric3d.inference({'input': mono_depth_input_rgb.unsqueeze(0)}) #B,C,H,W 
                        
                        toc_metric3d = get_time()
                        if not self.silence:
                            print("Metric3D prediction time     (ms):", (toc_metric3d-tic_metric3d)*1e3)  

                        pred_normal = output_dict['prediction_normal'][:, :3, :, :] # only available for Metric3Dv2 i.e., ViT models  # 1, 3, H, W
                        # normal_confidence = output_dict['prediction_normal'][:, 3, :, :] # see https://arxiv.org/abs/2109.09881 for details  # 1, H, W

                        # interpolate to the original image size
                        pred_depth = F.interpolate(pred_depth, size=(H, W), mode='bilinear', align_corners=False).squeeze(0) # 1, H, W
                        # confidence = F.interpolate(confidence, size=(H, W), mode='bilinear', align_corners=False).squeeze(0) # 1, H, W

                        pred_normal = F.interpolate(pred_normal, size=(H, W), mode='bilinear', align_corners=False).squeeze(0)  # 3, H, W

                        # pred_depth[invalid_mask] = 0 
                        pred_depth_np = pred_depth.permute(1,2,0).squeeze(-1).detach().cpu().numpy() # H, W
                        # pred_depth_np[sky_mask_np] = 0.0 # mask sky part

                        if use_mono_depth_for_gs_init and cur_img_depth_np is not None:
                            valid_depth_mask = (cur_img_depth_np > self.config.min_range) & (pred_depth_np > self.config.min_range) & (cur_img_depth_np < self.config.max_range*0.8)
                            valid_depth_measurement = cur_img_depth_np[valid_depth_mask]
                            valid_depth_count = np.shape(valid_depth_measurement)[0]
                            # print(valid_depth_count)

                            pred_depth_with_gt = pred_depth_np[valid_depth_mask]
                            residual_before = pred_depth_with_gt - valid_depth_measurement
                            rmse_before = np.sqrt((np.mean(residual_before**2)))
                            if not self.silence:
                                print("mono depth rmse (m): ", rmse_before) # RMSE (m)
                            
                            # depth least square fitting with regards to the lidar measurement
                            if valid_depth_count > 100: # at least some valid measurements available
                                coefficients, residuals, _, _, _  = np.polyfit(pred_depth_with_gt, valid_depth_measurement, 1, full=True)
                                k, b = coefficients
                                rmse_after = np.sqrt((residuals[0]/valid_depth_count))
                                if not self.silence:
                                    print("depth fitting rmse (m): ", rmse_after) # RMSE (m)
                                pred_depth_np = k * pred_depth_np + b
                            else:
                                use_mono_depth_for_gs_init = False
                                

                        toc_lidar_align = get_time()

                        # after the alignment, get the sky mask (monodepth depth-threshold heuristic)
                        sky_dist_thre = self.config.max_range * 1.2
                        sky_mask_np = pred_depth_np > sky_dist_thre
                        pred_depth_np[sky_mask_np] = 0.0
                        sky_mask = torch.tensor(sky_mask_np, dtype=torch.bool, device=self.device).unsqueeze(0) # 1, H, W             

                        # visualize 
                        if self.config.o3d_vis_on and self.config.vis_in_cv2:
                            cur_img_rgb_vis = cv2.cvtColor(cur_img_rgb_np, cv2.COLOR_RGB2BGR) 
                            cv2.imshow("Mono RGB", cur_img_rgb_vis)

                        if self.config.o3d_vis_on and self.config.vis_in_cv2:
                            pred_depth_color = (colorize_depth_maps(pred_depth_np, 0.1, self.config.max_range)*255.0).astype(np.uint8) # 1, 3, H, W 
                            pred_depth_color = np.transpose(pred_depth_color[0], (1, 2, 0)) # H, W, 3
                            pred_depth_color = cv2.cvtColor(pred_depth_color, cv2.COLOR_RGB2BGR) # for vis
                            cv2.imshow("Mono Depth", pred_depth_color)

                            cv2.waitKey(1)
                
                        # pred_normal[invalid_mask] = 0
                        # print(pred_normal)
                        pred_normal_np = pred_normal.permute(1,2,0).detach().cpu().numpy() # H, W, 3 # [-1, 1]
                        # print(pred_normal_np)
                        pred_normal_vis_np = ((0.5 - pred_normal_np * 0.5) * 255.0).astype(np.uint8) # convert to the normal vis color # surf_normal
                        pred_normal_vis_np[sky_mask_np] = 0
                        pred_normal_vis_np = np.ascontiguousarray(pred_normal_vis_np) 
                        if self.config.o3d_vis_on and self.config.vis_in_cv2:
                            pred_normal_vis_cv2 = cv2.cvtColor(pred_normal_vis_np, cv2.COLOR_RGB2BGR) # in current camera frame
                            cv2.imshow("Mono Normal", pred_normal_vis_cv2)

                        # show sky mask
                        # cur_img_rgb_np[sky_mask_np] = 0  
                        # cur_img_rgb_np[invalid_mask_np] = 0  
                        # cur_img_rgb_cvshow = cv2.cvtColor(cur_img_rgb_np, cv2.COLOR_RGB2BGR)
                        # # cv2.imshow(" Sky mask", cur_img_rgb_cvshow)
                        # cv2.imshow(" Invalid mask", cur_img_rgb_cvshow)
                        # FIXME
                        if self.loader.mono_depth_for_high_z:
                            pred_depth_np[(pred_depth_np > 0.8*self.config.max_range)] = 0.0 # remove too far away estimations
                            pred_depth_np[(pred_depth_np < 2.0*self.config.min_range)] = 0.0
                        else:
                            pred_depth_np[pred_depth_np < 0.2*self.config.max_range] = 0.0 # remove close estimations

                        cur_img_o3d = o3d.geometry.Image(cur_img_rgb_np)
                        pred_depth_o3d = o3d.geometry.Image(pred_depth_np)
                        cur_normal_o3d = o3d.geometry.Image(pred_normal_vis_np)

                        # change the depth trunc here
                        # rgb-d
                        rgbd_image_o3d = o3d.geometry.RGBDImage.create_from_color_and_depth(cur_img_o3d, pred_depth_o3d, 
                            depth_scale=1.0, depth_trunc= self.config.local_map_radius, convert_rgb_to_intensity=False)                                                     
                        pred_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                            rgbd_image_o3d, self.intrinsics_o3d[cam_name], self.T_c_l_mats[cam_name])
                        

                        toc_rgbd2pcd = get_time()

                        # downsample and filtering noise
                        pred_pcd = pred_pcd.voxel_down_sample(voxel_size=self.config.vox_down_m*1)

                        pred_pcd_merged += pred_pcd

                        toc_pcd_filtering = get_time()

                        # print("LiDAR align time            (ms):", (toc_lidar_align-toc_metric3d)*1e3)  # ||
                        # print("Point cloud generation time (ms):", (toc_rgbd2pcd-toc_lidar_align)*1e3)  # ||||
                        # print("Points filtering time       (ms):", (toc_pcd_filtering-toc_rgbd2pcd)*1e3)# ||| 

                    # set camera frame data
                    self.cur_cam_img[cam_name] = CamImage(frame_id, cur_img_rgb_torch, self.K_mats[cam_name],
                                                          self.config.min_range*0.5, self.config.local_map_radius*1.1,
                                                          cam_name, depth_image=cur_img_depth, normal_img=pred_normal,
                                                          sky_mask=sky_mask, obj_mask=obj_mask, device=self.device)
                    
                if monodepth_on and use_mono_depth_for_gs_init:
                    self.cur_frame_mono_depth_o3d = pred_pcd_merged # also may conatin normals

                    points_xyz = np.array(pred_pcd_merged.points, dtype=np.float64)
                    points_rgb = np.array(pred_pcd_merged.colors, dtype=np.float64) # [0-1]
                    points_xyzrgb = np.hstack((points_xyz, points_rgb))
                    # points_normals = np.array(pred_pcd.normals, dtype=np.float64) # uncomment this

                    self.cur_point_cloud_mono_depth = torch.tensor(points_xyzrgb, device=self.device, dtype=self.dtype)
                    # self.cur_point_normals_mono_depth = torch.tensor(points_normals, device=self.device, dtype=self.dtype) # uncomment this

                    #print("Time for loading camera {:.2f} (ms)".format((toc_load_cam-tic_load_cam)*1e3))
                    #print("Time for setting camera {:.2f} (ms)".format((toc_set_cam-tic_set_cam)*1e3))
        
        toc_0 = get_time()
        # print("Time for preprocessing input data to camera {:.2f} (ms)".format((toc_0-tic_0)*1e3))

        if points is not None:
            self.cur_point_cloud_torch = torch.tensor(points, device=self.device, dtype=self.dtype)

        if self.config.deskew: 
            self.get_point_ts(point_ts)

        if point_lidar_idx is not None:
            self.cur_point_lidar_idx_torch = torch.tensor(point_lidar_idx, device=self.device, dtype=int)

    
    def set_ref_pose(self, frame_id):
        # load gt pose if available
        if self.gt_pose_provided:
            self.cur_pose_ref = self.gt_poses[frame_id]
        else:  # or initialize with identity
            self.cur_pose_ref = np.eye(4)
        self.cur_pose_torch = torch.tensor(
            self.cur_pose_ref, device=self.device, dtype=self.dtype
        )

    def preprocess_frame(self): 
        """
        proprocessing main function: all the preprocessing steps for a input point cloud
        """  
        # T1 = get_time()

        # setup poses
        valid_frame_flag = self.initialize_pose()
        if not valid_frame_flag:   
            return False

        # set adaptive downsampling voxel size 
        if self.config.adaptive_range_on:
            self.set_adaptive_resolution()

        # T2 = get_time()

        # preprocessing, filtering, kitti intrinsic correction
        self.filter_and_correct()

        # T3 = get_time()

        # prepare for the registration (from the second frame)
        if self.processed_frame > 0:
            self.preprocess_source_points()
            
        # T4 = get_time()
        return True

    def initialize_pose(self):
        """
        initialize the poses, return value indicates whether the frame is valid
        """  

        frame_id = self.processed_frame
        cur_pose_init_guess = self.cur_pose_ref
        if frame_id == 0:  # initialize the first frame, no tracking yet
            if self.config.track_on:
                self.odom_poses[frame_id] = self.cur_pose_ref
            if self.config.pgo_on:
                self.pgo_poses[frame_id] = self.cur_pose_ref
            self.travel_dist[frame_id] = 0.0
            self.last_pose_ref = self.cur_pose_ref
        elif frame_id > 0:
            # pose initial guess
            # last_translation = np.linalg.norm(self.last_odom_tran[:3, 3])
            if self.config.uniform_motion_on and not self.lose_track: 
            # if self.config.uniform_motion_on:   
                # apply uniform motion model here
                cur_pose_init_guess = (
                    self.last_pose_ref @ self.last_odom_tran
                )  # T_world<-cur = T_world<-last @ T_last<-cur
            else:  # static initial guess
                cur_pose_init_guess = self.last_pose_ref

            if not self.config.track_on and self.gt_pose_provided:
                cur_pose_init_guess = self.gt_poses[frame_id]

            # pose initial guess tensor
            self.cur_pose_guess_torch = torch.tensor(
                cur_pose_init_guess, dtype=torch.float64, device=self.device
            )   

        if self.cur_point_cloud_torch is not None:
            original_count = self.cur_point_cloud_torch.shape[0]
            if original_count < 10:  # deal with missing data (invalid frame)
                print("[bold red]Not enough input point cloud, skip this frame[/bold red]")
                if self.config.track_on:
                    self.odom_poses[frame_id] = cur_pose_init_guess
                if self.config.pgo_on:
                    self.pgo_poses[frame_id] = cur_pose_init_guess
                return False # indicating invalid frame
        else:
            return False
        
        return True


    def filter_and_correct(self):
        """
        filter or crop the measured point cloud, possibly with the LiDAR intrinsic correction
        """  
        # preprocessing, filtering
        if self.cur_sem_labels_torch is not None:
            filter_idx = filter_sem_kitti(
                self.cur_point_cloud_torch,
                self.cur_sem_labels_full,
                True,
                self.config.filter_moving_object,
            )
        else:
            filter_idx = crop_frame(
                self.cur_point_cloud_torch,
                self.config.min_z,
                self.config.max_z,
                self.config.min_range,
                self.crop_max_range,
                self.config.range_filter_2d and (not self.is_rgbd),
            )

        self.cur_point_cloud_torch = self.cur_point_cloud_torch[filter_idx]
        
        if self.cur_point_ts_torch is not None:
            self.cur_point_ts_torch = self.cur_point_ts_torch[filter_idx]
        
        if self.cur_point_lidar_idx_torch is not None:
            self.cur_point_lidar_idx_torch = self.cur_point_lidar_idx_torch[filter_idx]

        if self.cur_sem_labels_torch is not None:
            self.cur_sem_labels_torch = self.cur_sem_labels_torch[filter_idx]
        
        # kitti intrinsic correction
        if self.config.kitti_correction_on:
            self.cur_point_cloud_torch = intrinsic_correct(
                self.cur_point_cloud_torch, self.config.correction_deg
            )
    

    def set_adaptive_resolution(self):
        """
        set resolution adpatively according to the measured scan point cloud
        """  
        pc_max_bound, _ = torch.max(self.cur_point_cloud_torch[:, :3], dim=0)
        pc_min_bound, _ = torch.min(self.cur_point_cloud_torch[:, :3], dim=0)

        min_x_range = min(torch.abs(pc_max_bound[0]), torch.abs(pc_min_bound[0]))
        min_y_range = min(torch.abs(pc_max_bound[1]), torch.abs(pc_min_bound[1]))
        max_x_y_min_range = max(min_x_range, min_y_range)

        self.crop_max_range = min(self.config.max_range, 2.0 * max_x_y_min_range)

        self.train_voxel_m = (
            self.crop_max_range / self.config.max_range
        ) * self.config.vox_down_m
        self.source_voxel_m = (
            self.crop_max_range / self.config.max_range
        ) * self.config.source_vox_down_m


    def preprocess_source_points(self):
        """
        downsampling the points for registration (source point cloud) and doing the 
        pre-deskewing using the uniform motion model
        """  

        # used for registration
        cur_source_torch = self.cur_point_cloud_torch.clone()
        
        # source point voxel downsampling (for registration)
        idx = voxel_down_sample_torch(cur_source_torch[:, :3], self.source_voxel_m)
        cur_source_torch = cur_source_torch[idx]
        self.cur_source_points = cur_source_torch[:, :3]
        if self.config.color_channel == 1 or self.config.color_channel == 3:
            self.cur_source_colors = cur_source_torch[:, 3:]

        if self.cur_point_ts_torch is not None:
            cur_source_ts = self.cur_point_ts_torch.clone()
            cur_source_ts = cur_source_ts[idx]
        else:
            cur_source_ts = None

        if self.cur_point_normals is not None:
            self.cur_source_normals = self.cur_point_normals[idx] # invalid part as 0

        if self.cur_point_lidar_idx_torch is not None:
            cur_source_lidar_idx = self.cur_point_lidar_idx_torch.clone()
            cur_source_lidar_idx = cur_source_lidar_idx[idx]
        else:
            cur_source_lidar_idx = None

        # deskewing (motion undistortion) for source point cloud
        if self.config.deskew and not self.lose_track:
            self.cur_source_points = deskewing(
                self.cur_source_points,
                cur_source_ts,
                self.last_odom_tran_torch,
                ts_ref_pose = self.config.deskew_ref_ratio,
                points_lidar_idx = cur_source_lidar_idx,
                T_l_lm_list=self.T_l_lm_list,
                ts_diff_list=self.ts_ref_ratio_diffs 
            )  # T_last<-cur

        # print("# Source point for registeration : ", cur_source_torch.shape[0])

    
    def update_odom_pose(self, cur_pose_torch: torch.tensor): 
        """
        Done after odometry to setup the latest poses, do the deskewing of raw lidar points and
        project to camera frames to generate colorized point cloud and depth map 
        """    

        cur_frame_id = self.processed_frame
        # needed to be at least the second frame
        assert (cur_frame_id > 0), "This function needs to be used from at least the second frame"

        # need to be out of the computation graph, used for mapping
        self.cur_pose_torch = cur_pose_torch.detach()
            
        self.cur_pose_ref = self.cur_pose_torch.cpu().numpy()

        self.last_odom_tran = inv(self.last_pose_ref) @ self.cur_pose_ref  # T_last<-cur # the odometry result instead of the inital guess

        # here we consider both rot and tran
        if tranmat_close_to_identity(
            self.last_odom_tran, 1e-3, self.config.voxel_size_m * 0.1
        ):
            self.stop_count += 1
        else:
            self.stop_count = 0

        if self.stop_count > self.config.stop_frame_thre:
            self.stop_status = True
            if not self.silence:
                print("Robot stopped")
        else:
            self.stop_status = False

        if self.config.pgo_on:  # initialize the pgo pose
            self.pgo_poses[cur_frame_id] = self.cur_pose_ref

        if self.odom_poses is not None:
            cur_odom_pose = self.odom_poses[cur_frame_id-1] @ self.last_odom_tran  # T_world<-cur
            self.odom_poses[cur_frame_id] = cur_odom_pose

        cur_frame_travel_dist = np.linalg.norm(self.last_odom_tran[:3, 3])
        cur_frame_travel_degree = np.abs(rotmat_to_degree_np(self.last_odom_tran[:3, :3])) # 0-180

        if (
            cur_frame_travel_dist > self.config.surface_sample_range_m * 40.0
            and self.config.track_on
        ):  # too large translation in one frame --> lose track
            self.lose_track = True
            self.write_results() # record before the failure point
            sys.exit("Too large translation in one frame, system failed")

        # for GS training keyframes
        self.accu_travel_dist_for_keyframe += cur_frame_travel_dist
        self.accu_travel_degree_for_keyframe += cur_frame_travel_degree

        self.accu_travel_dist += cur_frame_travel_dist
        self.travel_dist[cur_frame_id] = self.accu_travel_dist
        if not self.silence:
            print("Accumulated travel distance (m): %f" % self.accu_travel_dist)
        
        self.last_pose_ref = self.cur_pose_ref  # update for the next frame

        self.last_odom_tran_torch = torch.tensor(self.last_odom_tran, device=self.device, dtype=self.dtype)

        # deskewing (motion undistortion using the estimated transformation) for the points for mapping
        if self.config.deskew and not self.lose_track:
            self.cur_point_cloud_torch = deskewing(
                self.cur_point_cloud_torch,
                self.cur_point_ts_torch,
                self.last_odom_tran_torch,
                ts_ref_pose = self.config.deskew_ref_ratio,
                points_lidar_idx = self.cur_point_lidar_idx_torch,
                T_l_lm_list=self.T_l_lm_list,
                ts_diff_list=self.ts_ref_ratio_diffs 
            )  # T_last<-cur

        if self.lose_track:
            self.consecutive_lose_track_frame += 1
        else:
            self.consecutive_lose_track_frame = 0

        if self.consecutive_lose_track_frame > 10:
            self.write_results() # record before the failure point
            sys.exit("Lose track for a long time, system failed") 


    def voxel_downsample_points_for_mapping(self):
        # downsampling the point for mapping now
        idx = voxel_down_sample_torch(self.cur_point_cloud_torch[:, :3], self.train_voxel_m)
        self.cur_point_cloud_torch = self.cur_point_cloud_torch[idx]
        if self.cur_point_ts_torch is not None:
            self.cur_point_ts_torch = self.cur_point_ts_torch[idx]
        if self.cur_sem_labels_torch is not None:
            self.cur_sem_labels_torch = self.cur_sem_labels_torch[idx]
            self.cur_sem_labels_full = self.cur_sem_labels_full[idx]

    # get the relative timestamp shift from the reference lidar frame as a ratio
    def get_cur_cam_ref_ts_ratio(self, cam_name, lidar_t_interval: float = 0.1):
        
        # FIXME (lidar_t_interval), deal with other cases, use last frame lidar ts

        if self.cur_sensor_ts is None:
            return None

        cur_cam_ts = self.cur_sensor_ts[cam_name]
        cur_main_lidar_ts = self.cur_sensor_ts[self.loader.main_lidar_name]

        cur_cam_ref_ts_ratio = (cur_cam_ts - (cur_main_lidar_ts - lidar_t_interval)) / lidar_t_interval

        # cur_cam_ref_ts_ratio = 1.0 - cur_cam_ref_ts_ratio

        return cur_cam_ref_ts_ratio

    def project_pointcloud_to_cams(self, use_only_colorized_points: bool = True, tran_in_frame = None):
        # done after deskewing
        # to get a refined depth map and colorized point cloud

        point_count = self.cur_point_cloud_torch.shape[0]
        point_channel = self.cur_point_cloud_torch.shape[1]

        points_rgb_torch = -1.0 * torch.ones((point_count, 4)).to(self.cur_point_cloud_torch) # set as invalid in the beginning

        if self.cur_cam_img is None:
            return

        cur_cam_names = list(self.cur_cam_img.keys())
        for cam_name in cur_cam_names:
            cam_img: CamImage = self.cur_cam_img[cam_name]

            if cam_img is None:
                continue

            cam_rgb_torch = cam_img.rgb_image_list[0] # without downsampling

            cur_T_c_l = torch.tensor(self.T_c_l_mats[cam_name], device=self.device, dtype=self.dtype)
            
            # relative transformation between the lidar reference timestamp and the camera triggering timestamp
            if tran_in_frame is not None and self.cur_sensor_ts is not None:
                
                cur_cam_ref_ts_ratio = self.get_cur_cam_ref_ts_ratio(cam_name)
                # print(cur_cam_ref_ts_ratio)

                diff_pose_l_c_ts = slerp_pose(tran_in_frame, cur_cam_ref_ts_ratio, self.config.deskew_ref_ratio).to(cur_T_c_l)

                cur_T_c_l = cur_T_c_l @ torch.linalg.inv(diff_pose_l_c_ts)
            
            cur_K_mat = torch.tensor(self.K_mats[cam_name], device=self.device, dtype=self.dtype)

            points_rgb_torch, depth_map_torch = project_points_to_cam_torch(self.cur_point_cloud_torch, points_rgb_torch, 
                cam_rgb_torch, cur_T_c_l, cur_K_mat, min_depth=self.config.min_range)

            cam_img.set_depth_img(depth_map_torch)
        
        # color channels
        self.cur_point_cloud_torch[:, 3:] = points_rgb_torch[:, :3] # 4-6 rgb

        if use_only_colorized_points:
            with_rgb_mask = (points_rgb_torch[:, 3] == 0)
            # print("# not valid count:", torch.sum(~with_rgb_mask).item())
            self.cur_point_cloud_torch = self.cur_point_cloud_torch[with_rgb_mask]
            if self.cur_point_ts_torch is not None:
                self.cur_point_ts_torch = self.cur_point_ts_torch[with_rgb_mask]
            if self.cur_point_normals is not None:
                self.cur_point_normals = self.cur_point_normals[with_rgb_mask]
            if self.cur_sem_labels_torch is not None:
                self.cur_sem_labels_torch = self.cur_sem_labels_torch[with_rgb_mask]
                self.cur_sem_labels_full = self.cur_sem_labels_full[with_rgb_mask]


    def update_poses_after_pgo(self, pgo_poses):
        self.pgo_poses[:self.processed_frame+1] = pgo_poses  # update pgo pose
        self.cur_pose_ref = self.pgo_poses[self.processed_frame]
        self.last_pose_ref = self.cur_pose_ref  # update for next frame

    def update_o3d_map(self):

        frame_down_torch = self.cur_point_cloud_torch 
        # self.cur_point_cloud_torch is in current lidar frame

        frame_o3d = o3d.geometry.PointCloud()
        if frame_down_torch is not None:
            frame_points_np = (
                frame_down_torch[:, :3].detach().cpu().numpy().astype(np.float64)
            )

            frame_o3d.points = o3d.utility.Vector3dVector(frame_points_np)
        else:
            return

        # visualize or not
        # uncomment to visualize the dynamic mask
        if (self.config.dynamic_filter_on) and (self.static_mask is not None) and (not self.stop_status):
            static_mask = self.static_mask.detach().cpu().numpy()
            frame_colors_np = np.ones_like(frame_points_np) * 0.7
            frame_colors_np[~static_mask, 1:] = 0.0
            frame_colors_np[~static_mask, 0] = 1.0
            frame_o3d.colors = o3d.utility.Vector3dVector(
                frame_colors_np.astype(np.float64)
            )

        # if self.cur_point_normals is not None:
        #     frame_normals_np = (
        #         self.cur_point_normals[:, :3].detach().cpu().numpy().astype(np.float64)
        #     )
        #     frame_o3d.normals = o3d.utility.Vector3dVector(frame_normals_np)
        #     frame_o3d.orient_normals_towards_camera_location()

        # transform from lidar frame to world frame
        frame_o3d = frame_o3d.transform(self.cur_pose_ref)

        if self.config.color_channel > 0:
            frame_colors_np = (
                frame_down_torch[:, 3:].detach().cpu().numpy().astype(np.float64)
            )
            if self.config.color_channel == 1:
                frame_colors_np = np.repeat(frame_colors_np.reshape(-1, 1), 3, axis=1)
            frame_o3d.colors = o3d.utility.Vector3dVector(frame_colors_np)
        elif self.cur_sem_labels_torch is not None:
            frame_label_torch = self.cur_sem_labels_torch
            frame_label_np = frame_label_torch.detach().cpu().numpy()
            frame_label_color = [
                sem_kitti_color_map[sem_label] for sem_label in frame_label_np
            ]
            frame_label_color_np = (
                np.asarray(frame_label_color, dtype=np.float64) / 255.0
            )
            frame_o3d.colors = o3d.utility.Vector3dVector(frame_label_color_np)
        
        self.cur_frame_o3d = frame_o3d
        if self.cur_frame_o3d.has_points():
            self.cur_bbx = self.cur_frame_o3d.get_axis_aligned_bounding_box()

        # transform mono depth point cloud
        # transform from lidar frame to world frame
        if self.monodepth_on and self.cur_frame_mono_depth_o3d.has_points(): 
            self.cur_frame_mono_depth_o3d = self.cur_frame_mono_depth_o3d.transform(self.cur_pose_ref)

        cur_max_z = self.cur_bbx.get_max_bound()[-1]
        cur_min_z = self.cur_bbx.get_min_bound()[-1]

        bbx_center = self.cur_pose_ref[:3, 3]
        bbx_min = np.array(
            [
                bbx_center[0] - self.config.max_range,
                bbx_center[1] - self.config.max_range,
                cur_min_z,
            ]
        )
        bbx_max = np.array(
            [
                bbx_center[0] + self.config.max_range,
                bbx_center[1] + self.config.max_range,
                cur_max_z,
            ]
        )

        self.cur_bbx = o3d.geometry.AxisAlignedBoundingBox(bbx_min, bbx_max)

        # use the downsampled neural points here (done outside the class)

    def get_tran_in_frame(self, frame_id, use_gt_pose: bool = False):
        
        assert frame_id > 0, "frame_id needs to be larger than 0, because we use frame_id-1 here"

        cur_frame = frame_id
        last_frame = frame_id-1
        if use_gt_pose and self.gt_pose_provided:
            tran_in_frame = (
                np.linalg.inv(self.gt_poses[last_frame])
                @ self.gt_poses[cur_frame]
            )
        else:
            if self.config.track_on:
                tran_in_frame = (
                    np.linalg.inv(self.odom_poses[last_frame])
                    @ self.odom_poses[cur_frame]
                )
            elif self.gt_pose_provided:
                tran_in_frame = (
                    np.linalg.inv(self.gt_poses[last_frame])
                    @ self.gt_poses[cur_frame]
                )
            else:
                return None

        # tran_in_frame: T_last<-cur

        tran_in_frame = torch.tensor(tran_in_frame, device=self.device, dtype=torch.float64)

        return tran_in_frame

    def deskew_at_frame(self, tran_in_frame):

        # tran_in_frame: T_last<-cur

        self.cur_point_cloud_torch = deskewing(
            self.cur_point_cloud_torch,
            self.cur_point_ts_torch,
            tran_in_frame,
            self.config.deskew_ref_ratio,
            points_lidar_idx = self.cur_point_lidar_idx_torch,
            T_l_lm_list=self.T_l_lm_list,
            ts_diff_list=self.ts_ref_ratio_diffs 
        ) 

    def write_merged_point_cloud(self, down_vox_m=None, 
                                use_gt_pose=False, 
                                out_file_name="merged_point_cloud",
                                frame_step = 1,
                                merged_downsample = True,
                                tsdf_fusion_on=False):

        print("Begin to replay the dataset ...")

        o3d_device = o3d.core.Device("CPU:0")
        o3d_dtype = o3d.core.float32
        map_out_o3d = o3d.t.geometry.PointCloud(o3d_device)
        map_points_np = np.empty((0, 3))
        map_intensity_np = np.empty(0)
        map_color_np = np.empty((0, 3))
        
        # deprecated
        # if tsdf_fusion_on:
        #     # tsdf_fusion_voxel_size = self.config.voxel_size_m*0.6
        #     tsdf_fusion_voxel_size = self.config.tsdf_fusion_voxel_size
        #     sdf_trunc = tsdf_fusion_voxel_size * 4.0
        #     space_carving_on = self.config.tsdf_fusion_space_carving_on
        #     vdb_volume = vdbfusion.VDBVolume(tsdf_fusion_voxel_size,
        #                                     sdf_trunc,
        #                                     space_carving_on)

        for frame_id in tqdm(
            range(0, self.total_pc_count, frame_step), desc="Merge map point cloud"
        ):  # frame id as the idx of the frame in the data folder without skipping
            
            self.init_temp_data() # init cur frame temp data
            
            self.read_frame_with_loader(frame_id, False, False)

            if self.cur_point_cloud_torch is None:
                continue

            if self.config.kitti_correction_on:
                self.cur_point_cloud_torch = intrinsic_correct(
                    self.cur_point_cloud_torch, self.config.correction_deg
                )

            if self.config.deskew:
                if frame_id > 0:
                    tran_in_frame = self.get_tran_in_frame(frame_id, use_gt_pose)
                    self.deskew_at_frame(tran_in_frame)
                else: 
                    continue # FIXME, deal with first frame's deskewing

            if down_vox_m is None:
                down_vox_m = self.config.vox_down_m
            idx = voxel_down_sample_torch(self.cur_point_cloud_torch[:, :3], down_vox_m)

            frame_down_torch = self.cur_point_cloud_torch[idx]

            frame_crop_idx = crop_frame(
                frame_down_torch,
                self.config.min_z,
                self.config.max_z,
                self.config.min_range,
                self.config.max_range,
                self.config.range_filter_2d,
            )

            frame_down_torch = frame_down_torch[frame_crop_idx]

            # get pose
            if use_gt_pose and self.gt_pose_provided:
                cur_pose_torch = torch.tensor(
                    self.gt_poses[frame_id], device=self.device, dtype=torch.float64
                )
            else:
                if self.config.pgo_on:
                    cur_pose_torch = torch.tensor(
                        self.pgo_poses[frame_id],
                        device=self.device,
                        dtype=torch.float64,
                    )
                elif self.config.track_on:
                    cur_pose_torch = torch.tensor(
                        self.odom_poses[frame_id],
                        device=self.device,
                        dtype=torch.float64,
                    )
                elif self.gt_pose_provided:
                    cur_pose_torch = torch.tensor(
                        self.gt_poses[frame_id], device=self.device, dtype=torch.float64
                    )
            frame_down_torch[:, :3] = transform_torch(
                frame_down_torch[:, :3], cur_pose_torch
            )

            frame_points_np = frame_down_torch[:, :3].detach().cpu().numpy().astype(np.float64) # under map frame
            map_points_np = np.concatenate((map_points_np, frame_points_np), axis=0)
            if self.config.color_channel == 1:
                frame_intensity_np = frame_down_torch[:, 3].detach().cpu().numpy()
                map_intensity_np = np.concatenate(
                    (map_intensity_np, frame_intensity_np), axis=0
                )
            elif self.config.color_channel == 3:
                frame_color_np = frame_down_torch[:, 3:].detach().cpu().numpy()
                map_color_np = np.concatenate((map_color_np, frame_color_np), axis=0)

            cur_position_np = (cur_pose_torch.detach().cpu().numpy())[:3, 3]

            # if tsdf_fusion_on:
            #     vdb_volume.integrate(frame_points_np, cur_position_np)

        print("Replay done")

        map_out_o3d.point["positions"] = o3d.core.Tensor(
            map_points_np, o3d_dtype, o3d_device
        )
        if self.config.color_channel == 1:
            map_out_o3d.point["intensity"] = o3d.core.Tensor(
                np.expand_dims(map_intensity_np, axis=1), o3d_dtype, o3d_device
            )
        elif self.config.color_channel == 3:
            map_out_o3d.point["colors"] = o3d.core.Tensor(
                map_color_np, o3d_dtype, o3d_device
            )

        # print("Estimate normal")
        # map_out_o3d.estimate_normals(max_nn=20)
        
        # downsample again
        if merged_downsample:
            map_out_o3d = map_out_o3d.voxel_down_sample(voxel_size=down_vox_m)

        if self.run_path is not None:
            save_path = os.path.join(self.run_path, "map", out_file_name+".ply")
            o3d.t.io.write_point_cloud(save_path, map_out_o3d)
            print(f"save the merged raw point cloud map to {save_path}")

        map_out_o3d = None
        
        # if tsdf_fusion_on:

        #     # Extract triangle mesh (numpy arrays)
        #     vert, tri = vdb_volume.extract_triangle_mesh()

        #     mesh_tsdf_fusion = o3d.geometry.TriangleMesh(
        #         o3d.utility.Vector3dVector(vert),
        #         o3d.utility.Vector3iVector(tri),
        #     )

        #     mesh_tsdf_fusion.compute_vertex_normals()

        #     if self.run_path is not None:
        #         mesh_save_path = os.path.join(self.run_path, "mesh", "mesh_original_pc_tsdf_fusion_{}cm.ply".format(str(round(tsdf_fusion_voxel_size*1e2))))
        #         o3d.io.write_triangle_mesh(mesh_save_path, mesh_tsdf_fusion)
        #         print(f"save the tsdf fusion mesh from the original point cloud to {mesh_save_path}")
                
        #         vdb_grid_file = os.path.join(self.run_path, "map", "vdb_grid.npy")
        #         vdb_volume.extract_vdb_grids(vdb_grid_file)
        #         print(f"save the vdb volume to {vdb_grid_file}")

        #     vdb_volume = None

    def o3d_tsdf_fusion(self, frame_step = 1, output_path = None, vox_size = 0.02, trunc_dist = 0.06):

        scale = 1.0
        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=vox_size, # unit: m
            sdf_trunc=trunc_dist, # unit: m
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

        cam_intrinsic = self.loader.intrinsic
        T_c_l = self.loader.T_c_l   

        for frame_id in tqdm(range(0, self.total_pc_count, frame_step), desc="TSDF fusion"): 
            
            frame_id_in_folder = self.config.begin_frame + frame_id * self.config.step_frame
            frame_data = self.loader[frame_id_in_folder]

            cur_imgs = frame_data["img"]
            used_img = cur_imgs[self.loader.main_cam_name]

            rgb_image = o3d.geometry.Image(used_img[:,:,:3].astype(np.uint8))
            depth_image = o3d.geometry.Image(used_img[:,:,3].astype(np.float32))

            cur_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_image, 
                                                                        depth_image, 
                                                                        depth_scale=1.0, 
                                                                        depth_trunc=self.config.max_range, 
                                                                        convert_rgb_to_intensity=False)

            cur_T_l_w = np.linalg.inv(self.gt_poses[frame_id])
            cur_T_c_w = T_c_l @ cur_T_l_w

            volume.integrate(cur_rgbd, self.loader.intrinsic, cur_T_c_w)

        tsdf_fusion_mesh = volume.extract_triangle_mesh()

        if output_path is not None:
            o3d.io.write_triangle_mesh(str(output_path), tsdf_fusion_mesh)
            print(f"Save the mesh resulting from TSDF fusion to {output_path}")

        return tsdf_fusion_mesh


    def write_results_log(self):
        log_folder = "log"
        frame_str = str(self.processed_frame)
        
        if self.config.track_on:
            write_traj_as_o3d(
                self.odom_poses[:self.processed_frame+1],
                os.path.join(self.run_path, log_folder, frame_str + "_odom_poses.ply"),
            )
        if self.config.pgo_on:
            write_traj_as_o3d(
                self.pgo_poses[:self.processed_frame+1],
                os.path.join(self.run_path, log_folder, frame_str + "_slam_poses.ply"),
            )
        if self.gt_pose_provided:
            write_traj_as_o3d(
                self.gt_poses[:self.processed_frame+1],
                os.path.join(self.run_path, log_folder, frame_str + "_gt_poses.ply"),
            )

    def get_poses_np_for_vis(self, frame_id):
        odom_poses = None
        if self.odom_poses is not None:
            odom_poses = self.odom_poses[:frame_id+1]
        gt_poses = None
        if self.gt_poses is not None:
            gt_poses = self.gt_poses[:frame_id+1]
        
        if self.pgo_poses is not None:
            pgo_poses = self.pgo_poses[:frame_id+1]
        else:
            pgo_poses = odom_poses
        
        return odom_poses, gt_poses, pgo_poses

    def write_results(self):

        if self.config.track_on:
            odom_poses = self.odom_poses[:self.processed_frame+1]
            odom_poses_out = apply_kitti_format_calib(odom_poses, self.calib["Tr"])
            write_kitti_format_poses(os.path.join(self.run_path, "odom_poses"), odom_poses_out)
            # write_tum_format_poses(os.path.join(self.run_path, "odom_poses"), odom_poses_out, self.poses_ts, 0.1*self.config.step_frame)
            write_traj_as_o3d(odom_poses, os.path.join(self.run_path, "odom_poses.ply"))

            if self.config.pgo_on:
                pgo_poses = self.pgo_poses[:self.processed_frame+1]
                slam_poses_out = apply_kitti_format_calib(pgo_poses, self.calib["Tr"])
                write_kitti_format_poses(
                    os.path.join(self.run_path, "slam_poses"), slam_poses_out
                )
                # write_tum_format_poses(
                #     os.path.join(self.run_path, "slam_poses"), slam_poses_out, self.poses_ts, 0.1*self.config.step_frame
                # )
                write_traj_as_o3d(pgo_poses, os.path.join(self.run_path, "slam_poses.ply"))
        
        # timing report
        time_table = np.array(self.time_table)
        mean_time_s = np.sum(time_table) / self.processed_frame * 1.0
        mean_time_without_init_s = np.sum(time_table[1:]) / (self.processed_frame-1) * 1.0
        if not self.silence:
            print("Consuming time per frame        (s):", f"{mean_time_without_init_s:.3f}")
            print("Calculated over %d frames" % self.processed_frame)
        np.save(
            os.path.join(self.run_path, "time_table.npy"), time_table
        )  # save detailed time table

        plot_timing_detail(
                time_table,
                os.path.join(self.run_path, "time_details.png"),
                self.config.pgo_on,
            )

        pose_eval = None

        # pose estimation evaluation report
        if self.gt_pose_provided:
            gt_poses = self.gt_poses[:self.processed_frame+1]
            gt_poses_out = apply_kitti_format_calib(gt_poses, self.calib["Tr"])
            write_kitti_format_poses(os.path.join(self.run_path, "gt_poses"), gt_poses_out)
            # write_tum_format_poses(os.path.join(self.run_path, "gt_poses"), gt_poses_out, self.poses_ts, 0.1*self.config.step_frame)
            write_traj_as_o3d(gt_poses, os.path.join(self.run_path, "gt_poses.ply"))

            if self.config.track_on:
                
                print("Odometry evaluation:")
                avg_tra, avg_rot = relative_error(gt_poses, odom_poses)
                ate_rot, ate_trans, align_mat = absolute_error(
                    gt_poses, odom_poses, self.config.eval_traj_align
                )
                if avg_tra == 0:  # for rgbd dataset (shorter sequence)
                    print("Absoulte Trajectory Error      (cm):", f"{ate_trans*100.0:.3f}")
                else:
                    print("Average Translation Error       (%):", f"{avg_tra:.3f}")
                    print("Average Rotational Error (deg/100m):", f"{avg_rot*100.0:.3f}")
                    print("Absoulte Trajectory Error       (m):", f"{ate_trans:.3f}")

                if self.config.wandb_vis_on:
                    wandb_log_content = {
                        "Average Translation Error [%]": avg_tra,
                        "Average Rotational Error [deg/m]": avg_rot,
                        "Absoulte Trajectory Error [m]": ate_trans,
                        "Absoulte Rotational Error [deg]": ate_rot,
                        "Consuming time per frame [s]": mean_time_without_init_s,
                    }
                    wandb.log(wandb_log_content)

                if self.config.pgo_on:
                    print("SLAM evaluation:")
                    avg_tra_slam, avg_rot_slam = relative_error(
                        gt_poses, pgo_poses
                    )
                    ate_rot_slam, ate_trans_slam, align_mat_slam = absolute_error(
                        gt_poses, pgo_poses, self.config.eval_traj_align
                    )
                    if avg_tra_slam == 0:  # for rgbd dataset (shorter sequence)
                        print(
                            "Absoulte Trajectory Error      (cm):",
                            f"{ate_trans_slam*100.0:.3f}",
                        )
                    else:
                        print("Average Translation Error       (%):", f"{avg_tra_slam:.3f}")
                        print(
                            "Average Rotational Error (deg/100m):",
                            f"{avg_rot_slam*100.0:.3f}",
                        )
                        print(
                            "Absoulte Trajectory Error       (m):", f"{ate_trans_slam:.3f}"
                        )

                    if self.config.wandb_vis_on:
                        wandb_log_content = {
                            "SLAM Average Translation Error [%]": avg_tra_slam,
                            "SLAM Average Rotational Error [deg/m]": avg_rot_slam,
                            "SLAM Absoulte Trajectory Error [m]": ate_trans_slam,
                            "SLAM Absoulte Rotational Error [deg]": ate_rot_slam,
                        }
                        wandb.log(wandb_log_content)

                csv_columns = [
                    "Average Translation Error [%]",
                    "Average Rotational Error [deg/m]",
                    "Absoulte Trajectory Error [m]",
                    "Absoulte Rotational Error [deg]",
                    "Consuming time per frame [s]",
                    "Frame count",
                ]
                pose_eval = [
                    {
                        csv_columns[0]: avg_tra,
                        csv_columns[1]: avg_rot,
                        csv_columns[2]: ate_trans,
                        csv_columns[3]: ate_rot,
                        csv_columns[4]: mean_time_without_init_s,
                        csv_columns[5]: int(self.processed_frame),
                    }
                ]
                if self.config.pgo_on:
                    slam_eval_dict = {
                        csv_columns[0]: avg_tra_slam,
                        csv_columns[1]: avg_rot_slam,
                        csv_columns[2]: ate_trans_slam,
                        csv_columns[3]: ate_rot_slam,
                        csv_columns[4]: mean_time_without_init_s,
                        csv_columns[5]: int(self.processed_frame),
                    }
                    pose_eval.append(slam_eval_dict)
                output_csv_path = os.path.join(self.run_path, "pose_eval.csv")
                try:
                    with open(output_csv_path, "w") as csvfile:
                        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
                        writer.writeheader()
                        for data in pose_eval:
                            writer.writerow(data)
                except IOError:
                    print("I/O error")

                # if self.config.o3d_vis_on:  # x service issue for remote server
                output_traj_plot_path_2d = os.path.join(self.run_path, "traj_plot_2d.png")
                output_traj_plot_path_3d = os.path.join(self.run_path, "traj_plot_3d.png")
                # trajectory not aligned yet in the plot
                # require list of numpy arraies as the input

                gt_position_list = [self.gt_poses[i] for i in range(self.processed_frame)]
                odom_position_list = [self.odom_poses[i] for i in range(self.processed_frame)]

                if self.config.pgo_on:
                    pgo_position_list = [self.pgo_poses[i] for i in range(self.processed_frame)]
                    plot_trajectories(
                        output_traj_plot_path_2d,
                        pgo_position_list,
                        gt_position_list,
                        odom_position_list,
                        plot_3d=False,
                    )
                    plot_trajectories(
                        output_traj_plot_path_3d,
                        pgo_position_list,
                        gt_position_list,
                        odom_position_list,
                        plot_3d=True,
                    )
                else:
                    plot_trajectories(
                        output_traj_plot_path_2d,
                        odom_position_list,
                        gt_position_list,
                        plot_3d=False,
                    )
                    plot_trajectories(
                        output_traj_plot_path_3d,
                        odom_position_list,
                        gt_position_list,
                        plot_3d=True,
                    )

        return pose_eval

    # point-wise timestamp is now only used for motion undistortion (deskewing)
    def get_point_ts(self, point_ts=None): 
        # point_ts is already the normalized timestamp in a scan frame # [0,1]
        
        if self.cur_point_cloud_torch is None:
            return

        if self.config.deskew:
            if point_ts is not None and min(point_ts) < 1.0: # not all 1
                # if not self.silence:
                #     print("Pointwise timestamp available")
                self.cur_point_ts_torch = torch.tensor(
                    point_ts, device=self.device, dtype=self.dtype
                )
            else: # point_ts not available, guess the ts
                # FIXME: some isse here, better to directly read the ts instead of guessing from point index
                point_count = self.cur_point_cloud_torch.shape[0]
                if point_count == 64 * 1024:
                     # for Ouster 64-beam LiDAR
                    if not self.silence:
                        print("Ouster-64 point cloud deskewed")
                    self.cur_point_ts_torch = (
                        (torch.floor(torch.arange(point_count) / 64) / 1024)
                        .reshape(-1, 1)
                        .to(self.cur_point_cloud_torch)
                    )
                elif (
                    point_count == 128 * 1024 or point_count == 128 * 2048
                ):  # for Ouster 128-beam LiDAR
                    if not self.silence:
                        print("Ouster-128 point cloud deskewed")
                    hres = point_count / 128
                    self.cur_point_ts_torch = (
                        (torch.floor(torch.arange(point_count) / 128) / hres)
                        .reshape(-1, 1)
                        .to(self.cur_point_cloud_torch)
                    )
                else:
                    yaw = -torch.atan2(
                        self.cur_point_cloud_torch[:, 1],
                        self.cur_point_cloud_torch[:, 0],
                    )  # y, x -> rad (clockwise)
                    if self.config.lidar_type_guess == "velodyne":
                        # for velodyne LiDAR (from -x axis, clockwise)
                        self.cur_point_ts_torch = 0.5 * (yaw / math.pi + 1.0)  # [0,1]
                        if not self.silence:
                            print("Velodyne point cloud deskewed")
                    else:
                        # for Hesai LiDAR (from +y axis, clockwise)
                        self.cur_point_ts_torch = 0.5 * (
                            yaw / math.pi + 0.5
                        )  # [-0.25,0.75]
                        self.cur_point_ts_torch[
                            self.cur_point_ts_torch < 0
                        ] += 1.0  # [0,1]
                        if not self.silence:
                            print("HESAI point cloud deskewed")


def set_dataset_path(config: Config, dataset_name: str = '', seq: str = ''):
    
    if seq is None:
        seq = ''
    
    config.name = config.name + '_' + dataset_name + '_' + seq.replace("/", "")
    
    # now use_dataloader is by default
    assert config.use_dataloader, "You have to use a specific data loader for your input data for PINGS"
    config.data_loader_name = dataset_name
    config.data_loader_seq = seq


def read_kitti_format_calib(filename: str):
    """
    read calibration file (with the kitti format)
    returns -> dict calibration matrices as 4*4 numpy arrays
    """
    calib = {}
    calib_file = open(filename)

    for line in calib_file:
        key, content = line.strip().split(":")
        values = [float(v) for v in content.strip().split()]
        pose = np.zeros((4, 4))

        pose[0, 0:4] = values[0:4]
        pose[1, 0:4] = values[4:8]
        pose[2, 0:4] = values[8:12]
        pose[3, 3] = 1.0

        calib[key] = pose

    calib_file.close()
    return calib


def read_kitti_format_poses(filename: str) -> List[np.ndarray]:
    """
    read pose file (with the kitti format)
    returns -> list, transformation before calibration transformation
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
    
    return poses

def read_tum_format_poses(filename: str):
    """
    read pose file (with the tum format), support txt file
    # timestamp tx ty tz qx qy qz qw
    returns -> list, transformation before calibration transformation
    """
    from pyquaternion import Quaternion

    poses = []
    timestamps = []
    with open(filename, 'r') as file:
        first_line = file.readline().strip()
        
        # check if the first line contains any numeric characters
        # if contain, then skip the first line # timestamp tx ty tz qx qy qz qw
        if any(char.isdigit() for char in first_line):
            file.seek(0)
        
        for line in file: # read each line in the file 
            values = line.strip().split()
            if len(values) != 8 and len(values) != 9: 
                print('Not a tum format pose file')
                return None, None
            # some tum format pose file also contain the idx before timestamp
            idx_col =  len(values) - 8 # 0 or 1
            values = [float(value) for value in values]
            timestamps.append(values[idx_col])
            trans = np.array(values[1+idx_col:4+idx_col])
            quat = Quaternion(np.array([values[7+idx_col], values[4+idx_col], values[5+idx_col], values[6+idx_col]])) # w, i, j, k
            rot = quat.rotation_matrix
            # Build numpy transform matrix
            odom_tf = np.eye(4)
            odom_tf[0:3, 3] = trans
            odom_tf[0:3, 0:3] = rot
            poses.append(odom_tf)
    
    return poses, timestamps


def write_kitti_format_poses(filename: str, poses_np: np.ndarray, direct_use_filename = False):
    poses_out = poses_np[:, :3, :]
    poses_out_kitti = poses_out.reshape(poses_out.shape[0], -1)

    if direct_use_filename:
        fname = filename
    else:
        fname = f"{filename}_kitti.txt"
    
    np.savetxt(fname=fname, X=poses_out_kitti)

def write_tum_format_poses(filename: str, poses_np: np.ndarray,
                           timestamps=None, frame_s = 0.1, 
                           with_header = False, direct_use_filename = False):
    from pyquaternion import Quaternion

    frame_count = poses_np.shape[0]
    tum_out = np.empty((frame_count,8))
    for i in range(frame_count):
        tx, ty, tz = poses_np[i, :3, -1].flatten()

        determinant_check = np.linalg.det(poses_np[i, :3, :3])
        if determinant_check < 0.99:
            print("rotation matrix not valid, skip the pose output")
            return

        qw, qx, qy, qz = Quaternion(matrix=poses_np[i], atol=1e-3).elements
        if timestamps is None:
            ts = i * frame_s
        else:
            ts = float(timestamps[i])
        tum_out[i] = np.array([ts, tx, ty, tz, qx, qy, qz, qw])

    if with_header:
        header = "timestamp tx ty tz qx qy qz qw"
    else:
        header = ''
        
    if direct_use_filename:
        fname = filename
    else:
        fname = f"{filename}_tum.txt"

    np.savetxt(fname=fname, X=tum_out, fmt="%.4f", header=header)

def apply_kitti_format_calib(poses_np: np.ndarray, calib_T_cl: np.ndarray):
    """Converts from Velodyne to Camera Frame (# T_camera<-lidar)"""
    poses_calib_np = poses_np.copy()
    for i in range(poses_np.shape[0]):
        poses_calib_np[i, :, :] = calib_T_cl @ poses_np[i, :, :] @ inv(calib_T_cl)

    return poses_calib_np

# torch version
def crop_frame(
    points: torch.tensor,
    min_z_th=-3.0,
    max_z_th=100.0,
    min_range=2.75,
    max_range=100.0,
    on_xy_plane: bool = True
):
    if on_xy_plane:
        dist = torch.norm(points[:, :2], dim=1) # 2d distance
    else:
        dist = torch.norm(points[:, :3], dim=1) # 3d distance

    filtered_idx = (
        (dist > min_range)
        & (dist < max_range)
        & (points[:, 2] > min_z_th)
        & (points[:, 2] < max_z_th)
    )
    return filtered_idx

# torch version
def intrinsic_correct(points: torch.tensor, correct_deg=0.0):

    # # This function only applies for the KITTI dataset, and should NOT be used by any other dataset,
    # # the original idea and part of the implementation is taking from CT-ICP(Although IMLS-SLAM
    # # Originally introduced the calibration factor)
    # We set the correct_deg = 0.195 deg for KITTI odom dataset, inline with MULLS #issue 11
    if correct_deg == 0.0:
        return points

    dist = torch.norm(points[:, :3], dim=1)
    kitti_var_vertical_ang = correct_deg / 180.0 * math.pi
    v_ang = torch.asin(points[:, 2] / dist)
    v_ang_c = v_ang + kitti_var_vertical_ang
    hor_scale = torch.cos(v_ang_c) / torch.cos(v_ang)
    points[:, 0] *= hor_scale
    points[:, 1] *= hor_scale
    points[:, 2] = dist * torch.sin(v_ang_c)

    return points


# now only work for semantic kitti format dataset # torch version
def filter_sem_kitti(
    points: torch.tensor,
    sem_labels: torch.tensor,
    filter_outlier=True,
    filter_moving=False,
):

    # sem_labels_reduced is the reduced labels for mapping (20 classes for semantic kitti)
    # sem_labels is the original semantic label (0-255 for semantic kitti)

    if filter_outlier:  # filter the outliers according to semantic labels
        inlier_mask = sem_labels > 1  # not outlier
    else:
        inlier_mask = sem_labels >= 0  # all

    if filter_moving:
        static_mask = sem_labels < 100  # only for semantic KITTI dataset
        inlier_mask = inlier_mask & static_mask

    return inlier_mask


def write_traj_as_o3d(poses_np, path):

    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(poses_np[:, :3, 3])

    ts_np = np.linspace(0, 1, poses_np.shape[0])
    color_map = cm.get_cmap("jet")
    ts_color = color_map(ts_np)[:, :3].astype(np.float64)
    o3d_pcd.colors = o3d.utility.Vector3dVector(ts_color)

    if path is not None:
        o3d.io.write_point_cloud(path, o3d_pcd)

    return o3d_pcd