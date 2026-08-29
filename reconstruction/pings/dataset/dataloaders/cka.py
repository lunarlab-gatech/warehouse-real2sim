# MIT License
#
# Copyright (c) 2024 Yue Pan
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
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
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import importlib
import glob
import os
import json
from pathlib import Path

import numpy as np
import open3d as o3d

# CKA RGBD dataset in the greenhouse
# or general RGBD data collected by realsense
# For example, the BUP20 dataset and the fruit shape completion dataset (https://www.ipb.uni-bonn.de/data/shape_completion/index.html)

class CKADataset:
    def __init__(self, data_dir: Path, sequence: str, *_, **__):

        self.is_rgbd: bool = True
        self.contains_image: bool = True

        self.rgb_dir = os.path.join(data_dir, "color/")
        self.depth_dir = os.path.join(data_dir, "depth/")

        self.rgb_frames = sorted(glob.glob(self.rgb_dir + '*.png'))
        self.depth_frames = sorted(glob.glob(self.depth_dir + '*.npy'))

        assert len(self.rgb_frames) == len(self.depth_frames), "RGB frame and depth frame count are not identical"

        # for the shape completion challenge dataset

        pose_file = os.path.join(data_dir, "poses_metashape.npz") # poses_metashape.npz
        if os.path.exists(pose_file):
            self.gt_poses=np.load(pose_file, allow_pickle=True)['arr_0']
        else:
            pose_dir = os.path.join(data_dir, "poses/")
            assert os.path.exists(pose_dir)
            self.pose_frames = sorted(glob.glob(pose_dir + '*.txt'))

            # load gt poses
            gt_poses_list = []
            assert len(self.pose_frames) == len(self.depth_frames), "frame poses and depth frame count are not identical"
            for pose_frame in self.pose_frames:
                cur_pose = np.loadtxt(pose_frame)
                gt_poses_list.append(cur_pose)
            self.gt_poses = np.array(gt_poses_list) # N,4,4

        self.intrinsic_file = os.path.join(data_dir, "intrinsic.json")

        with open(self.intrinsic_file, 'r') as infile: # load intrinsic json file
            intrinsic_data = json.load(infile)
            intrinsic_mat = intrinsic_data["intrinsic_matrix"]
            width = intrinsic_data["width"]
            height = intrinsic_data["height"]
            if "depth_scale" in list(intrinsic_data.keys()):
                self.depth_scale = intrinsic_data["depth_scale"]
            else:
                self.depth_scale = 1.0

            self.fx = intrinsic_mat[0]
            self.fy = intrinsic_mat[4]
            self.cx = intrinsic_mat[6]
            self.cy = intrinsic_mat[7]

            self.K_mat = np.eye(3)
            self.K_mat[0,0]=self.fx
            self.K_mat[1,1]=self.fy
            self.K_mat[0,2]=self.cx
            self.K_mat[1,2]=self.cy

            self.main_cam_name = "cam_mid"

            self.K_mats = {self.main_cam_name: self.K_mat}

            self.T_l_c = np.eye(4)
            self.T_c_l = np.linalg.inv(self.T_l_c)

            self.T_c_l_mats = {self.main_cam_name: self.T_c_l}

            self.cam_heights = {self.main_cam_name: height}
            self.cam_widths = {self.main_cam_name: width}
        
            self.intrinsic = o3d.camera.PinholeCameraIntrinsic()
            self.intrinsic.set_intrinsics(height=height,
                                          width=width,
                                          fx=self.fx,
                                          fy=self.fy,
                                          cx=self.cx,
                                          cy=self.cy)
            self.extrinsic = self.T_c_l 
        
        self.max_depth_m = 2.0
        self.down_sample_on = False
        self.rand_down_rate = 0.1

        self.filter_depth = False

    def __len__(self):
        return len(self.depth_frames)

    def __getitem__(self, idx):
        rgb_image = o3d.io.read_image(self.rgb_frames[idx])

        depth = np.load(self.depth_frames[idx])

        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_image, 
                                                                            o3d.geometry.Image(depth),
                                                                            depth_scale=self.depth_scale, 
                                                                            depth_trunc=self.max_depth_m, 
                                                                            convert_rgb_to_intensity=False)

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image, self.intrinsic, self.extrinsic)
        if self.down_sample_on:
            pcd = pcd.random_down_sample(sampling_ratio=self.rand_down_rate)
        
        points_xyz = np.array(pcd.points, dtype=np.float64)
        points_rgb = np.array(pcd.colors, dtype=np.float64)
        points_xyzrgb = np.hstack((points_xyz, points_rgb))

        depth = 1.0 * depth / self.depth_scale
        depth_image = np.expand_dims(depth, axis=-1)
        rgb_image = np.array(rgb_image)
        # rgbd_image = np.concatenate((rgb_image, depth_image), axis=-1) # 4 channels

        image_dict = {self.main_cam_name: rgb_image}

        depth_img_dict = {self.main_cam_name: depth_image}

        frame_data = {"points": points_xyzrgb, "img": image_dict, "depth": depth_img_dict}

        return frame_data 