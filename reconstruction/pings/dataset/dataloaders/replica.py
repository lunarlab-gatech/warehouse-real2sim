# MIT License
#
# Copyright (c) 2022 Ignacio Vizzo, Tiziano Guadagnino, Benedikt Mersch, Cyrill
# Stachniss.
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
from pathlib import Path

import numpy as np
import open3d as o3d

# The Replica dataset is a commonly used synthetic RGB-D dataset
# It can be downloaded from: https://cvg-data.inf.ethz.ch/nice-slam/data/Replica.zip
# Or use the script: sh scripts/download_replica.sh

class ReplicaDataset:
    def __init__(self, data_dir: Path, sequence: str, *_, **__):

        sequence_dir = os.path.join(data_dir, sequence)

        self.contains_image: bool = True
        self.is_rgbd: bool = True

        self.img_dir = os.path.join(sequence_dir, "results/")
        self.rgb_frames = sorted(glob.glob(self.img_dir + '*.jpg'))
        self.depth_frames = sorted(glob.glob(self.img_dir + '*.png'))
        
        self.poses_fn = os.path.join(sequence_dir, "traj.txt")
        self.gt_poses = self.load_poses(self.poses_fn)

        self.intrinsic = o3d.camera.PinholeCameraIntrinsic()
        # From cam_params.json, shared by all sequences

        H, W = 680, 1200
        
        self.fx = 600.0
        self.fy = 600.0
        self.cx = 599.5
        self.cy = 339.5

        self.K_mat = np.eye(3)
        self.K_mat[0,0]=self.fx
        self.K_mat[1,1]=self.fy
        self.K_mat[0,2]=self.cx
        self.K_mat[1,2]=self.cy

        self.main_cam_name = "cam"
        self.K_mats = {self.main_cam_name: self.K_mat}

        self.T_l_c = np.eye(4)
        self.T_c_l = np.linalg.inv(self.T_l_c)

        self.T_c_l_mats = {self.main_cam_name: self.T_c_l}
        self.cam_heights = {self.main_cam_name: H}
        self.cam_widths = {self.main_cam_name: W}

        self.intrinsic.set_intrinsics(height=H,
                                      width=W,
                                      fx=self.fx,
                                      fy=self.fy,
                                      cx=self.cx,
                                      cy=self.cy)
        
        self.extrinsic = self.T_c_l

        self.depth_scale = 6553.5
        self.max_depth_m = 15.0
        self.down_sample_on = False
        self.rand_down_rate = 0.1

    def __len__(self):
        return len(self.depth_frames)

    def load_poses(self, poses_file: str) -> np.ndarray:
        poses = np.loadtxt(poses_file, delimiter=" ")
        n = poses.shape[0]
        return poses.reshape((n, 4, 4)) 

    def __getitem__(self, idx):
        rgb_image = o3d.io.read_image(self.rgb_frames[idx])
        depth_image = o3d.io.read_image(self.depth_frames[idx])

        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_image, 
                                                                        depth_image, 
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

        rgb_image = np.array(rgb_image)

        depth_image = np.expand_dims(np.array(depth_image)/self.depth_scale, axis=-1) # H, W, 1

        image_dict = {self.main_cam_name: rgb_image}
        depth_img_dict = {self.main_cam_name: depth_image}

        frame_data = {"points": points_xyzrgb, "img": image_dict, "depth": depth_img_dict}

        return frame_data 