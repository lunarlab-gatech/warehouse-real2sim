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
import os
from pathlib import Path
from natsort import natsorted
import numpy as np
import open3d as o3d


class NeuralRGBDDataset:
    def __init__(self, data_dir: Path, sequence: str, *_, **__):

        self.contains_image: bool = True

        self.is_rgbd: bool = True
        
        sequence_dir = os.path.join(data_dir, sequence)
        rgb_folder_path = os.path.join(sequence_dir, 'images')
        depth_folder_path = os.path.join(sequence_dir, 'depth')

        rgb_frames_name = natsorted(os.listdir(rgb_folder_path))    
        depth_frames_name = natsorted(os.listdir(depth_folder_path)) 

        self.rgb_frames = [os.path.join(rgb_folder_path, f) for f in rgb_frames_name if f.endswith('png')]
        self.depth_frames = [os.path.join(depth_folder_path, f) for f in depth_frames_name if f.endswith('png')]

        self.pose_path = os.path.join(sequence_dir, 'poses.txt')
        self.gt_poses, _ = self.load_poses(self.pose_path)

        H, W = 480, 640
        focal_file_path =  os.path.join(sequence_dir, "focal.txt")
        focal_file = open(focal_file_path, "r")
        focal_length = float(focal_file.readline())

        self.fx = focal_length
        self.fy = focal_length
        self.cx = (W-1)/2.0
        self.cy = (H-1)/2.0

        self.K_mat = np.eye(3)
        self.K_mat[0,0]=self.fx
        self.K_mat[1,1]=self.fy
        self.K_mat[0,2]=self.cx
        self.K_mat[1,2]=self.cy

        self.main_cam_name = "cam"
        self.K_mats = {self.main_cam_name: self.K_mat}

        self.intrinsic = o3d.camera.PinholeCameraIntrinsic()
        self.intrinsic.set_intrinsics(height=H,
                                      width=W,
                                      fx=self.fx,
                                      fy=self.fy,
                                      cx=self.cx,
                                      cy=self.cy)
        
        self.extrinsic = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

        self.T_l_c = self.extrinsic.astype(np.float64)
        self.T_c_l = np.linalg.inv(self.T_l_c)

        self.T_c_l_mats = {self.main_cam_name: self.T_c_l}
    
        self.cam_heights = {self.main_cam_name: H}
        self.cam_widths = {self.main_cam_name: W}

        self.calibration = {"Tr": np.ravel(self.T_l_c)[:12]} # FIXME

        self.depth_scale = 1000.0
        self.max_depth_m = 10.0
        self.down_sample_on = False
        self.rand_down_rate = 0.1

    def __len__(self):
        return len(self.depth_frames)

    def load_poses(self, path):
        file = open(path, "r")
        lines = file.readlines()
        file.close()
        poses = []
        valid = []
        lines_per_matrix = 4
        for i in range(0, len(lines), lines_per_matrix):
            if 'nan' in lines[i]:
                valid.append(False)
                poses.append(np.eye(4, 4, dtype=np.float32).tolist())
            else:
                valid.append(True)
                pose_floats = np.array([[float(x) for x in line.split()] for line in lines[i:i+lines_per_matrix]])
                poses.append(pose_floats)   
        poses = np.array(poses)
        valid = np.array(valid)
        return poses, valid

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
        depth_image = np.array(depth_image) / self.depth_scale

        rgbd_image = np.concatenate((rgb_image, np.expand_dims(depth_image, axis=-1)), axis=-1) # 4 channels

        rgbd_image_dict = {self.main_cam_name: rgbd_image}

        frame_data = {"points": points_xyzrgb, "img": rgbd_image_dict}

        return frame_data 