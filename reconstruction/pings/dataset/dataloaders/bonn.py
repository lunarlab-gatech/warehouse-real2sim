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
import os
from pathlib import Path

import numpy as np
import open3d as o3d

# Bonn RGB-D Dynamic Dataset: https://www.ipb.uni-bonn.de/data/rgbd-dynamic-dataset/index.html

class BonnRGBDDataset:
    def __init__(self, data_dir: Path, sequence: str, *_, **__):
 
        sequence_dir = os.path.join(data_dir, sequence)

        self.contains_image: bool = True

        self.is_rgbd: bool = True

        self.rgb_frames, self.depth_frames, self.gt_poses = self.loadtum(sequence_dir)

        init_pose = self.gt_poses[0]

        # apply the transform to the TLS's reference frame
        T_ros = np.array([[-1, 0, 0, 0],
                        [0, 0, 1, 0],
                        [0, 1, 0, 0],
                        [0, 0, 0, 1]])

        T_m = np.array([[1.0157, 0.1828, -0.2389, 0.0113],
                        [0.0009, -0.8431, -0.6413, -0.0098],
                        [-0.3009, 0.6147, -0.8085, 0.0111],
                        [0, 0, 0, 1]])

        self.gt_poses = T_ros @ self.gt_poses @ T_ros @ T_m
        
        # self.gt_poses = T_g @ self.gt_poses

        self.intrinsic = o3d.camera.PinholeCameraIntrinsic()
        
        self.H, self.W = 480, 640
        self.fx, self.fy, self.cx, self.cy = 542.822841, 542.576870, 315.593520, 237.756098

        # distortion coefficients (not applied yet)
        d0 = 0.039903
        d1 = -0.099343
        d2 = -0.000730
        d3 = -0.000144
        d4 = 0.000000
                
        self.depth_scale = 5000.0 # 5000.0

        self.intrinsic.set_intrinsics(height=self.H,
                                    width=self.W,
                                    fx=self.fx,
                                    fy=self.fy,
                                    cx=self.cx,
                                    cy=self.cy)
        
        self.K_mat = np.eye(3)
        self.K_mat[0,0]=self.fx
        self.K_mat[1,1]=self.fy
        self.K_mat[0,2]=self.cx
        self.K_mat[1,2]=self.cy

        self.main_cam_name = "cam"

        self.K_mats = {self.main_cam_name: self.K_mat}

        self.T_l_c = np.eye(4)
        self.T_c_l = np.linalg.inv(self.T_l_c)

        self.extrinsic = self.T_c_l

        self.T_c_l_mats = {self.main_cam_name: self.T_c_l}

        self.cam_heights = {self.main_cam_name: self.H}
        self.cam_widths = {self.main_cam_name: self.W}
        
        self.down_sample_on = False
        self.rand_down_rate = 0.1

    def __len__(self):
        return len(self.depth_frames)

    # Bonn RGBD shares the same data format and sturcture as TUM RGBD
    def loadtum(self, datapath, frame_rate=-1):
        """ read video data in tum-rgbd format """
        if os.path.isfile(os.path.join(datapath, 'groundtruth.txt')):
            pose_list = os.path.join(datapath, 'groundtruth.txt')

        image_list = os.path.join(datapath, 'rgb.txt')
        depth_list = os.path.join(datapath, 'depth.txt')

        def parse_list(filepath, skiprows=0):
            """ read list data """
            data = np.loadtxt(filepath, delimiter=' ',
                                dtype=np.unicode_, skiprows=skiprows)
            return data

        def associate_frames(tstamp_image, tstamp_depth, tstamp_pose, max_dt=0.08):
            """ pair images, depths, and poses """
            associations = []
            for i, t in enumerate(tstamp_image):
                if tstamp_pose is None:
                    j = np.argmin(np.abs(tstamp_depth - t))
                    if (np.abs(tstamp_depth[j] - t) < max_dt):
                        associations.append((i, j))
                else:
                    j = np.argmin(np.abs(tstamp_depth - t))
                    k = np.argmin(np.abs(tstamp_pose - t))

                    if (np.abs(tstamp_depth[j] - t) < max_dt) and \
                            (np.abs(tstamp_pose[k] - t) < max_dt):
                        associations.append((i, j, k))
            return associations
        
        def pose_matrix_from_quaternion(pvec):
            """ convert 4x4 pose matrix to (t, q) """
            from scipy.spatial.transform import Rotation
            pose = np.eye(4)
            pose[:3, :3] = Rotation.from_quat(pvec[3:]).as_matrix() # rotation
            pose[:3, 3] = pvec[:3] # translation
            return pose

        image_data = parse_list(image_list)
        depth_data = parse_list(depth_list)
        pose_data = parse_list(pose_list, skiprows=1)
        pose_vecs = pose_data[:, 1:].astype(np.float64)

        tstamp_image = image_data[:, 0].astype(np.float64)
        tstamp_depth = depth_data[:, 0].astype(np.float64)
        tstamp_pose = pose_data[:, 0].astype(np.float64)
        associations = associate_frames(
            tstamp_image, tstamp_depth, tstamp_pose)

        indicies = [0]
        for i in range(1, len(associations)):
            t0 = tstamp_image[associations[indicies[-1]][0]]
            t1 = tstamp_image[associations[i][0]]
            if t1 - t0 > 1.0 / frame_rate:
                indicies += [i]

        images, poses, depths = [], [], []
        for ix in indicies:
            (i, j, k) = associations[ix]
            images += [os.path.join(datapath, image_data[i, 1])]
            depths += [os.path.join(datapath, depth_data[j, 1])]
            c2w = pose_matrix_from_quaternion(pose_vecs[k])
            poses += [c2w]

        poses = np.array(poses)

        return images, depths, poses

    def __getitem__(self, idx): 
        
        im_color = o3d.io.read_image(self.rgb_frames[idx])
        # print(im_color)

        im_depth = o3d.io.read_image(self.depth_frames[idx]) 
        rgbd_image = o3d.geometry.RGBDImage.create_from_tum_format(im_color,
            im_depth, convert_rgb_to_intensity=False)

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image,
            self.intrinsic,
            self.extrinsic)

        if self.down_sample_on:
            pcd = pcd.random_down_sample(sampling_ratio=self.rand_down_rate)

        points_xyz = np.array(pcd.points, dtype=np.float64)
        points_rgb = np.array(pcd.colors, dtype=np.float64)
        points_xyzrgb = np.hstack((points_xyz, points_rgb))

        rgb_image = np.array(im_color)

        depth_image = np.expand_dims(np.array(im_depth)/self.depth_scale, axis=-1)

        image_dict = {self.main_cam_name: rgb_image}
        depth_img_dict = {self.main_cam_name: depth_image}

        frame_data = {"points": points_xyzrgb, "img": image_dict, "depth": depth_img_dict}

        return frame_data 