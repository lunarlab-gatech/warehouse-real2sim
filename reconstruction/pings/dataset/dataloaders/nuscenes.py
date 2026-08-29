# MIT License
#
# Copyright (c) 2022 Ignacio Vizzo, Tiziano Guadagnino, Benedikt Mersch, Cyrill
# Stachniss.
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
import sys
from pathlib import Path
from typing import List

import cv2
import numpy as np

# Reference:
# https://github.com/fudan-zvg/S-NeRF/blob/main/scripts/nuscenes_preprocess.py
# search and check nuscenes img loaders
# what is the v1.01-mini split

class NuScenesDataset:
    def __init__(self, data_dir: Path, sequence: str, *_, **__):
        try:
            importlib.import_module("nuscenes")
        except ModuleNotFoundError:
            print("nuscenes-devkit is not installed on your system")
            print('run "pip install nuscenes-devkit"')
            sys.exit(1)
        
        self.contains_image: bool = True

        # NOTE: If someone needs more splits from nuScenes expose this 2 parameters
        #  nusc_version: str = "v1.0-trainval"
        #  split: str = "train"
        nusc_version: str = "v1.0-mini"
        split: str = "mini_train"
        # self.lidar_name: str = "LIDAR_TOP" # this is unique
        # self.cam_name: str = "CAM_FRONT" # we actually need all these images

        self.used_part: str = "sample_data"
        self.keyframe_only: bool = True

        self.use_only_colorized_points: bool = True

        # self.used_part: str = "sample"

        # Lazy loading
        from nuscenes.nuscenes import NuScenes
        from nuscenes.utils.splits import create_splits_logs

        self.sequence_id = str(sequence).zfill(4)

        self.nusc = NuScenes(dataroot=str(data_dir), version=nusc_version)
        self.scene_name = f"scene-{self.sequence_id}"
        if self.scene_name not in [s["name"] for s in self.nusc.scene]:
            print(f'[ERROR] Sequence "{self.sequence_id}" not available scenes')
            print("\nAvailable scenes:")
            self.nusc.list_scenes()
            sys.exit(1)

        # Load nuScenes read from file inside dataloader module
        self.load_point_cloud = importlib.import_module(
            "nuscenes.utils.data_classes"
        ).LidarPointCloud.from_file

        # Get assignment of scenes to splits.
        split_logs = create_splits_logs(split, self.nusc)

        # TODO: synchronization issues

        # print(split_logs)

        # print(len(self.nusc.sample)) # 404 ?
        # print(self.nusc.sample)

        # you should use sweep instead of samples
        # sample data are keyframes sampled per 0.5s (2Hz)
        # my_sample = self.nusc.sample[0] # this is list the frame

        # print(my_sample)
        # self.nusc.list_sample(my_sample['token'])
        # self.nusc.render_pointcloud_in_image(my_sample['token'], dot_size = 8, pointsensor_channel='LIDAR_TOP', camera_channel='CAM_BACK')

        # Use only the samples from the current split.
        scene_token = self._get_scene_token(split_logs) # get all the samples in a scene, each sample contains different sensor data 
        self.lidar_tokens = self._get_sensor_tokens(scene_token, "LIDAR_TOP", self.keyframe_only) # 382
        self.cam_front_tokens = self._get_sensor_tokens(scene_token, "CAM_FRONT", self.keyframe_only) # 224 # why so few
        self.cam_front_left_tokens = self._get_sensor_tokens(scene_token, "CAM_FRONT_LEFT", self.keyframe_only) 
        self.cam_front_right_tokens = self._get_sensor_tokens(scene_token, "CAM_FRONT_RIGHT", self.keyframe_only) 
        self.cam_back_tokens = self._get_sensor_tokens(scene_token, "CAM_BACK", self.keyframe_only) 
        self.cam_back_left_tokens = self._get_sensor_tokens(scene_token, "CAM_BACK_LEFT", self.keyframe_only) 
        self.cam_back_right_tokens = self._get_sensor_tokens(scene_token, "CAM_BACK_RIGHT", self.keyframe_only) 

        # support multiple cameras (dictionary of camera intrinsics, transformation to lidar frame)
        self.K_mats = {}  
        self.T_c_l_mats = {} 

        self._load_calib() # load calibs

        self.mono_depth_for_high_z: bool = True

        # print(len(self.lidar_tokens))
        # print(len(self.cam_front_tokens))

        self.gt_poses = self._load_poses()

    def __len__(self):
        return len(self.lidar_tokens)

    def __getitem__(self, idx):
       
        points = self.read_point_cloud(self.lidar_tokens[idx])

        point_ts = self.get_timestamps(points)

        # use all the imgs
        img_front = self.read_img(self.cam_front_tokens[idx])
        img_front_left = self.read_img(self.cam_front_left_tokens[idx])
        img_front_right = self.read_img(self.cam_front_right_tokens[idx])
        img_back = self.read_img(self.cam_back_tokens[idx])
        img_back_left = self.read_img(self.cam_back_left_tokens[idx])
        img_back_right = self.read_img(self.cam_back_right_tokens[idx])
        # imgs as dictionary
        img_dict = {"cam_front": img_front, "cam_front_left": img_front_left, "cam_front_right": img_front_right,
                    "cam_back": img_back, "cam_back_left": img_back_left, "cam_back_right": img_back_right}

        # project to the image plane to get the corresponding color        
        points_rgb = np.ones_like(points)
        for cam_name in list(img_dict.keys()):
            points_rgb = self.project_points_to_cam(points, points_rgb, img_dict[cam_name], self.T_c_l_mats[cam_name], self.K_mats[cam_name])

        if self.use_only_colorized_points:
            with_rgb_mask = (points_rgb[:, 3] == 0)
            points = points[with_rgb_mask]
            points_rgb = points_rgb[with_rgb_mask]
            point_ts = point_ts[with_rgb_mask]

        # we skip the intensity here for now (and also the color mask)
        points = np.hstack((points[:,:3], points_rgb[:,:3]))

        frame_data = {"points": points, "point_ts": point_ts, "img": img_dict}

        return frame_data
    
    # velodyne lidar
    @staticmethod
    def get_timestamps(points):
        x = points[:, 0]
        y = points[:, 1]
        yaw = -np.arctan2(y, x)
        timestamps = 0.5 * (yaw / np.pi + 1.0)
        return timestamps

    def read_point_cloud(self, token: str):
        filename = self.nusc.get(self.used_part, token)["filename"]
        # print("LiDAR name:", filename)
        pcl = self.load_point_cloud(os.path.join(self.nusc.dataroot, filename))
        points = pcl.points.T[:, :4] # include intensity
        return points.astype(np.float64)
    
    def read_img(self, token: str):
        filename = self.nusc.get(self.used_part, token)["filename"]
        # print("Img name:", filename)
        file_path = os.path.join(self.nusc.dataroot, filename)
        # print(file_path)

        img = cv2.imread(file_path)
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img

    def _load_poses(self) -> np.ndarray:
        from nuscenes.utils.geometry_utils import transform_matrix
        from pyquaternion import Quaternion

        sd_record_lid = self.nusc.get(self.used_part, self.lidar_tokens[0])
        cs_record_lid = self.nusc.get(
                "calibrated_sensor", sd_record_lid["calibrated_sensor_token"]
            )
        car_to_velo = transform_matrix(
                cs_record_lid["translation"],
                Quaternion(cs_record_lid["rotation"]),
            ) # lidar frame to car body frame # T_b_l

        poses = np.empty((len(self), 4, 4), dtype=np.float32)
        for i, lidar_token in enumerate(self.lidar_tokens):
            sd_record_lid = self.nusc.get(self.used_part, lidar_token)
            ep_record_lid = self.nusc.get("ego_pose", sd_record_lid["ego_pose_token"]) # ego_pose per frame (or keyframe)

            pose_car = transform_matrix(
                ep_record_lid["translation"],
                Quaternion(ep_record_lid["rotation"]),
            ) # in car body frame

            poses[i:, :] = pose_car @ car_to_velo # poses in LiDAR frame T_w_b @ T_b_l = T_w_l

        # print(poses) # global coordinate

        # Convert from global coordinate poses to local poses
        first_pose = poses[0, :, :]
        poses = np.linalg.inv(first_pose) @ poses
        return poses
    
    def _load_calib(self):
        from nuscenes.utils.geometry_utils import transform_matrix
        from pyquaternion import Quaternion

        # LiDAR
        sd_record_lid = self.nusc.get(self.used_part, self.lidar_tokens[0])
        cs_record_lid = self.nusc.get(
                "calibrated_sensor", sd_record_lid["calibrated_sensor_token"]
            )
        car_to_velo = transform_matrix(
                cs_record_lid["translation"],
                Quaternion(cs_record_lid["rotation"]),
            ) # lidar frame to car body frame # T_b_l
        self.T_b_l = car_to_velo
        
        # cam front
        sd_record_cam_front = self.nusc.get(self.used_part, self.cam_front_tokens[0])
        cs_record_cam_front = self.nusc.get(
                "calibrated_sensor", sd_record_cam_front["calibrated_sensor_token"]
            )
        self.T_b_cf = transform_matrix(
                cs_record_cam_front["translation"],
                Quaternion(cs_record_cam_front["rotation"]),
            ) # front cam frame to car body frame  # T_b_cf
        self.T_cf_l = np.linalg.inv(self.T_b_cf) @ self.T_b_l
        self.K_front = np.array(cs_record_cam_front["camera_intrinsic"])

        self.K_mats["cam_front"] = self.K_front 
        self.T_c_l_mats["cam_front"] = self.T_cf_l 

        # cam front left
        sd_record_cam_front_left = self.nusc.get(self.used_part, self.cam_front_left_tokens[0])
        cs_record_cam_front_left = self.nusc.get(
                "calibrated_sensor", sd_record_cam_front_left["calibrated_sensor_token"]
            )
        self.T_b_cfl = transform_matrix(
                cs_record_cam_front_left["translation"],
                Quaternion(cs_record_cam_front_left["rotation"]),
            ) # front cam frame to car body frame  # T_b_cfl
        self.T_cfl_l = np.linalg.inv(self.T_b_cfl) @ self.T_b_l
        self.K_front_left = np.array(cs_record_cam_front_left["camera_intrinsic"])

        self.K_mats["cam_front_left"] = self.K_front_left 
        self.T_c_l_mats["cam_front_left"] = self.T_cfl_l 

        # cam front right
        sd_record_cam_front_right = self.nusc.get(self.used_part, self.cam_front_right_tokens[0])
        cs_record_cam_front_right = self.nusc.get(
                "calibrated_sensor", sd_record_cam_front_right["calibrated_sensor_token"]
            )
        self.T_b_cfr = transform_matrix(
                cs_record_cam_front_right["translation"],
                Quaternion(cs_record_cam_front_right["rotation"]),
            ) # front cam frame to car body frame  # T_b_cfr
        self.T_cfr_l = np.linalg.inv(self.T_b_cfr) @ self.T_b_l
        self.K_front_right = np.array(cs_record_cam_front_right["camera_intrinsic"])

        self.K_mats["cam_front_right"] = self.K_front_right 
        self.T_c_l_mats["cam_front_right"] = self.T_cfr_l 

        # cam back
        sd_record_cam_back = self.nusc.get(self.used_part, self.cam_back_tokens[0])
        cs_record_cam_back = self.nusc.get(
                "calibrated_sensor", sd_record_cam_back["calibrated_sensor_token"]
            )
        self.T_b_cb = transform_matrix(
                cs_record_cam_back["translation"],
                Quaternion(cs_record_cam_back["rotation"]),
            ) # front cam frame to car body frame  # T_b_cb
        self.T_cb_l = np.linalg.inv(self.T_b_cb) @ self.T_b_l
        self.K_back = np.array(cs_record_cam_back["camera_intrinsic"])

        self.K_mats["cam_back"] = self.K_back
        self.T_c_l_mats["cam_back"] = self.T_cb_l 

        # cam back left
        sd_record_cam_back_left = self.nusc.get(self.used_part, self.cam_back_left_tokens[0])
        cs_record_cam_back_left = self.nusc.get(
                "calibrated_sensor", sd_record_cam_back_left["calibrated_sensor_token"]
            )
        self.T_b_cbl = transform_matrix(
                cs_record_cam_back_left["translation"],
                Quaternion(cs_record_cam_back_left["rotation"]),
            ) # front cam frame to car body frame  # T_b_cbl
        self.T_cbl_l = np.linalg.inv(self.T_b_cbl) @ self.T_b_l
        self.K_back_left = np.array(cs_record_cam_back_left["camera_intrinsic"])

        self.K_mats["cam_back_left"] = self.K_back_left
        self.T_c_l_mats["cam_back_left"] = self.T_cbl_l 

        # cam back right
        sd_record_cam_back_right = self.nusc.get(self.used_part, self.cam_back_right_tokens[0])
        cs_record_cam_back_right = self.nusc.get(
                "calibrated_sensor", sd_record_cam_back_right["calibrated_sensor_token"]
            )
        self.T_b_cbr = transform_matrix(
                cs_record_cam_back_right["translation"],
                Quaternion(cs_record_cam_back_right["rotation"]),
            ) # front cam frame to car body frame  # T_b_cbl
        self.T_cbr_l = np.linalg.inv(self.T_b_cbr) @ self.T_b_l
        self.K_back_right = np.array(cs_record_cam_back_right["camera_intrinsic"])

        self.K_mats["cam_back_right"] = self.K_back_right
        self.T_c_l_mats["cam_back_right"] = self.T_cbr_l 


    def _get_scene_token(self, split_logs: List[str]) -> str:
        """
        Convenience function to get the samples in a particular split.
        :param split_logs: A list of the log names in this split.
        :return: The list of samples.
        """
        scene_tokens = [s["token"] for s in self.nusc.scene if s["name"] == self.scene_name][0]
        scene = self.nusc.get("scene", scene_tokens)
        log = self.nusc.get("log", scene["log_token"])
        return scene["token"] if log["logfile"] in split_logs else ""

    def _get_sensor_tokens(self, scene_token: str, sensor_name: str, keyframe_only = True) -> List[str]:
        # Get records from DB.
        scene_rec = self.nusc.get("scene", scene_token)
        start_sample_rec = self.nusc.get("sample", scene_rec["first_sample_token"])
        sd_rec = self.nusc.get(self.used_part, start_sample_rec["data"][sensor_name])

        # Make list of frames
        cur_sd_rec = sd_rec
        sd_tokens = [cur_sd_rec["token"]]
        while cur_sd_rec["next"] != "":
            cur_sd_rec = self.nusc.get(self.used_part, cur_sd_rec["next"])
            if not keyframe_only or cur_sd_rec["is_key_frame"]:
                sd_tokens.append(cur_sd_rec["token"])
        return sd_tokens
    
    def project_points_to_cam(self, points, points_rgb, img, T_c_l, K_mat):
        
        # points as np.numpy (N,4)

        # cm = plt.get_cmap('jet')
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
        depth_map = np.zeros((img_height, img_width))
        #
        mask = np.logical_and(np.logical_and(np.logical_and(u>=0, u<img_width), v>=0), v<img_height)
        
        # visualize points within 30 meters
        min_depth = 1.0
        max_depth = 100.0
        mask = np.logical_and(np.logical_and(mask, depth>min_depth), depth<max_depth)
        
        v_valid = v[mask]
        u_valid = u[mask]

        depth_map[v_valid,u_valid] = depth[mask]

        # print(np.shape(points_rgb))

        points_rgb[mask, :3] = img[v_valid,u_valid].astype(np.float64)/255.0 # 0-1
        points_rgb[mask, 3] = 0 # has color

        return points_rgb
    
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
