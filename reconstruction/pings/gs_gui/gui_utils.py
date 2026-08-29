# This file is adapted from the GUI of [MonoGS](https://github.com/muskie82/MonoGS)
# Yue Pan, 2025

import queue

import copy
import cv2
import numpy as np
import open3d as o3d
import torch

from utils.tools import feature_pca_torch

from gaussian_splatting.utils.general_utils import (
    build_scaling_rotation,
    strip_symmetric,
)

cv_gl = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

# actually not only gaussians, we also send the cameras, point cloud and mesh data
class VisPacket:
    def __init__(
        self,
        frame_id = None,
        current_frames=None,
        gaussian_xyz=None,
        gaussian_scale=None,
        gaussian_rot=None,
        gaussian_alpha=None,
        gaussian_color=None,
        keyframes=None,
        finish=False,
        kf_window=None,
        current_pointcloud_xyz=None,
        current_pointcloud_rgb=None,
        current_rendered_xyz=None,
        current_rendered_rgb=None,
        mesh_verts=None,
        mesh_faces=None,
        mesh_verts_rgb=None,
        odom_poses=None,
        gt_poses=None,
        slam_poses=None,
        local_only=True,
        gpu_mem_usage_gb=None,
        img_down_rate=0,
    ):
        self.has_gaussians = False
        self.has_neural_points = False
        self.has_sorrounding_points = False

        self.neural_points_data = None
        self.sorrounding_neural_points_data = None

        self.local_gaussian_count = 0

        self.frame_id = frame_id

        if gaussian_xyz is not None:
            self.has_gaussians = True

            self.dtype = gaussian_xyz.dtype
            self.device = gaussian_xyz.device

            self.max_sh_degree = 0
            self.active_sh_degree = 0

            self.gaussian_xyz = gaussian_xyz.detach()
            self.gaussian_scale = gaussian_scale.detach()
            self.gaussian_rot = gaussian_rot.detach()
            self.gaussian_alpha = gaussian_alpha.detach()
            self.gaussian_color = gaussian_color.detach()

            self.local_gaussian_count = self.gaussian_xyz.shape[0]

        self.current_frames = current_frames # as Dict, key use cam_id
        
        self.keyframes = keyframes # as Dict, key use uid

        self.cam_list = {}
        if current_frames is not None:
            self.cam_list = list(current_frames.keys())            

        self.keyframe_list = {}
        if keyframes is not None:
            self.keyframe_list = list(keyframes.keys())

        # for cur frames and also for the train frames
        self.gtcolor = {}
        self.gtdepth = {}
        self.gtnormal = {}

        # add camera frames, set gtcolor, gtdepth, gtnormal
        self.add_cam_frames(current_frames, img_down_rate)

        # # add train frames, better to also set the gtcolor, gtdepth, gtnormal
        self.add_cam_frames(keyframes, img_down_rate) # could take too much memory (FIXME)
        
        self.add_scan(current_pointcloud_xyz, current_pointcloud_rgb)

        self.add_rendered_scan(current_rendered_xyz, current_rendered_rgb)

        self.add_mesh(mesh_verts, mesh_faces, mesh_verts_rgb)

        self.add_traj(odom_poses, gt_poses, slam_poses)

        self.img_down_rate = img_down_rate

        self.sdf_slice_xyz = None
        self.sdf_slice_rgb = None

        self.sdf_pool_xyz = None
        self.sdf_pool_rgb = None

        self.gpu_mem_usage_gb = gpu_mem_usage_gb

        self.kf_window = kf_window

        self.finish = finish


    def add_cam_frames(self, cam_frames, img_down_rate: int = 0):

        if cam_frames is not None:
            
            for cam in list(cam_frames.keys()):

                current_frame = cam_frames[cam]

                cur_img_down_rate = max(img_down_rate, current_frame.cur_best_level)
                
                if current_frame.rgb_image_list[cur_img_down_rate] is not None:
                    
                    gtcolor = current_frame.rgb_image_list[cur_img_down_rate]
                    if current_frame.sky_mask_on:
                        # mask the sky part
                        cur_sky_mask = current_frame.sky_mask_list[cur_img_down_rate] # still torch
                        cur_sky_mask_used = cur_sky_mask.expand(3, -1, -1)
                        gtcolor[cur_sky_mask_used] = 1.0
                    
                    gtcolor = self.resize_img(gtcolor)

                    # exposure correction for vis (don't apply this, we only apply exposure correction to render views to let them align with the real obseravtion)
                    # with torch.no_grad(): # - /
                    #     gtcolor = (gtcolor - current_frame.exposure_b) /  torch.exp(current_frame.exposure_a)

                    self.gtcolor[cam] = gtcolor
                    
                    if current_frame.depth_on:
                        gtdepth = current_frame.depth_image_list[cur_img_down_rate]
                        gtdepth = self.resize_img(gtdepth, is_sparse=True)
                    else:
                        gtdepth = None
                    self.gtdepth[cam] = gtdepth
                    
                    if current_frame.mono_normal_on:
                        gtnormal = current_frame.normal_img_list[cur_img_down_rate]
                        if current_frame.sky_mask_on:
                            # mask the sky part
                            gtnormal[cur_sky_mask_used] = 0.0
                        gtnormal = self.resize_img(gtnormal)    
                    else:
                        gtnormal = None
                    self.gtnormal[cam] = gtnormal


    # the sorrounding map is also added here
    def add_neural_points_data(self, neural_points, only_local_map: bool = True, 
                               add_sorrounding_points: bool = True,
                               pca_color_on: bool = True):
        
        if neural_points is not None:
            self.has_neural_points = True
            self.neural_points_data = {}
            self.neural_points_data["count"] = neural_points.count()
            self.neural_points_data["valid_count"] = neural_points.count(valid_gs_only=True)
            self.neural_points_data["local_count"] = neural_points.local_count()
            self.neural_points_data["valid_local_count"] = neural_points.local_count(valid_gs_only=True)
            self.neural_points_data["map_memory_mb"] = neural_points.cur_memory_mb
            self.neural_points_data["resolution"] = neural_points.resolution
            
            if only_local_map:
                self.neural_points_data["position"] = neural_points.local_neural_points
                self.neural_points_data["orientation"] = neural_points.local_point_orientations
                self.neural_points_data["geo_feature"] = neural_points.local_geo_features.detach()
                if neural_points.color_on:
                    self.neural_points_data["color"] = neural_points.local_point_colors
                    self.neural_points_data["color_feature"] = neural_points.local_color_features.detach()
                self.neural_points_data["free_mask"] = neural_points.local_free_gs_mask
                self.neural_points_data["valid_mask"] = neural_points.local_valid_gs_mask
                self.neural_points_data["ts"] = neural_points.local_point_ts_update
                self.neural_points_data["stability"] = neural_points.local_point_certainties

                if pca_color_on and self.neural_points_data["local_count"] > 10:
                    local_geo_feature_3d, _ = feature_pca_torch((self.neural_points_data["geo_feature"])[:-1], principal_components=neural_points.geo_feature_pca, down_rate=17)
                    self.neural_points_data["color_pca_geo"] = local_geo_feature_3d

                    if neural_points.color_on:
                        local_color_feature_3d, _ = feature_pca_torch((self.neural_points_data["color_feature"])[:-1], principal_components=neural_points.color_feature_pca, down_rate=17)
                        self.neural_points_data["color_pca_color"] = local_color_feature_3d

                if add_sorrounding_points:
                    sorrounding_mask = neural_points.sorrounding_mask
                    sorrounding_mask_a = sorrounding_mask[:-1]
                    if torch.sum(sorrounding_mask_a).item() > 10:
                        self.has_sorrounding_points = True
                        self.sorrounding_neural_points_data = {}
                        self.sorrounding_neural_points_data["center"] = neural_points.local_position
                        self.sorrounding_neural_points_data["position"] = neural_points.neural_points[sorrounding_mask_a]
                        self.sorrounding_neural_points_data["orientation"] = neural_points.point_orientations[sorrounding_mask_a]
                        self.sorrounding_neural_points_data["geo_feature"] = neural_points.geo_features[sorrounding_mask]
                        if neural_points.color_on:
                            self.sorrounding_neural_points_data["color"] = neural_points.point_colors[sorrounding_mask_a]
                            self.sorrounding_neural_points_data["color_feature"] = neural_points.color_features[sorrounding_mask]
                        self.sorrounding_neural_points_data["resolution"] = neural_points.resolution
                        self.sorrounding_neural_points_data["free_mask"] = neural_points.free_gs_mask[sorrounding_mask_a] # but now this is actually per neural point
                        self.sorrounding_neural_points_data["valid_mask"] = neural_points.valid_gs_mask[sorrounding_mask_a]

            else:
                self.neural_points_data["position"] = neural_points.neural_points
                self.neural_points_data["orientation"] = neural_points.point_orientations
                self.neural_points_data["geo_feature"] = neural_points.geo_features
                if neural_points.color_on:
                    self.neural_points_data["color"] = neural_points.point_colors
                    self.neural_points_data["color_feature"] = neural_points.color_features
                self.neural_points_data["free_mask"] = neural_points.free_gs_mask
                self.neural_points_data["valid_mask"] = neural_points.valid_gs_mask
                self.neural_points_data["ts"] = neural_points.point_ts_update
                self.neural_points_data["stability"] = neural_points.point_certainties
                
                if neural_points.local_mask is not None:
                    self.neural_points_data["local_mask"] = neural_points.local_mask[:-1]

                if pca_color_on:
                    geo_feature_3d, _ = feature_pca_torch(neural_points.geo_features[:-1], principal_components=neural_points.geo_feature_pca, down_rate=97)
                    self.neural_points_data["color_pca_geo"] = geo_feature_3d

                    if neural_points.color_on:
                        color_feature_3d, _ = feature_pca_torch(neural_points.color_features[:-1], principal_components=neural_points.color_feature_pca, down_rate=97)
                        self.neural_points_data["color_pca_color"] = color_feature_3d

            if neural_points.color_on:
                invalid_color_mask = (self.neural_points_data["color"][:,0] < 0)
                invalid_color_part = self.neural_points_data["color"][invalid_color_mask]
                self.neural_points_data["color"][invalid_color_mask] = torch.ones_like(invalid_color_part).to(invalid_color_part) # show as white instead of black


    def add_gaussians(self,  
                    gaussian_xyz=None,
                    gaussian_scale=None,
                    gaussian_rot=None,
                    gaussian_alpha=None,
                    gaussian_color=None):

        if gaussian_xyz is not None:
            self.has_gaussians = True

            self.dtype = gaussian_xyz.dtype
            self.device = gaussian_xyz.device

            # now set to 0 (TODO)
            self.max_sh_degree = 0
            self.active_sh_degree = 0

            self.gaussian_xyz = gaussian_xyz.detach()
            self.gaussian_scale = gaussian_scale.detach()
            self.gaussian_rot = gaussian_rot.detach()
            self.gaussian_alpha = gaussian_alpha.detach()
            self.gaussian_color = gaussian_color.detach()

            self.local_count = self.gaussian_xyz.shape[0]

    def add_scan(self, current_pointcloud_xyz=None, current_pointcloud_rgb=None):
        self.current_pointcloud_xyz = current_pointcloud_xyz
        self.current_pointcloud_rgb = current_pointcloud_rgb

        if current_pointcloud_rgb is not None:
            invalid_mask = (current_pointcloud_rgb[:,0] < 0)
            invalid_rgb = current_pointcloud_rgb[invalid_mask]
            self.current_pointcloud_rgb[invalid_mask] = np.ones_like(invalid_rgb)
        # show invalid as white (all 1) instead of black

        # TODO: add normal later

    def add_rendered_scan(self, current_rendered_xyz=None, current_rendered_rgb=None):
        self.current_rendered_xyz = current_rendered_xyz
        self.current_rendered_rgb = current_rendered_rgb
        # TODO: add normal later

    def add_sdf_slice(self, sdf_slice_xyz=None, sdf_slice_rgb=None):
        self.sdf_slice_xyz = sdf_slice_xyz
        self.sdf_slice_rgb = sdf_slice_rgb

    def add_sdf_training_pool(self, sdf_pool_xyz=None, sdf_pool_rgb=None):
        self.sdf_pool_xyz = sdf_pool_xyz
        self.sdf_pool_rgb = sdf_pool_rgb

    def add_mesh(self, mesh_verts=None, mesh_faces=None, mesh_verts_rgb=None):
        self.mesh_verts = mesh_verts
        self.mesh_faces = mesh_faces
        self.mesh_verts_rgb = mesh_verts_rgb

    def add_traj(self, odom_poses=None, gt_poses=None, slam_poses=None, loop_edges=None):
        self.odom_poses = odom_poses
        self.gt_poses = gt_poses
        self.slam_poses = slam_poses

        if slam_poses is None:
            self.slam_poses = odom_poses

        self.loop_edges = loop_edges

    def resize_img(self, img, resize_width = None, is_sparse: bool = False):
        if img is None:
            return None
        
        if resize_width is None:
            return img

        # check if img is numpy
        if isinstance(img, np.ndarray):
            height = int(resize_width * img.shape[0] / img.shape[1])
            return cv2.resize(img, (resize_width, height))
        # or as torch
        resize_height = int(resize_width * img.shape[1] / img.shape[2])
        # img is 3xHxW
        if is_sparse: # check this
            img = torch.nn.functional.interpolate(img.unsqueeze(0), size=(resize_height, resize_width), mode='nearest-exact')
        else:
            img = torch.nn.functional.interpolate(img.unsqueeze(0), size=(resize_height, resize_width), mode="bilinear", align_corners=False)

        return img.squeeze(0)

    # deprecated
    def get_covariance(self, scaling_modifier=1):
        return self.build_covariance_from_scaling_rotation(
            self.get_xyz, self.get_scaling, scaling_modifier, self.rotation
        )

    # this is for 3D GS, update the version for 2D GS
    def build_covariance_from_scaling_rotation(
        self, center, scaling, scaling_modifier, rotation
    ): # center not used
        L = build_scaling_rotation(scaling_modifier * scaling, rotation)
        actual_covariance = L @ L.transpose(1, 2)
        symm = strip_symmetric(actual_covariance)
        return symm


def get_latest_queue(q):
    message = None
    while True:
        try:
            message_latest = q.get_nowait()
            if message is not None:
                del message
            message = message_latest
        except queue.Empty:
            if q.empty():
                break
    return message


class ControlPacket:
    flag_pause = False
    flag_vis = True
    flag_mesh = False
    flag_sdf = False
    flag_global = False
    flag_source = False
    mc_res_m = 0.2
    mesh_min_nn = 10
    mesh_freq_frame = 10
    sdf_freq_frame = 1
    sdf_slice_height = -1.0
    sdf_res_m = 0.1
    cur_frame_id = 0

class ParamsGUI:
    def __init__(
        self,
        decoders=None,
        background=None,
        q_main2vis=None,
        q_vis2main=None,
        config=None, # PINGS configs
        is_rgbd: bool = False,
        gs_default_on: bool = False,
        local_map_default_on: bool = True,
        still_view_default_on: bool = True,
        robot_default_on: bool = True,
        follow_cam_default_on: bool = True,
        neural_point_map_default_on: bool = False,
        mesh_default_on: bool = False,
        sdf_default_on: bool = False,
        neural_point_color_default_mode: int = 1, # 0: original rgb, 1: geo feature pca, 2: photo feature pca, 3: time, 4: stability
        neural_point_vis_down_rate: int = 1,
        frustum_size: float = 0.05,
    ):
        self.decoders = decoders # dict of MLPs
        
        self.background = background
        self.q_main2vis = q_main2vis
        self.q_vis2main = q_vis2main
        self.config = config

        self.is_rgbd = is_rgbd
        self.gs_default_on = gs_default_on
        self.local_map_default_on = local_map_default_on
        self.still_view_default_on = still_view_default_on
        self.follow_cam_default_on = follow_cam_default_on
        self.robot_default_on = robot_default_on
        self.neural_point_map_default_on = neural_point_map_default_on
        self.mesh_default_on = mesh_default_on
        self.sdf_default_on = sdf_default_on
        self.neural_point_color_default_mode = neural_point_color_default_mode
        self.neural_point_vis_down_rate = neural_point_vis_down_rate
       
        self.frustum_size = frustum_size


class Frustum:
    def __init__(self, line_set, view_dir=None, view_dir_behind=None, size=None):
        self.line_set_origin = line_set
        self.view_dir = view_dir
        self.view_dir_behind = view_dir_behind
        self.size = size

    def update_pose(self, pose):

        self.line_set = copy.deepcopy(self.line_set_origin)
        self.line_set.transform(pose)

        points = np.asarray(self.line_set.points)
        # points_hmg = np.hstack([points, np.ones((points.shape[0], 1))])
        # points = (pose @ points_hmg.transpose())[0:3, :].transpose()

        base = np.array([[0.0, 0.0, 0.0]]) * self.size
        base_hmg = np.hstack([base, np.ones((base.shape[0], 1))])
        cameraeye = pose @ base_hmg.transpose()
        cameraeye = cameraeye[0:3, :].transpose()
        eye = cameraeye[0, :]

        # base_behind = np.array([[0.0, -2.5, -30.0]]) * self.size # original z -30.0
        base_behind = np.array([[0.0, -1.0, -15.0]]) * self.size 

        base_behind_hmg = np.hstack([base_behind, np.ones((base_behind.shape[0], 1))])
        cameraeye_behind = pose @ base_behind_hmg.transpose()
        cameraeye_behind = cameraeye_behind[0:3, :].transpose()
        eye_behind = cameraeye_behind[0, :]

        center = np.mean(points[1:, :], axis=0)
        up = points[2] - points[4]

        self.view_dir = (center, eye, up, pose)
        self.view_dir_behind = (center, eye_behind, up, pose)

        self.center = center
        self.eye = eye
        self.up = up

# camera frustum
def create_frustum(pose, frusutum_color=[0, 1, 0], size=0.02, h_w_ratio = 0.5, z_ratio = 1.5): 
    points = (
        np.array(
            [
                [0.0, 0.0, 0],
                [1.0, -h_w_ratio, z_ratio],
                [-1.0, -h_w_ratio, z_ratio],
                [1.0, h_w_ratio, z_ratio],
                [-1.0, h_w_ratio, z_ratio],
            ]
        )
        * size # too small
    )

    lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [1, 3], [2, 4], [3, 4]]
    colors = [frusutum_color for i in range(len(lines))]

    canonical_line_set = o3d.geometry.LineSet()
    canonical_line_set.points = o3d.utility.Vector3dVector(points)
    canonical_line_set.lines = o3d.utility.Vector2iVector(lines)
    canonical_line_set.colors = o3d.utility.Vector3dVector(colors)
    frustum = Frustum(canonical_line_set, size=size)
    frustum.update_pose(pose)
    return frustum
