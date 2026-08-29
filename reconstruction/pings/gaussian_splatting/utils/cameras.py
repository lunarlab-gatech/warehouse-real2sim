#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import random

import torch
from torch import nn
import torch.nn.functional as F

from gaussian_splatting.utils.graphics_utils import getProjectionMatrix, focal2fov


# used by us
class CamImage:
    def __init__(self, frame_id, rgb_image, K_mat, z_min=0.1, z_max=100.0,
        cam_id: str = "cam", img_down_rate = 0,
        depth_image = None, normal_img = None, sky_mask = None, obj_mask = None,
        device = "cuda", cam_pose = None, img_width = None, img_height = None, pyramid_level: int = 4):
        
        self.frame_id = frame_id
        self.cam_id = cam_id
        self.uid = f"{frame_id:05d}_{cam_id}"

        self.device = device
        self.dtype = torch.float32

        self.train_view: bool = False # is used as train view or test view
        self.in_long_term_memory: bool = False

        if rgb_image is not None:
            rgb_image = rgb_image.clamp(0.0, 1.0) # only for the RGB part
            self.image_width = rgb_image.shape[2]
            self.image_height = rgb_image.shape[1]
        else:
            self.image_width = img_width
            self.image_height = img_height
        
        # if input rgb_image is None, then you need to input valid img_width and height

        # numpy array
        self.K_mat = K_mat
        self.fx = K_mat[0,0]
        self.fy = K_mat[1,1]
        self.cx = K_mat[0,2]
        self.cy = K_mat[1,2]

        self.K_mat_torch = torch.tensor(K_mat, dtype=self.dtype, device=self.device)

        self.FoVx = focal2fov(self.fx, self.image_width)
        self.FoVy = focal2fov(self.fy, self.image_height)

        # principle point (not always at the center) as a ratio, like 0.5, 0.5
        self.prcppoint = torch.tensor([self.cx / self.image_width, self.cy / self.image_height]).to(dtype=self.dtype, device=self.device)

        self.zfar = z_max # 100.0
        self.znear = z_min # 0.1

        # GL
        # OpenGL projection matrix
        self.projection_matrix = (getProjectionMatrix(znear=self.znear, zfar=self.zfar,
             fovX=self.FoVx, fovY=self.FoVy,
              W=self.image_width, H=self.image_height, prcp=self.prcppoint).T).to(dtype=self.dtype, device=self.device) # T_gi        
        
        self.world_view_transform = None # as (T_cw.T)
        self.camera_center = None
        self.full_proj_transform = None 
        
        # set the poses related transformations

        # init value
        self.R = torch.eye(3, dtype=self.dtype, device=self.device)
        # this is not the camera center in world frame, use camera_center instead
        self.T = torch.zeros(3, dtype=self.dtype, device=self.device) 

        self.set_pose(cam_pose)

        # pyramid of images
        self.pyramid_level = pyramid_level
        self.rgb_image_list = [None] * pyramid_level
        self.depth_image_list = [None] * pyramid_level
        self.sky_mask_list = [None] * pyramid_level
        self.obj_mask_list = [None] * pyramid_level
        self.normal_img_list = [None] * pyramid_level

        # safe defaults for cameras constructed without an rgb image (pose-only /
        # tracker path); the branch below overwrites them when an image is attached
        self.depth_on = False
        self.sky_mask_on = False
        self.obj_mask_on = False
        self.mono_normal_on = False

        # camera pose optimization
        self.cam_rot_delta = nn.Parameter(
            torch.zeros(3, requires_grad=True, device=device)
        )
        self.cam_trans_delta = nn.Parameter(
            torch.zeros(3, requires_grad=True, device=device)
        )

        # exposure correction parameters
        self.exposure_a = nn.Parameter(
            torch.tensor([0.0], requires_grad=True, device=device)
        )
        self.exposure_b = nn.Parameter(
            torch.tensor([0.0], requires_grad=True, device=device)
        )

        # exposure correction affine transformation parameters
        self.exposure_mat = nn.Parameter(
            torch.eye(3, requires_grad=True, device=device)
        )
        self.exposure_offset = nn.Parameter(
            torch.zeros(3, requires_grad=True, device=device)
        )

        if rgb_image is not None:
            rgb_image = rgb_image.to(self.device)
            
            if depth_image is not None: # 1, H, W
                depth_image = depth_image.to(self.device)
                self.depth_on = True
            else:
                self.depth_on = False

            # TODO: may add normal
            # TODO: the issue of the depth rendering loss lie in the depth image downsampling, bilinear may not be a good idea, update it 
            
            # C can be either 3 or 4
            # NOTE: F.interpolate require 4D input, it's not an in-place operation, the original input tensor remains unchanged
            # Downsample to Cx(H/2)x(W/2)
            down_level1_image = F.interpolate(rgb_image.unsqueeze(0), scale_factor=0.5, mode='bilinear', align_corners=False).squeeze(0)
            # Downsample to Cx(H/4)x(W/4)
            down_level2_image = F.interpolate(down_level1_image.unsqueeze(0), scale_factor=0.5, mode='bilinear', align_corners=False).squeeze(0)
            # Downsample to Cx(H/8)x(W/8)
            down_level3_image = F.interpolate(down_level2_image.unsqueeze(0), scale_factor=0.5, mode='bilinear', align_corners=False).squeeze(0)

            if self.depth_on:  # 1, H, W
                down_level1_depth = F.interpolate(depth_image.unsqueeze(0), scale_factor=0.5, mode='nearest-exact').squeeze(0)
                down_level2_depth = F.interpolate(down_level1_depth.unsqueeze(0), scale_factor=0.5, mode='nearest-exact').squeeze(0)
                down_level3_depth = F.interpolate(down_level2_depth.unsqueeze(0), scale_factor=0.5, mode='nearest-exact').squeeze(0)
            else:
                down_level1_depth = down_level2_depth = down_level3_depth = None

            if sky_mask is not None: # sky_mask 1, H, W
                self.sky_mask_on = True
                sky_mask = sky_mask.to(self.device)
                down_level1_sky_mask = F.interpolate(sky_mask.float().unsqueeze(0), scale_factor=0.5, mode='nearest').squeeze(0).bool()
                down_level2_sky_mask = F.interpolate(down_level1_sky_mask.float().unsqueeze(0), scale_factor=0.5, mode='nearest').squeeze(0).bool()
                down_level3_sky_mask = F.interpolate(down_level2_sky_mask.float().unsqueeze(0), scale_factor=0.5, mode='nearest').squeeze(0).bool()
            else:
                down_level1_sky_mask = down_level2_sky_mask = down_level3_sky_mask = None
                self.sky_mask_on = False

            if obj_mask is not None: # obj_mask 1, H, W (masked object pixels get NO supervision)
                self.obj_mask_on = True
                obj_mask = obj_mask.to(self.device)
                down_level1_obj_mask = F.interpolate(obj_mask.float().unsqueeze(0), scale_factor=0.5, mode='nearest').squeeze(0).bool()
                down_level2_obj_mask = F.interpolate(down_level1_obj_mask.float().unsqueeze(0), scale_factor=0.5, mode='nearest').squeeze(0).bool()
                down_level3_obj_mask = F.interpolate(down_level2_obj_mask.float().unsqueeze(0), scale_factor=0.5, mode='nearest').squeeze(0).bool()
            else:
                down_level1_obj_mask = down_level2_obj_mask = down_level3_obj_mask = None
                self.obj_mask_on = False

            if normal_img is not None: # normal already in device  # sky_mask 3, H, W
                self.mono_normal_on = True
                normal_img = normal_img.to(self.device)
                down_level1_normal = F.interpolate(normal_img.unsqueeze(0), scale_factor=0.5, mode='bilinear', align_corners=False).squeeze(0)
                down_level2_normal = F.interpolate(down_level1_normal.unsqueeze(0), scale_factor=0.5, mode='bilinear', align_corners=False).squeeze(0)
                down_level3_normal = F.interpolate(down_level2_normal.unsqueeze(0), scale_factor=0.5, mode='bilinear', align_corners=False).squeeze(0)
            else:
                down_level1_normal = down_level2_normal = down_level3_normal = None
                self.mono_normal_on = False

            self.rgb_image_list[0] = rgb_image
            self.depth_image_list[0] = depth_image
            self.sky_mask_list[0] = sky_mask
            self.obj_mask_list[0] = obj_mask
            self.normal_img_list[0] = normal_img

            self.rgb_image_list[1] = down_level1_image
            self.depth_image_list[1] = down_level1_depth
            self.sky_mask_list[1] = down_level1_sky_mask
            self.obj_mask_list[1] = down_level1_obj_mask
            self.normal_img_list[1] = down_level1_normal

            self.rgb_image_list[2] = down_level2_image
            self.depth_image_list[2] = down_level2_depth
            self.sky_mask_list[2] = down_level2_sky_mask
            self.obj_mask_list[2] = down_level2_obj_mask
            self.normal_img_list[2] = down_level2_normal

            self.rgb_image_list[3] = down_level3_image
            self.depth_image_list[3] = down_level3_depth
            self.sky_mask_list[3] = down_level3_sky_mask
            self.obj_mask_list[3] = down_level3_obj_mask
            self.normal_img_list[3] = down_level3_normal

            self.cur_best_level: int = 0

            self.free_memory_under_levels(img_down_rate-1)
    
    # deprecated
    def random_patch(self, h_size=float('inf'), w_size=float('inf')):
        # just use part (a random patch) of the image
        h = self.image_height
        w = self.image_width
        h_size = min(h_size, h) # h
        w_size = min(w_size, w) # w
        h0 = random.randint(0, h - h_size) # 0
        w0 = random.randint(0, w - w_size) # 0
        h1 = h0 + h_size - 1 
        w1 = w0 + w_size - 1
        return torch.tensor([h0, w0, h1, w1]).to(dtype=self.dtype, device=self.device)

    def full_patch(self, img_down_rate: int = 0):
        img_down_scale = 2**(img_down_rate)
        h1 = int(self.image_height / img_down_scale) - 1
        w1 = int(self.image_width / img_down_scale) - 1
        return torch.tensor([0, 0, h1, w1]).to(dtype=self.dtype, device=self.device)

    def set_pose(self, cam_pose):
        # input pose is torch tensor, and is T_w_c (c to w)

        if cam_pose is not None: # we also directly load the camera pose here
                    
            T_cw = torch.linalg.inv(cam_pose).to(dtype=self.dtype, device=self.device) 

            self.world_view_transform = (T_cw.T)
            self.camera_center = torch.linalg.inv(self.world_view_transform)[3, :3]
            self.full_proj_transform = self.world_view_transform @ self.projection_matrix 
            
            self.R = T_cw[:3, :3] # rotation part
            self.T = T_cw[:3, 3] # translation part

    def set_exposure_ab(self, exposure_a, exposure_b):
        self.exposure_a = exposure_a
        self.exposure_b = exposure_b

    def set_exposure_affine(self, exposure_mat, exposure_offset):
        self.exposure_mat = exposure_mat
        self.exposure_offset = exposure_offset

    def set_delta_pose(self, delta_r, delta_t):
        self.cam_rot_delta = delta_r
        self.cam_trans_delta = delta_t

    def set_depth_img(self, depth_img_torch):
        if depth_img_torch is not None:  # 1, H, W
            self.depth_on = True
            depth_img_torch = depth_img_torch.to(self.device)   
            down_level1_depth = F.interpolate(depth_img_torch.unsqueeze(0), scale_factor=0.5, mode='nearest-exact').squeeze(0)
            down_level2_depth = F.interpolate(down_level1_depth.unsqueeze(0), scale_factor=0.5, mode='nearest-exact').squeeze(0)
            down_level3_depth = F.interpolate(down_level2_depth.unsqueeze(0), scale_factor=0.5, mode='nearest-exact').squeeze(0)
            self.depth_image_list[0] = depth_img_torch
            self.depth_image_list[1] = down_level1_depth
            self.depth_image_list[2] = down_level2_depth
            self.depth_image_list[3] = down_level3_depth

    def free_memory_at_level(self, down_level_to_free: int = 0):
        if len(self.rgb_image_list) > down_level_to_free and down_level_to_free >= 0:
            self.rgb_image_list[down_level_to_free] = None
            self.depth_image_list[down_level_to_free] = None
            self.normal_img_list[down_level_to_free] = None
            self.sky_mask_list[down_level_to_free] = None
            self.obj_mask_list[down_level_to_free] = None
            self.cur_best_level = down_level_to_free + 1

    def free_memory_under_levels(self, hightest_down_level_to_free: int = 0):
        free_levels = min(hightest_down_level_to_free+1, len(self.rgb_image_list))
        if free_levels >= 1:
            for l in range(free_levels):
                self.free_memory_at_level(l)
        self.cur_best_level = hightest_down_level_to_free+1

