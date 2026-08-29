import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os, cv2
import matplotlib.pyplot as plt
import math

from gaussian_splatting.utils.graphics_utils import fov2focal


# used by 2D GS
def depths_to_points(camera, depth, in_cam_frame: bool = False, img_scale: int = 1):
    """
        camera: view camera
        depth: depthmap 
    """
    # device = view.device

    # print(camera.world_view_transform)
    
    if in_cam_frame:
        c2w = torch.eye(4).to(camera.full_proj_transform)
    else:
        assert camera.world_view_transform is not None, "camera.world_view_transform is None"
        c2w = torch.linalg.inv(camera.world_view_transform.T)

    W, H = depth.shape[2], depth.shape[1]

    # ndc2pix = torch.tensor([
    #     [W / 2, 0, 0, (W) / 2],
    #     [0, H / 2, 0, (H) / 2],
    #     [0, 0, 0, 1]]).float().cuda().T
    # projection_matrix = c2w.T @ camera.full_proj_transform
    
    # projection_matrix = camera.projection_matrix
    # intrins = (projection_matrix @ ndc2pix)[:3,:3].T

    # print("Intrinsics:")
    # print(intrins)

    # print("K_mat:")
    # print(camera.K_mat_torch)

    intrins = camera.K_mat_torch
    intrins[0,0] /= img_scale
    intrins[1,1] /= img_scale
    intrins[0,2] /= img_scale
    intrins[1,2] /= img_scale
    
    grid_x, grid_y = torch.meshgrid(torch.arange(W, device='cuda').float(), torch.arange(H, device='cuda').float(), indexing='xy')
    points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(-1, 3)
    rays_d = points @ torch.linalg.inv(intrins).T @ c2w[:3,:3].T
    rays_o = c2w[:3,3]
    points = depth.reshape(-1, 1) * rays_d + rays_o

    return points

# used by 2D GS
def depth_to_normal(camera, depth, in_cam_frame: bool = False, img_scale: int = 1):
    """
        camera: view camera
        depth: rendered depthmap 
        the output normal is in the world frame  3, H, W 
    """
    # print(depth.shape)
    points = depths_to_points(camera, depth, in_cam_frame, img_scale).reshape(*depth.shape[1:], 3) # already in world frame
    output = torch.zeros_like(points)
    dx = torch.cat([points[2:, 1:-1] - points[:-2, 1:-1]], dim=0)
    dy = torch.cat([points[1:-1, 2:] - points[1:-1, :-2]], dim=1)
    # this is only done once, not like depth2normal functions
    # better to have a mask
    normal_map = torch.nn.functional.normalize(torch.linalg.cross(dx, dy, dim=-1), dim=-1) # norm = 1
    output[1:-1, 1:-1, :] = normal_map # boundary still zero, no padding, better to mask there
    # as the gradient of depth 
    # pointing towards the surface

    output = output.permute([2,0,1]) # convert to 3, H, W # in world frame

    return output

# used by Gaussian Surfels
def depth2normal(depth, mask, camera, img_scale: int = 1):
    """
        depth: rendered depthmap 
        mask: visible mask
        camera: view camera
        the output normal is in the camera frame  3, H, W 
    """
    # convert to camera position
    camD = depth.permute([1, 2, 0])
    mask = mask.permute([1, 2, 0])
    shape = camD.shape # H, W, 1
    device = camD.device 
    h, w, _ = torch.meshgrid(torch.arange(0, shape[0]), torch.arange(0, shape[1]), torch.arange(0, shape[2]), indexing='ij')
    # print(h)
    h = h.to(torch.float32).to(device)
    w = w.to(torch.float32).to(device)
    p = torch.cat([w, h], axis=-1)

    # cx, cy
    p[..., 0:1] -= camera.prcppoint[0] * camera.image_width / img_scale
    p[..., 1:2] -= camera.prcppoint[1] * camera.image_height / img_scale
    p *= camD

    K00 = camera.fx / img_scale
    K11 = camera.fy / img_scale

    K = torch.tensor([K00, 0, 0, K11]).reshape([2,2])
    Kinv = torch.inverse(K).to(torch.float32).to(device)
    # print(p.shape, Kinv.shape)
    p = p @ Kinv.t() # unprojected to 3D, still in camera frame
    camPos = torch.cat([p, camD], -1) # position under camera frame

    # padded = mod.contour_padding(camPos.contiguous(), mask.contiguous(), torch.zeros_like(camPos), filter_size // 2)
    # camPos = camPos + padded
    p = torch.nn.functional.pad(camPos[None], [0, 0, 1, 1, 1, 1], mode='replicate')
    mask = torch.nn.functional.pad(mask[None].to(torch.float32), [0, 0, 1, 1, 1, 1], mode='replicate').to(torch.bool)
    
    p_c = (p[:, 1:-1, 1:-1, :]      ) * mask[:, 1:-1, 1:-1, :]
    p_u = (p[:,  :-2, 1:-1, :] - p_c) * mask[:,  :-2, 1:-1, :]
    p_l = (p[:, 1:-1,  :-2, :] - p_c) * mask[:, 1:-1,  :-2, :]
    p_b = (p[:, 2:  , 1:-1, :] - p_c) * mask[:, 2:  , 1:-1, :]
    p_r = (p[:, 1:-1, 2:  , :] - p_c) * mask[:, 1:-1, 2:  , :]

    n_ul = torch.linalg.cross(p_u, p_l) # changed for torch.cross [FIXME](yue)
    n_ur = torch.linalg.cross(p_r, p_u)
    n_br = torch.linalg.cross(p_b, p_r)
    n_bl = torch.linalg.cross(p_l, p_b)
    # the finally result would be the average of these four
    
    n = n_ul + n_ur + n_br + n_bl
    n = n[0] # what does this mean?
    
    # n *= -torch.sum(camVDir * camN, -1, True).sign() # no cull back

    mask = mask[0, 1:-1, 1:-1, :]

    # n = gaussian_blur(n, filter_size, 1) * mask

    n = torch.nn.functional.normalize(n, dim=-1)
    # n[..., 1] *= -1
    # n *= -1

    n = (n * mask).permute([2, 0, 1]) # 3, H, W

    # this is the normal in current camera frame

    return n