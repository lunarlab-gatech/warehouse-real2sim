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

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

def l1_loss(network_output, gt, weight=1):
    return torch.abs((network_output - gt) * weight).mean()

# tukey robust kernel
def tukey_loss(network_output, gt, c=4.685):
    residuals = network_output - gt 
    abs_residuals = torch.abs(residuals)
    if c > 0:
        loss = torch.zeros_like(residuals).to(residuals)
        mask = abs_residuals <= c
        loss[mask] = (c ** 2 / 6) * (1 - (1 - (residuals[mask] / c) ** 2) ** 3)
        loss[~mask] = (c ** 2) / 6
    else: 
        loss = abs_residuals # this is just l1 loss
    return loss.mean()

# GM robust kernel
def gm_loss(network_output, gt, c=0.1):
    residuals = network_output - gt 
    abs_residuals = torch.abs(residuals)
    if c > 0:
        loss = torch.zeros_like(residuals).to(residuals)
        mask = abs_residuals <= c
        loss[mask] = (c ** 2 / 6) * (1 - (1 - (residuals[mask] / c) ** 2) ** 3)
        loss[~mask] = (c ** 2) / 6
    else: 
        loss = abs_residuals # this is just l1 loss
    return loss.mean()

def l2_loss(network_output, gt, weight=1):
    return (((network_output - gt) ** 2) * weight).mean()

def cos_loss(output, gt, thrsh=0, weight=1):
    cos = torch.sum(output * gt * weight, 0)
    return (1 - cos[cos < np.cos(thrsh)]).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def normal_consist_loss(vec1, vec2):
    cos = torch.sum(vec1 * vec2, 1)
    return 1 - cos

def smooth_loss(disp, img):
    grad_disp_x = torch.abs(disp[:,1:-1, :-2] + disp[:,1:-1,2:] - 2 * disp[:,1:-1,1:-1])
    grad_disp_y = torch.abs(disp[:,:-2, 1:-1] + disp[:,2:,1:-1] - 2 * disp[:,1:-1,1:-1])
    grad_img_x = torch.mean(torch.abs(img[:, 1:-1, :-2] - img[:, 1:-1, 2:]), 0, keepdim=True) * 0.5
    grad_img_y = torch.mean(torch.abs(img[:, :-2, 1:-1] - img[:, 2:, 1:-1]), 0, keepdim=True) * 0.5
    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)
    return grad_disp_x.mean() + grad_disp_y.mean()

def geo_loss(normals):
    # Copyright: Liren Jin
    b, _, h, w = normals.shape
    count_h = _tensor_size(normals[:, :, 1:, :])
    count_w = _tensor_size(normals[:, :, :, 1:])
    h_tv = torch.pow((normals[:, :, 1:, :] - normals[:, :, : h - 1, :]), 2).sum()
    w_tv = torch.pow((normals[:, :, :, 1:] - normals[:, :, :, : w - 1]), 2).sum()
    return 2 * (h_tv / count_h + w_tv / count_w) / b


def normal_smooth_loss(normals, depths, mask, normal_diff_sigma=0.3, depth_jump_thre_m=0.02): 
    # normal smoothness loss
    # Copyright: Liren Jin
    # Compute edge-aware weights

    normal_diff_norms = central_diff(normals.unsqueeze(0)) # square of normal difference
    depth_diff_norms = central_diff(depths.unsqueeze(0).detach())  # Shape: (b, 4, h, w) # detach depth here?

    depth_mask = (depth_diff_norms <= depth_jump_thre_m**2).float()
    weights = torch.exp(-normal_diff_norms / (2 * normal_diff_sigma**2))  # Shape: (b, 4, h, w)

    # Compute weighted normal consistency loss
    loss = torch.mean(depth_mask * weights * normal_diff_norms * mask.unsqueeze(0))
    # loss = torch.mean(normal_diff_norms)

    return loss


def central_diff(map):
    # Copyright: Liren Jin
    shift_left = map[:, :, :, :-1] - map[:, :, :, 1:]
    shift_right = map[:, :, :, 1:] - map[:, :, :, :-1]
    shift_up = map[:, :, :-1, :] - map[:, :, 1:, :]
    shift_down = map[:, :, 1:, :] - map[:, :, :-1, :]

    pad = (0, 1, 0, 0)  # Padding for left-shifted differences
    shift_left = F.pad(shift_left, pad, mode="constant", value=0)
    pad = (1, 0, 0, 0)  # Padding for right-shifted differences
    shift_right = F.pad(shift_right, pad, mode="constant", value=0)
    pad = (0, 0, 0, 1)  # Padding for up-shifted differences
    shift_up = F.pad(shift_up, pad, mode="constant", value=0)
    pad = (0, 0, 1, 0)  # Padding for down-shifted differences
    shift_down = F.pad(shift_down, pad, mode="constant", value=0)
    diffs = torch.stack(
        [shift_left, shift_right, shift_up, shift_down], dim=2
    )  # Shape: (b, 3, 4, h, w)

    # Compute the squared norm of the differences
    diff_norms = torch.sum(diffs**2, dim=1)  # Shape: (b, 4, h, w)
    return diff_norms


def normal_reg_loss(normals, masks):
    # Copyright: Liren Jin

    # normals shape: (n, 3, h, w)
    n, c, h, w = normals.shape

    # Padding to handle the borders
    normals_padded = F.pad(normals, (1, 1, 1, 1), mode="replicate")
    # Unfold the padded tensor to get 3x3 neighborhoods
    neighborhoods = normals_padded.unfold(2, 3, 1).unfold(3, 3, 1)
    # neighborhoods shape: (n, 3, h, w, 3, 3)

    # Reshape neighborhoods to get all 8 neighbors and the central pixel
    neighbors = neighborhoods.permute(0, 2, 3, 4, 5, 1).reshape(n, h, w, 3, -1)
    # neighbors shape: (n, h, w, 3, 9)

    # Separate the central pixel from the neighbors
    central_pixel = neighbors[:, :, :, :, 4]
    neighbors = torch.cat(
        [neighbors[:, :, :, :, :4], neighbors[:, :, :, :, 5:]], dim=-1
    )
    # central_pixel shape: (n, h, w, 3)
    # neighbors shape: (n, h, w, 3, 8)

    # Compute dot product
    # dot_product = torch.einsum("nhwc,nhwkc->nhwk", central_pixel, neighbors)
    dot_product = (central_pixel.unsqueeze(-1) * neighbors).sum(dim=-2)
    # dot_product shape: (n, h, w, 8)

    # Compute norms
    central_norm = torch.norm(central_pixel, dim=-1, keepdim=True)
    neighbor_norm = torch.norm(neighbors, dim=-2)
    # central_norm shape: (n, h, w, 1)
    # neighbor_norm shape: (n, h, w, 8)

    # Compute cosine similarity
    cosine_similarity = dot_product / (central_norm * neighbor_norm + 1e-8)
    loss = torch.mean(1 - cosine_similarity, dim=-1)
    return (loss * masks).mean()

# def opacity_entropy_loss(opacities):
#     op_loss = torch.exp(-((opacities - 0.5) ** 2) / 0.05).mean()
#     return op_loss

def opacity_entropy_loss(opacities):
    opacities = torch.clamp(opacities, min=1e-6, max=1-1e-6)
    entropy = -opacities * torch.log(opacities) - (1 - opacities) * torch.log(1 - opacities)
    return entropy.mean()

def sky_bce_loss(sky_mask, alpha):
    # reference: https://github.com/fudan-zvg/PVG/
    o = alpha.clamp(1e-6, 1-1e-6)
    sky = sky_mask.float()
    loss_sky = (-sky * torch.log(1 - o)).mean()
    return loss_sky

def sky_mask_loss(sky_mask, alpha):
    loss_sky = alpha[sky_mask].mean()
    return loss_sky

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

