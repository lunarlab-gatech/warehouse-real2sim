#!/usr/bin/env python3
# @file      tools.py
# @author    Yue Pan     [yue.pan@igg.uni-bonn.de]
# Copyright (c) 2024 Yue Pan, all rights reserved

import getpass
import json
import multiprocessing
import os
import random
import shutil
import subprocess
import sys
import time
import imageio
import warnings
import yaml
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import open3d as o3d
from rich import print
import roma
import torch
import torch.nn as nn
from tqdm import tqdm
import wandb
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.cm import viridis
from torch import optim
from torch.autograd import grad
from torch.optim.optimizer import Optimizer

from utils.config import Config


# setup this run
def setup_experiment(config: Config, argv=None, debug_mode: bool = False):

    os.environ["NUMEXPR_MAX_THREADS"] = str(multiprocessing.cpu_count())
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_id)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # begining timestamp

    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    warnings.filterwarnings("ignore", category=FutureWarning) 

    cuda_available = torch.cuda.is_available()
    if not cuda_available:
        print("No CUDA device available, use CPU instead")
        config.device = "cpu"
    else:
        torch.cuda.empty_cache()

    # this would make the processing slower, disabling it when you are not debugging
    # torch.autograd.set_detect_anomaly(True)

    # set the random seed for all
    seed_anything(config.seed)

    run_path = None

    if not debug_mode:

        run_name = config.name + "_" + ts  # modified to a name that is easier to index
        config.run_name = run_name

        run_path = os.path.join(config.output_root, run_name)

        access = 0o755
        os.makedirs(run_path, access, exist_ok=True)
        assert os.access(run_path, os.W_OK)
        if not config.silence:
            print(f"Start {run_path}")

        config.run_path = run_path

        mesh_path = os.path.join(run_path, "mesh")
        map_path = os.path.join(run_path, "map")
        model_path = os.path.join(run_path, "model")
        log_path = os.path.join(run_path, "log")
        meta_data_path = os.path.join(run_path, "meta")
        os.makedirs(mesh_path, access, exist_ok=True)
        os.makedirs(map_path, access, exist_ok=True)
        os.makedirs(model_path, access, exist_ok=True)
        os.makedirs(log_path, access, exist_ok=True)
        os.makedirs(meta_data_path, access, exist_ok=True)

        if config.wandb_vis_on:
            # set up wandb
            setup_wandb()
            wandb.init(
                project="PINGS", config=vars(config), dir=run_path
            )  # your own worksapce
            wandb.run.name = run_name

        # config file and reproducable shell script
        if argv is not None:
            if len(argv) > 1 and os.path.exists(argv[1]):
                config_path = argv[1]
            else:
                config_path = "config/run_pin_slam.yaml"
            # copy the config file to the result folder
            shutil.copy2(config_path, run_path)  

            git_commit_id = (
                subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
            )  # current git commit
            with open(os.path.join(run_path, "run.sh"), "w") as reproduce_shell:
                reproduce_shell.write(" ".join(["git checkout ", git_commit_id, "\n"]))
                run_str = "python3 " + " ".join(argv)
                reproduce_shell.write(run_str)

        # disable lidar deskewing when not input per frame 
        if config.step_frame > 1:
            config.deskew = False

        # write the full configs to yaml file
        config_dict = vars(config)
        config_out_path = os.path.join(meta_data_path, "config_all.yaml")
        with open(config_out_path, 'w') as file:
            yaml.dump(config_dict, file, default_flow_style=False)

    # set up dtypes, note that torch stuff cannot be write to yaml, so we set it up after write out the yaml for the whole config
    config.setup_dtype()
    torch.set_default_dtype(config.dtype)

    return run_path


def seed_anything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    o3d.utility.random.seed(seed)


# all in one
def setup_optimizer(
    config: Config,
    neural_point_geo_feat=None,
    neural_point_color_feat=None,
    mlp_sdf_param=None,
    mlp_sem_param=None,
    mlp_color_param=None,
    mlp_gs_xyz_param=None,
    mlp_gs_scale_param=None,
    mlp_gs_rot_param=None,
    mlp_gs_alpha_param=None,
    mlp_gs_color_param=None,
    cams=None,
    gs_xyz=None,
    gs_features_dc=None,
    gs_features_rest=None,
    gs_opacity=None,
    gs_scaling=None,
    gs_rotation=None,
    exposure_correction_on=True,
    cam_pose_correction_on=True
) -> Optimizer:
    
    """
        main optimizer setup function for all the stuff that are optimizable
    """

    # weight_decay is for L2 regularization
    weight_decay = config.weight_decay
    weight_decay_mlp = 0.0
    opt_setting = []

    # 0.01
    lr_mlp_sdf = config.lr_mlp_base
    lr_mlp_color = config.lr_mlp_base
    lr_mlp_sem = config.lr_mlp_base

    if mlp_sdf_param is not None:
        mlp_sdf_param_opt_dict = {
            "params": mlp_sdf_param,
            "lr": lr_mlp_sdf,
            "weight_decay": weight_decay_mlp,
            "name": "sdf_mlp_param",
        }
        opt_setting.append(mlp_sdf_param_opt_dict)
    if config.color_on and mlp_color_param is not None:
        mlp_color_param_opt_dict = {
            "params": mlp_color_param,
            "lr": lr_mlp_color,
            "weight_decay": weight_decay_mlp,
            "name": "color_mlp_param",
        }
        opt_setting.append(mlp_color_param_opt_dict)
    if config.semantic_on and mlp_sem_param is not None:
        mlp_sem_param_opt_dict = {
            "params": mlp_sem_param,
            "lr": lr_mlp_sem,
            "weight_decay": weight_decay_mlp,
            "name": "sem_mlp_param",
        }
        opt_setting.append(mlp_sem_param_opt_dict)

    if config.gs_on:
        if mlp_gs_xyz_param is not None:
            mlp_gs_xyz_param_opt_dict = {
                "params": mlp_gs_xyz_param,
                "lr": config.lr_mlp_gs_xyz,
                "weight_decay": weight_decay_mlp,
                "name": "gs_xyz_mlp_param",
            }
            opt_setting.append(mlp_gs_xyz_param_opt_dict)
        if mlp_gs_scale_param is not None:
            mlp_gs_scale_param_opt_dict = {
                "params": mlp_gs_scale_param,
                "lr": config.lr_mlp_gs_scale,
                "weight_decay": weight_decay_mlp,
                "name": "gs_scale_mlp_param",
            }
            opt_setting.append(mlp_gs_scale_param_opt_dict)
        if mlp_gs_rot_param is not None:
            mlp_gs_rot_param_opt_dict = {
                "params": mlp_gs_rot_param,
                "lr": config.lr_mlp_gs_rot,
                "weight_decay": weight_decay_mlp,
                "name": "gs_rot_mlp_param",
            }
            opt_setting.append(mlp_gs_rot_param_opt_dict)
        if mlp_gs_alpha_param is not None:
            mlp_gs_alpha_param_opt_dict = {
                "params": mlp_gs_alpha_param,
                "lr": config.lr_mlp_gs_alpha,
                "weight_decay": weight_decay_mlp,
                "name": "gs_xyz_alpha_param",
            }
            opt_setting.append(mlp_gs_alpha_param_opt_dict)
        if mlp_gs_color_param is not None:
            mlp_gs_color_param_opt_dict = {
                "params": mlp_gs_color_param,
                "lr": config.lr_mlp_gs_color,
                "weight_decay": weight_decay_mlp,
                "name": "gs_color_mlp_param",
            }
            opt_setting.append(mlp_gs_color_param_opt_dict)

        
        # original GS parameters
        if gs_xyz is not None:
            gs_xy_opt_dict = {
                "params": gs_xyz,
                "lr": config.lr_gs_position * config.max_range,
                "name": "gs_xyz",
            }
            opt_setting.append(gs_xy_opt_dict)
        if gs_features_dc is not None:
            gs_features_dc_opt_dict = {
                "params": gs_features_dc,
                "lr": config.lr_gs_features,
                "name": "gs_features_dc",
            }
            opt_setting.append(gs_features_dc_opt_dict)
        if gs_features_rest is not None:
            gs_features_rest_opt_dict = {
                "params": gs_features_rest,
                "lr": config.lr_gs_features / 20.0,
                "name": "gs_features_rest",
            }
            opt_setting.append(gs_features_rest_opt_dict)
        if gs_opacity is not None:
            gs_opacity_opt_dict = {
                "params": gs_opacity,
                "lr": config.lr_gs_opacity,
                "name": "gs_opacity",
            }
            opt_setting.append(gs_opacity_opt_dict)
        if gs_scaling is not None:
            gs_scaling_opt_dict = {
                "params": gs_scaling,
                "lr": config.lr_gs_scaling,
                "name": "gs_scaling",
            }
            opt_setting.append(gs_scaling_opt_dict)
        if gs_rotation is not None:
            gs_rotation_opt_dict = {
                "params": gs_rotation,
                "lr": config.lr_gs_rotation,
                "name": "gs_rotation",
            }
            opt_setting.append(gs_rotation_opt_dict)

    if cams is not None:
        for cam in cams:
            # exposure
            if exposure_correction_on:
                opt_setting.append(
                    {
                        "params": [cam.exposure_a],
                        "lr": config.lr_exposure,
                        "name": "cam_{}_exposure_a".format(cam.uid),
                    }
                )
                opt_setting.append(
                    {
                        "params": [cam.exposure_b],
                        "lr": config.lr_exposure,
                        "name": "cam_{}_exposure_b".format(cam.uid),
                    }
                )
                opt_setting.append(
                    {
                        "params": [cam.exposure_mat],
                        "lr": config.lr_exposure,
                        "name": "cam_{}_exposure_a".format(cam.uid),
                    }
                )
                opt_setting.append(
                    {
                        "params": [cam.exposure_offset],
                        "lr": config.lr_exposure,
                        "name": "cam_{}_exposure_b".format(cam.uid),
                    }
                )
            if cam_pose_correction_on:
                opt_setting.append(
                    {
                        "params": [cam.cam_rot_delta],
                        "lr": config.lr_cam_dr,
                        "name": "cam_{}_dr".format(cam.uid),
                    }
                )
                opt_setting.append(
                    {
                        "params": [cam.cam_trans_delta],
                        "lr": config.lr_cam_dt,
                        "name": "cam_{}_dt".format(cam.uid),
                    }
                )

    weight_decay_feature = 0.0

    if neural_point_geo_feat is not None:
        geo_feat_opt_dict = {
            "params": neural_point_geo_feat,
            "lr": config.lr_geo,
            "weight_decay": weight_decay_feature,
            "name": "neural_point_geo_features",
        }
        opt_setting.append(geo_feat_opt_dict)

    if neural_point_color_feat is not None:
        color_feat_opt_dict = {
            "params": neural_point_color_feat,
            "lr": config.lr_color,
            "weight_decay": weight_decay_feature,
            "name": "neural_point_color_features",
        }
        opt_setting.append(color_feat_opt_dict)

    if config.opt_adam:
        #opt = optim.Adam(opt_setting, betas=(0.9, 0.99), eps=config.adam_eps) # 1e-15
        opt = optim.AdamW(opt_setting, betas=(0.9, 0.99), eps=config.adam_eps) # 1e-15
    else:
        opt = optim.SGD(opt_setting, momentum=0.9)

    return opt

# set up weight and bias
def setup_wandb():
    print(
        "Weight & Bias logging option is on. Disable it by setting  wandb_vis_on: False  in the config file."
    )
    username = getpass.getuser()
    # print(username)
    wandb_key_path = username + "_wandb.key"
    if not os.path.exists(wandb_key_path):
        wandb_key = input(
            "[You need to firstly setup and login wandb] Please enter your wandb key (https://wandb.ai/authorize):"
        )
        with open(wandb_key_path, "w") as fh:
            fh.write(wandb_key)
    else:
        print("wandb key already set")
    os.system('export WANDB_API_KEY=$(cat "' + wandb_key_path + '")')


def step_lr_decay(
    optimizer: Optimizer,
    learning_rate: float,
    iteration_number: int,
    steps: List,
    reduce: float = 1.0,
):

    if reduce > 1.0 or reduce <= 0.0:
        sys.exit("The decay reta should be between 0 and 1.")

    if iteration_number in steps:
        steps.remove(iteration_number)
        learning_rate *= reduce
        print("Reduce base learning rate to {}".format(learning_rate))

        for param in optimizer.param_groups:
            param["lr"] *= reduce

    return learning_rate


# calculate the analytical gradient by pytorch auto diff
def get_gradient(inputs, outputs):
    d_points = torch.ones_like(outputs, requires_grad=False, device=outputs.device)
    points_grad = grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=d_points,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    return points_grad


def freeze_model(model: nn.Module):
    for child in model.children():
        for param in child.parameters():
            param.requires_grad = False


def unfreeze_model(model: nn.Module):
    for child in model.children():
        for param in child.parameters():
            param.requires_grad = True


def freeze_decoders(mlps, config):
    if not config.silence:
        print("Freeze the decoders")
    
    keys = list(mlps.keys())
    for key in keys:
        mlp = mlps[key]
        if mlp is not None:
            freeze_model(mlp)


def save_checkpoint(
    neural_points,
    sdf_mlp,
    color_mlp,
    sem_mlp,
    optimizer,
    run_path,
    checkpoint_name,
    iters,
):
    torch.save(
        {
            "iters": iters,
            "neural_points": neural_points,  # save the whole NN module
            "sdf_mlp": sdf_mlp.state_dict(),
            "color_mlp": color_mlp.state_dict(),
            "sem_mlp": sem_mlp.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        os.path.join(run_path, f"{checkpoint_name}.pth"),
    )
    print(f"save the model to {run_path}/{checkpoint_name}.pth")


def save_implicit_map(
    run_path, neural_points, mlp_dict, with_footprint: bool = False
):
    # together with the mlp decoders

    map_model = {"neural_points": neural_points}

    for key in list(mlp_dict.keys()):
        if mlp_dict[key] is not None:
            map_model[key] = mlp_dict[key].state_dict()
        else:
            map_model[key] = None

    model_save_path = os.path.join(run_path, "model", "pin_map.pth")  # end with .pth
    torch.save(map_model, model_save_path)

    print(f"save the map to {model_save_path}")

    if with_footprint:
        np.save(
            os.path.join(run_path, "memory_footprint.npy"),
            np.array(neural_points.memory_footprint),
        )  # save detailed memory table

# TODO: have some problem, neural points are not assigned back
def load_implicit_map(model_path, neural_points, mlp_dict, 
    neural_points_key_name = "neural_points"):
    map_model = torch.load(model_path)

    for key in list(map_model.keys()):
        if key != neural_points_key_name:
            if map_model[key] is not None:
                mlp_dict[key].load_state_dict(map_model[key])
                freeze_model(mlp_dict[key])
        else: # == neural_points_key_name
            neural_points = map_model[neural_points_key_name]
            # print(neural_points.neural_points)

    print("Implicit map loaded")


def load_decoder(config, geo_mlp, sem_mlp, color_mlp):
    loaded_model = torch.load(config.model_path)
    geo_mlp.load_state_dict(loaded_model["sdf_mlp"])
    print("Pretrained decoder loaded")
    freeze_model(geo_mlp)  # fixed the decoder
    if config.semantic_on:
        sem_mlp.load_state_dict(loaded_model["sem_mlp"])
        freeze_model(sem_mlp)  # fixed the decoder
    if config.color_on:
        color_mlp.load_state_dict(loaded_model["color_mlp"])
        freeze_model(color_mlp)  # fixed the decoder

def load_decoders(loaded_model, mlp_dict, freeze_decoders: bool = True):

    for key in list(loaded_model.keys()):
        if key != "neural_points":
            if loaded_model[key] is not None:
                mlp_dict[key].load_state_dict(loaded_model[key])
                if freeze_decoders:
                    freeze_model(mlp_dict[key])

    print("Pretrained decoders loaded")

def remove_gpu_cache():
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        torch.cuda.empty_cache()

def get_gpu_memory_usage_gb(return_cached: bool = True):
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        if return_cached:
            return torch.cuda.memory_cached() / (1024 ** 3)
        else:
            return torch.cuda.memory_allocated() / (1024 ** 3)
    else:
        return 0.0

def create_bbx_o3d(center, half_size):
    return o3d.geometry.AxisAlignedBoundingBox(center - half_size, center + half_size)

def get_time():
    """
    :return: get timing statistics
    """
    cuda_available = torch.cuda.is_available()
    if cuda_available:  # issue #10
        torch.cuda.synchronize()
    return time.time()

def track_progress():
    progress_bar = tqdm(desc="Processing", total=0, unit="calls")

    def decorator(func):
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            wrapper.calls += 1
            progress_bar.update(1)
            progress_bar.set_description("Processing point cloud frame")
            return result
        wrapper.calls = 0
        return wrapper
    return decorator

def is_prime(n):
    """Helper function to check if a number is prime."""
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True

def find_closest_prime(n):
    """Find the closest prime number to n."""
    if n < 2:
        return 2
    
    if is_prime(n):
        return n
        
    # Check numbers both above and below n
    lower = n - 1
    upper = n + 1
    
    while True:
        if is_prime(lower):
            return lower
        if is_prime(upper):
            return upper
        lower -= 1
        upper += 1

def load_from_json(filename: Path):
    """Load a dictionary from a JSON filename.
    Args:
        filename: The filename to load from.
    """
    assert filename.suffix == ".json"
    with open(filename, encoding="UTF-8") as file:
        return json.load(file)


def write_to_json(filename: Path, content: dict):
    """Write data to a JSON file.
    Args:
        filename: The filename to write to.
        content: The dictionary data to write.
    """
    assert filename.suffix == ".json"
    with open(filename, "w", encoding="UTF-8") as file:
        json.dump(content, file)


def feature_pca_torch(data, principal_components = None,
                     principal_dim: int = 3,
                     down_rate: int = 1,
                     project_data: bool = True,
                     normalize: bool = True):
    """
        do PCA to a NxD torch tensor to get the data along the K principle dimensions
        N is the data count, D is the dimension of the data

        We can also use a pre-computed principal_components for only the projection of input data
    """

    N, D = data.shape

    # Step 1: Center the data (subtract the mean of each dimension)
    data_centered = data - data.mean(dim=0)

    if principal_components is None:
        data_centered_for_compute = data_centered[::down_rate]

        assert data_centered_for_compute.shape[0] > principal_dim, "not enough data for PCA computation, down_rate might be too large or original data count is too small"

        # Step 2: Compute the covariance matrix (D x D)
        cov_matrix = torch.matmul(data_centered_for_compute.T, data_centered_for_compute) / (N - 1)

        # Step 3: Perform eigen decomposition of the covariance matrix
        eigenvalues, eigenvectors = torch.linalg.eig(cov_matrix)
        eigenvalues_r = eigenvalues.real.to(data)
        eigenvectors_r = eigenvectors.real.to(data)
        # print(eigenvalues)
        # print(eigenvectors)
        # eigenvalues = eigenvalues[:, 0]  # Only the real parts are needed

        # Step 4: Sort eigenvalues and eigenvectors in descending order
        sorted_indices = torch.argsort(eigenvalues_r, descending=True)
        principal_components = eigenvectors_r[:, sorted_indices[:principal_dim]]  # First 3 principal components

    data_pca = None
    if project_data:
        # Step 5: Project data onto the top 3 principal components
        data_pca = torch.matmul(data_centered, principal_components) # N, D @ D, P

        # normalize to show as rgb
        if normalize: 
            # min_vals = data_pca.min(dim=0, keepdim=True).values
            # max_vals = data_pca.max(dim=0, keepdim=True).values

            # # deal with outliers
            down_rate = 19
            min_vals = torch.quantile(data_pca[::down_rate], 0.02, dim=0, keepdim=True)
            max_vals = torch.quantile(data_pca[::down_rate], 0.98, dim=0, keepdim=True)

            # Normalize to range [0, 1]
            data_pca = (data_pca - min_vals) / (max_vals - min_vals)

            data_pca = data_pca.clamp(0, 1)

    return data_pca, principal_components

def color_to_intensity(colors: torch.tensor):
    intensity = 0.144 * colors[:, 0] + 0.299 * colors[:, 1] + 0.587 * colors[:, 2]
    return intensity.unsqueeze(1)


def create_axis_aligned_bounding_box(center, size):
    # Calculate the min and max coordinates based on the center and size
    min_coords = center - (size / 2)
    max_coords = center + (size / 2)

    # Create an Open3D axis-aligned bounding box
    bounding_box = o3d.geometry.OrientedBoundingBox()
    bounding_box.center = center
    bounding_box.R = np.identity(3)  # Identity rotation matrix for axis-aligned box
    # bounding_box.extent = (max_coords - min_coords) / 2
    bounding_box.extent = max_coords - min_coords

    return bounding_box


def apply_quaternion_rotation(quat: torch.tensor, points: torch.tensor) -> torch.tensor:
    # apply passive rotation: coordinate system rotation w.r.t. the points
    # p' = qpq^-1
    quat_w = quat[..., 0].unsqueeze(-1)
    quat_xyz = -quat[..., 1:]
    t = 2 * torch.linalg.cross(quat_xyz, points)
    points_t = points + quat_w * t + torch.linalg.cross(quat_xyz, t)
    return points_t


# pytorch implementations
def rotmat_to_quat(rot_matrix: torch.tensor):
    """
    Convert a batch of 3x3 rotation matrices to quaternions.
    rot_matrix: N,3,3
    return N,4
    """
    qw = (
        torch.sqrt(
            1.0 + rot_matrix[:, 0, 0] + rot_matrix[:, 1, 1] + rot_matrix[:, 2, 2]
        )
        / 2.0
    )
    qx = (rot_matrix[:, 2, 1] - rot_matrix[:, 1, 2]) / (4.0 * qw)
    qy = (rot_matrix[:, 0, 2] - rot_matrix[:, 2, 0]) / (4.0 * qw)
    qz = (rot_matrix[:, 1, 0] - rot_matrix[:, 0, 1]) / (4.0 * qw)

    # this is a unit quaternion
    quat_out = torch.stack((qw, qx, qy, qz), dim=1) 

    return quat_out


def quat_to_rotmat(quaternions: torch.tensor):
    # Ensure quaternions are normalized
    quaternions /= torch.norm(quaternions, dim=1, keepdim=True)

    # Extract quaternion components
    w, x, y, z = (
        quaternions[:, 0],
        quaternions[:, 1],
        quaternions[:, 2],
        quaternions[:, 3],
    )

    # Calculate rotation matrix elements
    w2, x2, y2, z2 = w * w, x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotation_matrix = torch.stack(
        [
            1 - 2 * (y2 + z2),
            2 * (xy - wz),
            2 * (xz + wy),
            2 * (xy + wz),
            1 - 2 * (x2 + z2),
            2 * (yz - wx),
            2 * (xz - wy),
            2 * (yz + wx),
            1 - 2 * (x2 + y2),
        ],
        dim=1,
    ).view(-1, 3, 3)

    return rotation_matrix


def quat_multiply(q1: torch.tensor, q2: torch.tensor):
    """
    Perform quaternion multiplication for batches.
    q' = q1 @ q2
    apply rotation q1 to quat q2 (q1 and q2 are both unit quaternions)
    both in the shape of N, 4
    the return q_out are also batchs of unit quaternions
    """
    w1, x1, y1, z1 = torch.unbind(q1, dim=1)  # quaternion representing the rotation
    w2, x2, y2, z2 = torch.unbind(q2, dim=1)  # quaternion to be rotated

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    q_out = torch.stack((w, x, y, z), dim=1) # N, 4

    return q_out

def quat_inverse(quat: torch.tensor):
    """
    Compute the inverse of a batch of unit quaternions.
    The input quaternions are in the shape of (N, 4), where each quaternion is (w, x, y, z).
    Returns the inverse quaternions, also in the shape of (N, 4).
    Note: For unit quaternions, the inverse is the conjugate.
    """
    # Unpack the quaternions into scalar (w) and vector (x, y, z) parts
    w, x, y, z = torch.unbind(quat, dim=1)
    
    # Compute the conjugate, which is the inverse for unit quaternions
    quat_inv = torch.stack((w, -x, -y, -z), dim=1)
    
    return quat_inv

def rotmat_to_degree_np(Rmat):
    # Ensure R is a valid rotation matrix
    # assert np.allclose(np.dot(Rmat, Rmat.T), np.eye(3))  # R * R.T should be identity
    # assert np.isclose(np.linalg.det(Rmat), 1.0)       # Determinant should be 1
    
    # Compute the rotation angle in radians using the trace of the matrix
    angle_rad = np.arccos((np.trace(Rmat) - 1) / 2.0)
    
    # Convert to degrees
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg

# def vec2quat(vec: torch.tensor):
#     v = v / v.norm()

#     # Angle of rotation (in radians)
#     theta = torch.tensor(torch.pi)  # For example, a 180-degree rotation

#     # Compute the quaternion components
#     w = torch.cos(theta / 2)
#     x = v[0] * torch.sin(theta / 2)
#     y = v[1] * torch.sin(theta / 2)
#     z = v[2] * torch.sin(theta / 2)

#     # Quaternion
#     quaternion = torch.tensor([w, x, y, z])


def torch2o3d(points_torch):
    pc_o3d = o3d.geometry.PointCloud()
    points_np = points_torch.cpu().detach().numpy().astype(np.float64)
    pc_o3d.points = o3d.utility.Vector3dVector(points_np)
    return pc_o3d


def o3d2torch(o3d, device="cpu", dtype=torch.float32):
    return torch.tensor(np.asarray(o3d.points), dtype=dtype, device=device)


def transform_torch(points: torch.tensor, transformation: torch.tensor):
    # points [N, 3]
    # transformation [4, 4]
    # Add a homogeneous coordinate to each point in the point cloud
    points_homo = torch.cat([points, torch.ones(points.shape[0], 1).to(points)], dim=1)

    # Apply the transformation by matrix multiplication
    transformed_points_homo = torch.matmul(points_homo, transformation.to(points).T)

    # Remove the homogeneous coordinate from each transformed point
    transformed_points = transformed_points_homo[:, :3]

    return transformed_points


def transform_batch_torch(points: torch.tensor, transformation: torch.tensor):
    # points [N, 3]
    # transformation [N, 4, 4]
    # N,3,3 @ N,3,1 -> N,3,1 + N,3,1 -> N,3,1 -> N,3

    # Extract rotation and translation components
    rotation = transformation[:, :3, :3].to(points)
    translation = transformation[:, :3, 3:].to(points)

    # Reshape points to match dimensions for batch matrix multiplication
    points = points.unsqueeze(-1)

    # Perform batch matrix multiplication using torch.bmm(), instead of memory hungry matmul
    transformed_points = torch.bmm(rotation, points) + translation

    # Squeeze to remove the last dimension
    transformed_points = transformed_points.squeeze(-1)

    return transformed_points


def voxel_down_sample_torch(points: torch.tensor, voxel_size: float):
    """
        voxel based downsampling. Returns the indices of the points which are closest to the voxel centers.
    Args:
        points (torch.Tensor): [N,3] point coordinates
        voxel_size (float): grid resolution

    Returns:
        indices (torch.Tensor): [M] indices of the original point cloud, downsampled point cloud would be `points[indices]`

    Reference: Louis Wiesmann
    """
    _quantization = 1000  # if change to 1, then it would take the first (smallest) index lie in the voxel

    offset = torch.floor(points.min(dim=0)[0] / voxel_size).long()
    grid = torch.floor(points / voxel_size)
    center = (grid + 0.5) * voxel_size
    dist = ((points - center) ** 2).sum(dim=1) ** 0.5
    dist = (
        dist / dist.max() * (_quantization - 1)
    ).long()  # for speed up # [0-_quantization]

    grid = grid.long() - offset
    v_size = grid.max().ceil()
    grid_idx = grid[:, 0] + grid[:, 1] * v_size + grid[:, 2] * v_size * v_size

    unique, inverse = torch.unique(grid_idx, return_inverse=True)
    idx_d = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)

    offset = 10 ** len(str(idx_d.max().item()))

    idx_d = idx_d + dist.long() * offset

    idx = torch.empty(
        unique.shape, dtype=inverse.dtype, device=inverse.device
    ).scatter_reduce_(
        dim=0, index=inverse, src=idx_d, reduce="amin", include_self=False
    )
    # https://pytorch.org/docs/stable/generated/torch.Tensor.scatter_reduce_.html
    # This operation may behave nondeterministically when given tensors on
    # a CUDA device. consider to change a more stable implementation

    idx = idx % offset
    return idx


def voxel_down_sample_min_value_torch(
    points: torch.tensor, voxel_size: float, value: torch.tensor
):
    """
        voxel based downsampling. Returns the indices of the points which has the minimum value in the voxel.
    Args:
        points (torch.Tensor): [N,3] point coordinates
        voxel_size (float): grid resolution

    Returns:
        indices (torch.Tensor): [M] indices of the original point cloud, downsampled point cloud would be `points[indices]`
    """
    _quantization = 1000

    offset = torch.floor(points.min(dim=0)[0] / voxel_size).long()
    grid = torch.floor(points / voxel_size)
    grid = grid.long() - offset
    v_size = grid.max().ceil()
    grid_idx = grid[:, 0] + grid[:, 1] * v_size + grid[:, 2] * v_size * v_size

    unique, inverse = torch.unique(grid_idx, return_inverse=True)
    idx_d = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)

    offset = 10 ** len(str(idx_d.max().item()))

    # not same value, taker the smaller value, same value, consider the smaller index
    value = (value / value.max() * (_quantization - 1)).long()  # [0-_quantization]
    idx_d = idx_d + value * offset

    idx = torch.empty(
        unique.shape, dtype=inverse.dtype, device=inverse.device
    ).scatter_reduce_(
        dim=0, index=inverse, src=idx_d, reduce="amin", include_self=False
    )
    # https://pytorch.org/docs/stable/generated/torch.Tensor.scatter_reduce_.html
    # This operation may behave nondeterministically when given tensors on
    # a CUDA device. consider to change a more stable implementation

    idx = idx % offset
    return idx


# split a large point cloud into bounding box chunks
def split_chunks(
    pc: o3d.geometry.PointCloud(),
    aabb: o3d.geometry.AxisAlignedBoundingBox() = None,
    chunk_m: float = 100.0
):

    if not pc.has_points():
        return None

    if aabb is None:
        aabb = pc.get_axis_aligned_bounding_box()
    
    chunk_aabb = []

    min_bound = aabb.get_min_bound()
    max_bound = (
        aabb.get_max_bound() + 1e-5
    )  # just to gurantee there's a zero on one side
    bbx_range = max_bound - min_bound
    if bbx_range[0] > bbx_range[1]:
        axis_split = 0
        axis_kept = 1
    else:
        axis_split = 1
        axis_kept = 0
    chunk_split = np.arange(min_bound[axis_split], max_bound[axis_split], chunk_m)

    chunk_count = 0

    for i in range(len(chunk_split)):
        cur_min_bound = np.copy(min_bound)  # you need to clone, otherwise value changed
        cur_max_bound = np.copy(max_bound)
        cur_min_bound[axis_split] = chunk_split[i]
        if i < len(chunk_split) - 1:
            cur_max_bound[axis_split] = chunk_split[i + 1]

        cur_aabb = o3d.geometry.AxisAlignedBoundingBox(cur_min_bound, cur_max_bound)
        # pc_clone = copy.deepcopy(pc)

        pc_chunk = pc.crop(cur_aabb)  # crop as clone, original one would not be changed
        cur_pc_aabb = pc_chunk.get_axis_aligned_bounding_box()

        chunk_min_bound = cur_pc_aabb.get_min_bound()
        chunk_max_bound = cur_pc_aabb.get_max_bound()
        chunk_range = chunk_max_bound - chunk_min_bound
        if chunk_range[axis_kept] > chunk_m * 3:
            chunk_split_2 = np.arange(
                chunk_min_bound[axis_kept], chunk_max_bound[axis_kept], chunk_m
            )
            for j in range(len(chunk_split_2)):
                cur_chunk_min_bound = np.copy(
                    chunk_min_bound
                )  # you need to clone, otherwise value changed
                cur_chunk_max_bound = np.copy(chunk_max_bound)
                cur_chunk_min_bound[axis_kept] = chunk_split_2[j]
                if j < len(chunk_split_2) - 1:
                    cur_chunk_max_bound[axis_kept] = chunk_split_2[j + 1]
                cur_aabb = o3d.geometry.AxisAlignedBoundingBox(
                    cur_chunk_min_bound, cur_chunk_max_bound
                )
                pc_chunk = pc.crop(
                    cur_aabb
                )  # crop as clone, original one would not be changed
                cur_pc_aabb = pc_chunk.get_axis_aligned_bounding_box()
                chunk_count += 1
                chunk_aabb.append(cur_pc_aabb)
        else:
            chunk_count += 1
            chunk_aabb.append(cur_pc_aabb)

    # print("# Chunk for meshing: ", chunk_count)
    return chunk_aabb

# FIXME: for multi-lidar version, it's not implemented in a neat way
# torch version of lidar undistortion (deskewing)
def deskewing(
    points: torch.tensor, ts: torch.tensor, 
    pose: torch.tensor, ts_ref_pose=0.5, 
    points_lidar_idx = None, T_l_lm_list = None, ts_diff_list = None,
):
    """
        LiDAR point cloud deskewing (motion compensation) function,
        note that pose indicates T_last<-cur
        ts_ref_pose =0.0, we deskew the scan to the beginning of the frame
        ts_ref_pose =0.5, we deskew the scan to the mid of the frame
        ts_ref_pose =1.0, we deskew the scan to the end of the frame

        T_l_lm_list: contain the inter-LiDAR calibrations, the transformation from the main LiDAR to the other LiDAR  
        ts_diff_list: contain the timestamp shift from the other LiDAR to the main LiDAR in ratio
    """  
    # print("Applying deskewing")
    # ts_ref_pose =  (ts_ref - ts_min) / (ts_max - ts_min)

    if ts is None:
        return points  # no deskewing

    # pose as T_last<-cur
    # ts is from 0 to 1 as the ratio
    ts = ts.squeeze(-1)

    # print(ts)

    # Normalize the tensor to the range [0, 1]
    # NOTE: you need to figure out the begin and end of a frame because
    # sometimes there's only partial measurements, some part are blocked by some occlussions
    min_ts = torch.min(ts)
    max_ts = torch.max(ts)
    ts = (ts - min_ts) / (max_ts - min_ts)

    # this is related to: https://github.com/PRBonn/kiss-icp/issues/299
    ts -= ts_ref_pose 

    points_deskewd = points

    # does not make sense
    # # for LiDARs that are not the same as the main LiDAR frame
    if T_l_lm_list is not None and points_lidar_idx is not None:
        other_lidar_count = len(T_l_lm_list)
        if other_lidar_count > 0:
            for lidar_idx in range(other_lidar_count):
                indices_lidar_idx = torch.nonzero(points_lidar_idx == (lidar_idx+1), as_tuple=True)[0]
                T_l_lm = torch.tensor(T_l_lm_list[lidar_idx]).to(pose) # transformation from main LiDAR to this LiDAR
                T_lm_l = torch.inverse(T_l_lm)
                pose_other_lidar = T_l_lm @ pose @ T_lm_l
                ts_other_lidar = ts[indices_lidar_idx]

                if ts_diff_list is not None:
                    ts_other_lidar += ts_diff_list[lidar_idx]

                # print(ts_other_lidar)
                rotmat_slerp = roma.rotmat_slerp(torch.eye(3).to(points), pose_other_lidar[:3, :3].to(points), ts_other_lidar)
                tran_lerp = ts_other_lidar[:, None] * pose_other_lidar[:3, 3].to(points)
                
                points_deskewd[indices_lidar_idx, :3] = transform_torch(points[indices_lidar_idx, :3], T_l_lm) # back to its own coordinate frame first
                points_deskewd[indices_lidar_idx, :3] = (rotmat_slerp @ points_deskewd[indices_lidar_idx, :3].unsqueeze(-1)).squeeze(-1) + tran_lerp
                points_deskewd[indices_lidar_idx, :3] = transform_torch(points_deskewd[indices_lidar_idx, :3], T_lm_l) # back to main lidar's frame

            # main LiDAR
            indices_main_lidar_idx = torch.nonzero(points_lidar_idx == 0, as_tuple=True)[0]
            ts_main_lidar = ts[indices_main_lidar_idx]
            rotmat_slerp = roma.rotmat_slerp(torch.eye(3).to(points), pose[:3, :3].to(points), ts_main_lidar)
            tran_lerp = ts_main_lidar[:, None] * pose[:3, 3].to(points)
            points_deskewd[indices_main_lidar_idx, :3] = (rotmat_slerp @ points[indices_main_lidar_idx, :3].unsqueeze(-1)).squeeze(-1) + tran_lerp

    else:
        # standard case, there's only one LiDAR (this is the standard way)
        rotmat_slerp = roma.rotmat_slerp(torch.eye(3).to(points), pose[:3, :3].to(points), ts)
        tran_lerp = ts[:, None] * pose[:3, 3].to(points)    
        points_deskewd[:, :3] = (rotmat_slerp @ points[:, :3].unsqueeze(-1)).squeeze(-1) + tran_lerp

    return points_deskewd

def slerp_pose(relative_pose: torch.tensor, ts: float, ts_ref_pose: float):
    ts -= ts_ref_pose 
    rotmat_slerp = roma.rotmat_slerp(torch.eye(3).to(relative_pose).unsqueeze(0), relative_pose[:3, :3].unsqueeze(0), torch.tensor(ts).to(relative_pose)).squeeze(0)
    tran_lerp = ts * relative_pose[:3, 3]

    # print(ts)
    # print(tran_lerp)

    T_slerp = torch.eye(4).to(relative_pose)
    T_slerp[:3,:3] = rotmat_slerp
    T_slerp[:3, 3] = tran_lerp

    return T_slerp

# for stop status check
def tranmat_close_to_identity(mats: np.ndarray, rot_thre: float, tran_thre: float):

    rot_diff = np.abs(mats[:3, :3] - np.identity(3))

    rot_close_to_identity = np.all(rot_diff < rot_thre)

    tran_diff = np.abs(mats[:3, 3])

    tran_close_to_identity = np.all(tran_diff < tran_thre)

    if rot_close_to_identity and tran_close_to_identity:
        return True
    else:
        return False

# borrow from marigold
def colorize_depth_maps(
    depth_map, min_depth, max_depth, cmap="Spectral", valid_mask=None, use_valid_depth_mask = True
):  
    """
    Colorize depth maps. computed in numpy
    """
    assert len(depth_map.shape) >= 2, "Invalid dimension"

    # use other cmap such as Spectral, inferno_r, jet

    if isinstance(depth_map, torch.Tensor):
        depth = depth_map.detach().squeeze().numpy()
    elif isinstance(depth_map, np.ndarray):
        depth = depth_map.copy().squeeze()
    # reshape to [ (B,) H, W ]
    if depth.ndim < 3:
        depth = depth[np.newaxis, :, :]

    if valid_mask is None and use_valid_depth_mask:
        valid_mask = (depth > 0)

    # colorize
    cm = matplotlib.colormaps[cmap]
    depth = ((depth - min_depth) / (max_depth - min_depth)).clip(0, 1)
    img_colored_np = cm(depth, bytes=False)[:, :, :, 0:3]  # value from 0 to 1
    img_colored_np = np.rollaxis(img_colored_np, 3, 1)

    if valid_mask is not None:
        if isinstance(depth_map, torch.Tensor):
            valid_mask = valid_mask.detach().numpy()
        valid_mask = valid_mask.squeeze()  # [H, W] or [B, H, W]
        if valid_mask.ndim < 3:
            valid_mask = valid_mask[np.newaxis, np.newaxis, :, :]
        else:
            valid_mask = valid_mask[:, np.newaxis, :, :]
        valid_mask = np.repeat(valid_mask, 3, axis=1)
        img_colored_np[~valid_mask] = 0
    
    if isinstance(depth_map, torch.Tensor):
        img_colored = torch.from_numpy(img_colored_np).float()
    elif isinstance(depth_map, np.ndarray):
        img_colored = img_colored_np

    return img_colored


def project_points_to_cam_torch(points_torch, 
                                points_rgb_torch, 
                                img_torch, 
                                T_c_l, K_mat, 
                                min_depth=1.0, 
                                max_depth=100.0):

    # points_torch and points_rgb_torch as torch.Tensor (N, 4)
    # img_torch as torch.Tensor (C, H, W), rgb channel has the float value [0,1]
    
    device = points_torch.device

    # FIXME: check if points_torch would also get changed
    # if so, clone it
    # print(points_torch)
    points_cam_torch = transform_torch(points_torch[:, :3], T_c_l)
    # print(points_torch)
    # points_torch[:, 3] = 1  # homo coordinate

    # # Transform LiDAR points to camera coordinates
    # points_cam = torch.matmul(T_c_l, points_torch.T).T  # (N, 4)
    # points_cam = points_cam[:, :3]  # (N, 3)

    # Project to image space (do it in batch)
    u, v, depth = perspective_cam2image_torch(points_cam_torch.T, K_mat) # N, 1

    _, img_height, img_width = img_torch.shape

    # Prepare depth map for visualization
    depth_map_torch = torch.zeros((1, img_height, img_width)).to(img_torch)
    count_map = torch.zeros((img_height, img_width), dtype=torch.int, device=device)
    
    mask = (u >= 0) & (u < img_width) & (v >= 0) & (v < img_height)

    mask = mask & (depth > min_depth) & (depth < max_depth)
    
    v_valid = v[mask]
    u_valid = u[mask]

    masked_depth = depth[mask]

    per_pixel_point_counter = torch.ones_like(masked_depth, dtype=torch.int, device=device)
    
    indices_1d = v_valid * img_width + u_valid # int64

    flat_depth_map = depth_map_torch.view(-1)  # Flatten depth map to match the scatter format
    flat_count_map= count_map.view(-1) 

    flat_depth_map.scatter_reduce_(0, indices_1d, masked_depth, reduce='amin', include_self=False)
    depth_map_torch = flat_depth_map.view(1, img_height, img_width)

    flat_count_map.scatter_reduce_(0, indices_1d, per_pixel_point_counter, reduce='sum')

    # TODO: now we are using amin, but better to directly not use these points for depth map
    # print("# Ambigious projection pixel:", torch.sum(flat_count_map>1).item()) # actually not much, why this would have very large impact?

    count_map = flat_count_map.view(img_height, img_width) # count of the points (rays) projected to each pixel 
    # or we just remove those ambigious ones from the depth map

    # print(depth_map_torch)

    # depth_map_torch[0, v_valid, u_valid] = masked_depth # FIXME; here might have some problem, one piexl <--> N rays (depth)

    masked_img_rgb = torch.transpose(img_torch[:, v_valid, u_valid], 0, 1) # N, 3
    mask_count = masked_depth.shape[0]
    masked_indicator = torch.zeros((mask_count, 1)).to(masked_depth) # N, 1

    # FIXME
    # unique_elements, counts = torch.unique(indices_1d, return_counts=True)
    # repeated_elements = unique_elements[counts > 1]
    # ambigious_mask = torch.isin(indices_1d, repeated_elements)
    # ambigious_indices = torch.nonzero(ambigious_mask).squeeze()
    # masked_indicator[ambigious_mask] = 1 # ambigious ones are set to invalid

    # mask indicating this point has color assigned by a corresponding pixel
    # last dimension, if 0: assigned with valid color, if 1: has not assigned with valid color
    rgb_with_mask = torch.cat((masked_img_rgb, masked_indicator), dim=-1) # N, 4 
    
    points_rgb_torch[mask] = rgb_with_mask # color [0,1]

    # points_rgb_torch[mask, :3] = img_torch[v_valid, u_valid] # color [0,1]

    # # mask indicating this point has color assigned by a corresponding pixel
    # points_rgb_torch[mask, 3] = 0  

    return points_rgb_torch, depth_map_torch

def perspective_cam2image_torch(points, K_mat):
    # project the 3D points in camera frame to 2D image frame with given intrinsic matrix

    # if there's multiple points projected to the same pixel, assign the pixel's color to the point with the smallest depth

    ndim = points.dim()
    if ndim == 2:
        points = points.unsqueeze(0)
    
    points_proj = torch.matmul(K_mat[:3, :3].reshape([1, 3, 3]), points)
    depth = points_proj[:, 2, :]
    depth[depth == 0] = -1e-6
    
    u = torch.round(points_proj[:, 0, :] / torch.abs(depth))
    v = torch.round(points_proj[:, 1, :] / torch.abs(depth))

    if ndim == 2:
        u, v, depth = u[0], v[0], depth[0]

    u = u.to(torch.int64)
    v = v.to(torch.int64)
    
    return u, v, depth

def plot_timing_detail(time_table: np.ndarray, saving_path: str, with_loop=False):

    frame_count = time_table.shape[0]
    time_table_ms = time_table * 1e3

    for i in range(time_table.shape[1] - 1):  # accumulated time
        time_table_ms[:, i + 1] += time_table_ms[:, i]

    # font1 = {'family': 'Times New Roman', 'weight' : 'normal', 'size': 16}
    # font2 = {'family': 'Times New Roman', 'weight' : 'normal', 'size': 18}
    font2 = {"weight": "normal", "size": 18}

    color_values = np.linspace(0, 1, 6)
    # Get the colors from the "viridis" colormap at the specified values
    # plasma, Pastel1, tab10
    colors = [viridis(x) for x in color_values]

    fig = plt.figure(figsize=(12.0, 4.0))

    frame_array = np.arange(frame_count)
    realtime_limit = 100.0 * np.ones([frame_count, 1])
    ax1 = fig.add_subplot(111)

    line_width_1 = 0.6
    line_width_2 = 1.0
    alpha_value = 1.0

    ax1.fill_between(
        frame_array,
        time_table_ms[:, 0],
        facecolor=colors[0],
        edgecolor="face",
        where=time_table_ms[:, 0] > 0,
        alpha=alpha_value,
        interpolate=True,
    )
    ax1.fill_between(
        frame_array,
        time_table_ms[:, 0],
        time_table_ms[:, 1],
        facecolor=colors[1],
        edgecolor="face",
        where=time_table_ms[:, 1] > time_table_ms[:, 0],
        alpha=alpha_value,
        interpolate=True,
    )
    ax1.fill_between(
        frame_array,
        time_table_ms[:, 1],
        time_table_ms[:, 2],
        facecolor=colors[2],
        edgecolor="face",
        where=time_table_ms[:, 2] > time_table_ms[:, 1],
        alpha=alpha_value,
        interpolate=True,
    )
    ax1.fill_between(
        frame_array,
        time_table_ms[:, 2],
        time_table_ms[:, 3],
        facecolor=colors[3],
        edgecolor="face",
        where=time_table_ms[:, 3] > time_table_ms[:, 2],
        alpha=alpha_value,
        interpolate=True,
    )
    if with_loop:
        ax1.fill_between(
            frame_array,
            time_table_ms[:, 3],
            time_table_ms[:, 4],
            facecolor=colors[4],
            edgecolor="face",
            where=time_table_ms[:, 4] > time_table_ms[:, 3],
            alpha=alpha_value,
            interpolate=True,
        )

    # ax1.plot(frame_array, realtime_limit, "--", linewidth=line_width_2, color="k")

    plt.tick_params(labelsize=12)
    labels = ax1.get_xticklabels() + ax1.get_yticklabels()
    # [label.set_fontname('Times New Roman') for label in labels]

    plt.xlim((0, frame_count - 1))
    plt.ylim((0, 10000)) # 10s

    plt.xlabel("Frame ID", font2)
    plt.ylabel("Runtime (ms)", font2)
    plt.tight_layout()
    # plt.title('Timing table')

    if with_loop:
        legend = plt.legend(
            (
                "Pre-processing",
                "Odometry",
                "Mapping preparation",
                "Map optimization",
                "Loop closures",
            ),
            prop=font2,
            loc=2,
        )
    else:
        legend = plt.legend(
            ("Pre-processing", "Odometry", "Mapping preparation", "Map optimization"),
            prop=font2,
            loc=2,
        )

    plt.savefig(saving_path, dpi=500)
    # plt.show()

def save_video_np(
    frames, # list of H, W, C images in np
    output_path: str,
    fps: int = 10,
    flip: bool = False,
    verbose: bool = True,
) -> None:
    if flip:
        frames_flipped = [frame[::-1] for frame in frames] # x,y axis are both up-side down
        frames = frames_flipped

    writer = imageio.get_writer(output_path, fps=fps)
    for frame in frames:
        writer.append_data(frame)
    writer.close()

    if verbose:
        print("Save the video to {} at {} Hz".format(output_path, fps))

sem_kitti_color_map = {  # rgb
    0: [255, 255, 255],
    1: [100, 150, 245],
    2: [100, 230, 245],
    3: [30, 60, 150],
    4: [80, 30, 180],
    5: [0, 0, 255],
    6: [255, 30, 30],
    7: [255, 40, 200],
    8: [150, 30, 90],
    9: [255, 0, 255],
    10: [255, 150, 255],
    11: [75, 0, 75],
    12: [175, 0, 75],
    13: [255, 200, 0],
    14: [255, 120, 50],
    15: [0, 175, 0],
    16: [135, 60, 0],
    17: [150, 240, 80],
    18: [255, 240, 150],
    19: [255, 0, 0],
    20: [30, 30, 30],
}