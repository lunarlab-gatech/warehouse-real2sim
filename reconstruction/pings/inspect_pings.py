#!/usr/bin/env python3
# @file      inspect_pings.py
# @author    Yue Pan     [yue.pan@igg.uni-bonn.de]
# Copyright (c) 2025 Yue Pan, all rights reserved

import glob
import os
import sys
import time
import yaml
import csv
from datetime import datetime
from time import sleep

import dtyper as typer
from typing import Dict, List

import numpy as np
import open3d as o3d
import torch
import torch.multiprocessing as mp
from tqdm import tqdm
from rich import print
# import vdbfusion
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

import typer
from typing import Optional, Tuple

from dataset.slam_dataset import SLAMDataset, read_kitti_format_poses
from model.decoder import Decoder
from model.neural_gaussians import NeuralPoints
from utils.config import Config
from utils.mesher import Mesher, filter_isolated_vertices
from utils.tools import setup_experiment, split_chunks, load_decoders, save_video_np, remove_gpu_cache, colorize_depth_maps, slerp_pose, setup_optimizer
from utils.campose_utils import update_pose

from eval.eval_mesh_utils import eval_pair

from gaussian_splatting.gaussian_renderer import render, spawn_gaussians
from gaussian_splatting.utils.cameras import CamImage
from gaussian_splatting.utils.graphics_utils import fov2focal
from gaussian_splatting.utils.loss_utils import l1_loss, tukey_loss
from gaussian_splatting.utils.image_utils import psnr

from fused_ssim import fused_ssim

from gs_gui import slam_gui
from gs_gui.gui_utils import VisPacket, ParamsGUI, ControlPacket, get_latest_queue


'''
    Inspect the built PINGS map
'''

app = typer.Typer(add_completion=False, rich_markup_mode="rich", context_settings={"help_option_names": ["-h", "--help"]})

docstring = f"""
:pushpin: All the utilities to inspect the PINGS map

[bold green]Examples: [/bold green]

# Inspect the map built by the experiment
$ python3 inspect_pings.py ./pings_experiments/your_experiment_path

# Inspect the GS map and also the mesh 
$ python3 inspect_pings.py ./pings_experiments/your_experiment_path -m

# Inspect the GS map along the trajectory with a interval of 2 frames
$ python3 inspect_pings.py ./pings_experiments/your_experiment_path --range 0 2000 2

# Inspect the GS map along the trajectory and render the video with an interval of 5 frames
$ python3 inspect_pings.py ./pings_experiments/your_experiment_path -v --range 0 1000 5

"""

@app.command(help=docstring)
def inspect_pings_map(
    experiment_path: str = typer.Argument(..., help='Path to a certain experiment folder storing the PINGS map'),
    input_path: Optional[str] = typer.Option(None, "--input-path", "-i", help='Path to the dataset base input directory, this is mainly used for evaluation reference'),
    pose_path: Optional[str] = typer.Option(None, "--pose-path", "-p", help='Path to a certain pose txt file specified in KITTI format, the poses are used for rendering'),
    frame_range: Optional[Tuple[int, int, int]] = typer.Option(None, "--range", help='Specify the start, end and step of the frame for video rendering or evaluation, for example: --range 10 1000 1. If not specified, the whole sequence will be used. When doing evaluation, the frame range is according to the used dataset instead of the pose file'),
    center_frame_id: int = typer.Option(0, "--center-frame-id", "-f", help='set the PINGS local map to be centered at this frame id'),
    visualize: bool = typer.Option(True, '--visualize/--no-visualize', help='Turn on the PINGS visualizer GUI (default: on)'),
    log_on: bool = typer.Option(False, "--log-on", "-l", help='Turn on the logs printing'),
    eval_seq: bool = typer.Option(False, "--eval-seq", "-e", help='Do the evaluation on the input sequence'),
    use_free_view_camera: bool = typer.Option(False, "--use-free-view-camera", "-c", help='Render video with the recorded free view camera, input via -p'),
    render_video: bool = typer.Option(False, "--render-video", "-v", help='Render and save video with pre-defined trajectory in the PINGS map'),
    recon_3d: bool = typer.Option(False, "--recon-3d", help='Reconstruct 3D by rendering the PINGS map'),
    show_mesh: bool = typer.Option(False, "--show-mesh", "-m", help='Show the mesh reconstructed from PINGS map'),
    show_global: bool = typer.Option(False, "--show-global", "-g", help='Show the global map instead of the local map (might cost a lot of memory and not very fast during inferencing)'),
    cam_refine_on: bool = typer.Option(False, "--cam-refine-on", "-o",help='When doing evaluation using the test views, we would also refine the camera exposure and camera poses'),
    neural_point_color_mode: int = typer.Option(0, "--neural-point-color-mode", help='0: original rgb, 1: geo feature pca, 2: photo feature pca, 3: time, 4: stability'),
    mesh_mc_m: float = typer.Option(-1, "--mesh-mc-m", help='Marching cubes resolution (in meter) for mesh reconstruction'),
    mesh_min_nn_k: int = typer.Option(-1, "--mesh-min-nn-k", help='SDF querying min neighbor neural point count for mesh reconstruction'),
    sorrounding_map_r_m: float = typer.Option(-1, "--sorrounding-map-r-m", "-r", help='Radius of the sorrounding map in meter for far-away stuff rendering'),
    tsdf_fusion_max_range_m: float = typer.Option(-1, "--tsdf-fusion-max-range-m", help='Maximum range for doing the TSDF fusion'),
    neural_point_vis_down_rate: int = typer.Option(17, "--neural-point-vis-down-rate", help='Down rate for visualizing the neural points'),
    render_down_rate: int = typer.Option(-1, "--render-down-rate", help='Down rate of image resolution for rendering'),
    normal_with_alpha: bool = typer.Option(False, "--normal-with-alpha", help='Show the normal with alpha channel'),
    show_neural_point: bool = typer.Option(False, "--show-neural-point", "-n", help='Show the neural points by default'),
    show_gs: bool = typer.Option(True, "--show-gs/--show-gs-off", help='Show the GS rendering by default'),
    held_out: bool = typer.Option(False, "--held-out", help='Evaluate only frames that were NOT GS training keyframes (replayed from the render poses + the config keyframe gates) for uncontaminated metrics'),
    fit_exposure: bool = typer.Option(False, "--fit-exposure", help='Fit a closed-form per-channel affine color correction (a*render+b -> GT) per eval view before scoring; fair stand-in for the trained per-image exposure, which checkpoints do not persist'),
):

    yaml_files = glob.glob(f"{experiment_path}/*.yaml")
    # this might not be a clever way, fix this later (FIXME)
    if len(yaml_files) > 1: # Check if there is exactly one YAML file
        sys.exit("There are multiple YAML files. Please handle accordingly.")
    elif len(yaml_files) == 0:  # If no YAML files are found
        sys.exit("No YAML files found in the specified path.")
    
    config = Config()
    config.load(yaml_files[0])

    # deterministic eval: without this the -o camera refinement varies between runs
    np.random.seed(42)
    torch.manual_seed(42)

    model_path = os.path.join(experiment_path, "model", "pin_map.pth")

    full_config_path = os.path.join(experiment_path, "meta", "config_all.yaml")
    config.model_path = model_path

    config.silence = not log_on
    config.gs_on = show_gs

    if os.path.exists(full_config_path):
        full_config_args = yaml.safe_load(open(full_config_path))
        config.pc_path = full_config_args["pc_path"]
        config.data_loader_name = full_config_args["data_loader_name"]
        config.data_loader_seq = full_config_args["data_loader_seq"]
        config.run_name = full_config_args["run_name"]
        # print("Please make sure the default values of configurations are not changed for the run")
        config.displacement_range_ratio = full_config_args["displacement_range_ratio"]
        config.max_scale_ratio = full_config_args["max_scale_ratio"]
        config.unit_scale_ratio = full_config_args["unit_scale_ratio"]    

    video_folder_path = None
    if render_video:
        video_folder_path = os.path.join(experiment_path, "video")
        os.makedirs(video_folder_path, 0o755, exist_ok=True)
    
    mesh_folder_path = None
    if recon_3d:
        mesh_folder_path = os.path.join(experiment_path, "mesh")
        os.makedirs(mesh_folder_path, 0o755, exist_ok=True)

    gs_eval_output_csv_path = os.path.join(experiment_path, "gs_eval.csv")

    # this will then overwrite the experiment folder
    if input_path is not None:
        config.pc_path = input_path

    print("[bold green]Load PINGS Map[/bold green]","📌" )

    run_path = setup_experiment(config, sys.argv, debug_mode=True)
    config.use_dataloader = True
    
    # initialize the mlp decoder
    geo_feature_dim = config.feature_dim
    color_feature_dim = config.color_feature_dim

    geo_mlp = Decoder(config, geo_feature_dim, config.geo_mlp_hidden_dim, config.geo_mlp_level, 1)
    color_mlp = Decoder(config, color_feature_dim, config.color_mlp_hidden_dim, config.color_mlp_level, config.color_channel) if config.color_on else None

    dist_concat_dim = 1 if config.dist_concat_on else 0
    view_concat_dim = 3 if config.view_concat_on else 0

    gaussian_xyz_mlp = Decoder(config, geo_feature_dim, config.gs_mlp_hidden_dim, config.gs_mlp_level, 3, config.spawn_n_gaussian, 0)
    gaussian_rot_mlp = Decoder(config, geo_feature_dim, config.gs_mlp_hidden_dim, config.gs_mlp_level, 4, config.spawn_n_gaussian, 0)
    gaussian_scale_mlp = Decoder(config, geo_feature_dim, config.gs_mlp_hidden_dim, config.gs_mlp_level, 3, config.spawn_n_gaussian, 0)
    gaussian_alpha_mlp = Decoder(config, geo_feature_dim, config.gs_mlp_hidden_dim, config.gs_mlp_level, 1, config.spawn_n_gaussian, dist_concat_dim) # concat distance
    gaussian_color_mlp = Decoder(config, color_feature_dim, config.gs_mlp_hidden_dim, config.gs_mlp_level, 3, config.spawn_n_gaussian, view_concat_dim) # concat view direction
    
    mlp_dict = {}
    
    mlp_dict["sdf"] = geo_mlp
    mlp_dict["color"] = color_mlp
    mlp_dict["semantic"] = None

    mlp_dict["gauss_xyz"] = gaussian_xyz_mlp
    mlp_dict["gauss_scale"] = gaussian_scale_mlp
    mlp_dict["gauss_rot"] = gaussian_rot_mlp
    mlp_dict["gauss_alpha"] = gaussian_alpha_mlp
    mlp_dict["gauss_color"] = gaussian_color_mlp

    loaded_model = torch.load(model_path)
    neural_points = loaded_model["neural_points"] # neural_points config are also loaded
    neural_points.config = config
    neural_points.temporal_local_map_on = False
    neural_points.compute_feature_principle_components(down_rate=31)

    if sorrounding_map_r_m > 0: # otherwise the default value
        config.sorrounding_map_radius = sorrounding_map_r_m
        neural_points.sorrounding_map_radius = sorrounding_map_r_m

    if render_down_rate >= 0:
        config.gs_vis_down_rate = render_down_rate

    # Load the map, decoders are then freezed
    # load decoders
    load_decoders(loaded_model, mlp_dict)

    # default case, we use the pose file of the experiment run (then it would be the in-sequence views)
    if pose_path is None or (not os.path.exists(pose_path)):
        pose_path_used = os.path.join(experiment_path, "slam_poses_kitti.txt")
        if not os.path.exists(pose_path_used):
            pose_path_used = os.path.join(experiment_path, "gt_poses_kitti.txt")
    else:
        pose_path_used = pose_path
    
    render_on = (render_video or recon_3d or eval_seq)
    vis_sequence_on = render_on or (frame_range is not None)
    still_view_on = (not vis_sequence_on) or show_neural_point

    mp.set_start_method("spawn") # don't forget this
    
    # GS visualizer
    q_main2vis = q_vis2main = None
    if visualize:
        # communicator between the processes
        q_main2vis = mp.Queue() 
        q_vis2main = mp.Queue()

        params_gui = ParamsGUI(
            decoders=mlp_dict,
            background=torch.tensor(config.bg_color, dtype=config.dtype, device=config.device),
            q_main2vis=q_main2vis,
            q_vis2main=q_vis2main,
            config=config,
            gs_default_on=show_gs,
            robot_default_on=False,
            neural_point_map_default_on=show_neural_point,
            local_map_default_on=(not show_global),
            follow_cam_default_on=True,
            mesh_default_on=show_mesh,
            neural_point_color_default_mode=neural_point_color_mode, # 0: original rgb, 1: geo feature pca, 2: photo feature pca, 3: time
            neural_point_vis_down_rate=neural_point_vis_down_rate, # better to be a prime number
            frustum_size=config.vis_frame_axis_len*2.0,
            still_view_default_on=still_view_on
        )

        gui_process = mp.Process(target=slam_gui.run, args=(params_gui,))
        gui_process.start()
        time.sleep(2) # second

    # dataset
    dataset = SLAMDataset(config)

    config.pose_path = pose_path_used # but how is this used?
    poses_used = read_kitti_format_poses(pose_path_used)

    if eval_seq:
        poses_for_render = dataset.gt_poses # need to guarantee gt_poses exist
        print("Evaluate the map built by the experiment {}".format(experiment_path))
        print("The provided frame ID is according to the used dataset at {}".format(config.pc_path))
    else:
        poses_for_render = poses_used
        print("The provided frame ID is according to the pose file {}".format(pose_path_used))

    held_out_frame_ids = None
    if held_out and eval_seq:
        held_out_frame_ids = replay_trained_keyframes(
            poses_for_render,
            config.gs_keyframe_accu_travel_dist,
            config.gs_keyframe_accu_travel_degree,
            config.gs_keyframe_interval)
        print(f"Held-out eval: excluding {len(held_out_frame_ids)} replayed training keyframes "
              f"out of {len(poses_for_render)} frames")

    if frame_range is not None:
        frame_begin, frame_end, frame_step = frame_range
        if frame_end == -1:
            frame_end = len(poses_for_render)
        poses_for_render_show = poses_for_render[frame_begin:frame_end]
    else:
        frame_begin, frame_end, frame_step = 0, len(poses_for_render), 1
        poses_for_render_show = poses_for_render
        if eval_seq:
            print("Warning: the frame ID is according to the dataset, but the frame range is not specified, using the whole sequence")
    
    # reset neural points
    if eval_seq and center_frame_id == 0: # if use dataset frame ID and do not specify the center frame ID
        c_frame_id = frame_begin
    else:
        c_frame_id = center_frame_id
    frame_count = len(poses_for_render)
    c_frame_id = min(c_frame_id, frame_count-1)
    
    ref_pose = torch.tensor(poses_for_render[c_frame_id], device=config.device, dtype=config.dtype)
    ref_position = ref_pose[:3,3]

    neural_points.recreate_hash(ref_position, with_ts=False)

    # mesh reconstructor
    mesher = Mesher(config, neural_points, mlp_dict)


    cur_mesh = None
    if show_mesh:

        print("Reconstruct mesh from the SDF")

        mesh_vox_size_m = mesh_mc_m
        mesh_min_nn_k_used = mesh_min_nn_k
        if mesh_mc_m < 0:
            mesh_vox_size_m = config.voxel_size_m*0.6 # use the default value
        if mesh_min_nn_k < 0:
            mesh_min_nn_k_used = config.mesh_min_nn # use the default value
        
        neural_pcd = neural_points.get_neural_points_o3d(query_global=show_global,random_down_ratio=17)
        mesh_aabb = neural_pcd.get_axis_aligned_bounding_box()
        chunks_aabb = split_chunks(neural_pcd, mesh_aabb, mesh_vox_size_m*500) 
        print("Number of chunks for reconstruction:", len(chunks_aabb))
        print("Marching cubes resolution: {:.2f} m".format(mesh_vox_size_m))
        if log_on:
            print("mesh_min_nn:", mesh_min_nn_k_used)
        print("It may take a while to reconstruct the mesh")

        out_mesh_path = os.path.join(experiment_path, "mesh", "remesh_minnn%d_%dcm.ply" % (mesh_min_nn_k_used, round(mesh_vox_size_m*100)))
        os.makedirs(os.path.dirname(out_mesh_path), exist_ok=True)
        cur_mesh = mesher.recon_aabb_collections_mesh(chunks_aabb, mesh_vox_size_m, out_mesh_path, False, False, \
                                                    config.color_on, filter_isolated_mesh=True, mesh_min_nn=mesh_min_nn_k_used)
        print("[remesh] saved ->", out_mesh_path)

    if visualize:
        packet_to_vis: VisPacket = VisPacket(frame_id=c_frame_id, img_down_rate=config.gs_vis_down_rate)
        packet_to_vis.add_neural_points_data(neural_points, only_local_map=(not show_global))
        # if not the same, then gt_poses as in-sequence views, slam_poses as the out-of-sequence views
        packet_to_vis.add_traj(gt_poses=np.array(poses_used), slam_poses=np.array(poses_for_render_show))
        
        if cur_mesh is not None:
            packet_to_vis.add_mesh(np.array(cur_mesh.vertices, dtype=np.float64), np.array(cur_mesh.triangles), np.array(cur_mesh.vertex_colors, dtype=np.float64))
        
        q_main2vis.put(packet_to_vis)
        print("Wait for 5 seconds ...")
        time.sleep(5)
    
    cam_names = dataset.cam_names 
    if use_free_view_camera: 
        cam_names = ["free_cam"]

    # used_poses
    if vis_sequence_on:
        render_with_poses(config, 
            dataset, 
            neural_points, 
            mlp_dict,
            poses_for_render, 
            cam_names, 
            vis_global_on=show_global,
            recon_3d_on=recon_3d, 
            recon_3d_tsdf_on=False, 
            eval_on=eval_seq,
            render_on=render_on,
            mesh_to_vis=cur_mesh,
            video_save_base_path=video_folder_path, 
            tsdf_mesh_save_base_path=mesh_folder_path,
            gs_eval_output_csv_path=gs_eval_output_csv_path,
            eval_down_rate=config.gs_vis_down_rate,
            tsdf_fusion_max_range=tsdf_fusion_max_range_m,
            update_neural_points_vis=show_gs,
            vis_on=visualize,
            eval_cam_refine_on=cam_refine_on,
            log_on=log_on,
            use_free_view_camera=use_free_view_camera,
            normal_with_alpha=normal_with_alpha,
            frame_begin=frame_begin,
            frame_end=frame_end,
            frame_step=frame_step,
            skip_frame_ids=held_out_frame_ids,
            fit_exposure=fit_exposure,
            q_main2vis=q_main2vis,
            q_vis2main=q_vis2main)


def replay_trained_keyframes(poses, accu_dist_thre, accu_degree_thre, keyframe_interval):
    """Recompute which frames the mapper adopted as GS training keyframes by replaying
    its accumulated-motion gate (utils/mapper.py keyframe check) over the trajectory.
    Checkpoints do not record the keyframe list, so this deterministic replay is the
    only offline way to separate trained views from held-out views."""
    trained = {0}  # the first frame always trains
    accu_dist = 0.0
    accu_degree = 0.0
    interval = max(int(keyframe_interval), 1)
    for i in range(1, len(poses)):
        rel = np.linalg.inv(poses[i - 1]) @ poses[i]
        accu_dist += float(np.linalg.norm(rel[:3, 3]))
        cos_ang = np.clip((np.trace(rel[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
        accu_degree += abs(float(np.degrees(np.arccos(cos_ang))))
        if (accu_dist > accu_dist_thre or accu_degree > accu_degree_thre) and i % interval == 0:
            trained.add(i)
            accu_dist = 0.0
            accu_degree = 0.0
    return trained


def fit_affine_exposure(rendered, gt):
    """Closed-form per-channel least-squares fit a*rendered+b ~= gt (the same affine
    model as PINGS' exposure correction). Used at eval because the per-image exposure
    parameters learned during training are not saved in the checkpoint."""
    out = rendered.clone()
    for c in range(rendered.shape[0]):
        r = rendered[c].reshape(-1)
        g = gt[c].reshape(-1)
        r_mean = r.mean()
        g_mean = g.mean()
        var = ((r - r_mean) ** 2).mean()
        if var < 1e-8:
            continue
        a = ((r - r_mean) * (g - g_mean)).mean() / var
        b = g_mean - a * r_mean
        out[c] = a * rendered[c] + b
    return torch.clamp(out, 0.0, 1.0)


def render_with_poses(config: Config, dataset: SLAMDataset,
                      neural_points: NeuralPoints, 
                      decoders: Dict[str, Decoder], 
                      view_poses: Dict[str, np.array], 
                      cam_list: List[str],
                      video_save_base_path: str = None,
                      tsdf_mesh_save_base_path: str = None,
                      vis_on: bool = False,
                      vis_global_on: bool = False,
                      recon_3d_on: bool = True,
                      recon_3d_tsdf_on: bool = False,
                      eval_on: bool = False,
                      render_on: bool = True,
                      update_neural_points_vis: bool = True,
                      mesh_to_vis: o3d.geometry.TriangleMesh = None,
                      gs_eval_output_csv_path: str = None,
                      eval_down_rate: int = 0, 
                      normal_in_world_frame: bool = True,
                      normal_with_alpha: bool = True,
                      tsdf_fusion_voxel_size: float = None,
                      tsdf_fusion_max_range: float = -1,
                      tsdf_fusion_space_carving_on: bool = False,
                      lpips_eval_on: bool = True,
                      pc_cd_eval_on: bool = True,
                      eval_cam_refine_on: bool = False,
                      cam_refine_max_iter: int = 50,
                      use_free_view_camera: bool = False,
                      frame_begin: int = 0,
                      frame_end: int = -1,
                      frame_step: int = 1,
                      skip_frame_ids=None,
                      fit_exposure: bool = False,
                      log_on: bool = False,
                      q_main2vis=None,
                      q_vis2main=None,
                      ):
    
    """
        Inspection of the PINGS map, conduct rendering with given poses, 
        Conduct evaluation and output rendered video if needed
    """

    # view_poses as list of np array
    view_poses_np = np.array(view_poses)
    background = torch.tensor(config.bg_color, dtype=config.dtype, device=config.device)
    bg_3d = background.view(3, 1, 1)

    save_video_on = False
    if video_save_base_path is not None:
        save_video_on = True

    if eval_on and lpips_eval_on:
        lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(config.device) 

    if recon_3d_tsdf_on:
        recon_3d_on = True

    # FIXME: 
    # if recon_3d_tsdf_on:
    #     if tsdf_fusion_voxel_size is None:
    #         tsdf_fusion_voxel_size = config.voxel_size_m*0.6 # use the default value
    #     sdf_trunc = tsdf_fusion_voxel_size * 3.0
    #     space_carving_on = tsdf_fusion_space_carving_on # False: fast, cannot deal with dynamics, True: slow, can deal with dynamics, may also remove thin objects
    #     vdb_volume = vdbfusion.VDBVolume(tsdf_fusion_voxel_size,
    #                                     sdf_trunc,
    #                                     space_carving_on)

    rendered_rgb_cam_dict = {}
    rendered_depth_cam_dict = {}
    rendered_normal_cam_dict = {}

    intrinsic_o3d_cam_dict = {}
    extrinsic_o3d_cam_dict = {}

    eval_down_scale = 2**(eval_down_rate)

    psnr_list = []
    ssim_list = []
    lpips_list = []
    depthl1_list = []
    depth_rmse_list = []
    cd_list = []
    f1_list = []
    per_cam_metrics = {}  # cam_name -> {"psnr": [...], "ssim": [...], "lpips": [...]}

    eval_depth_max = config.max_range * 0.8
    eval_depth_min = config.min_range


    # free camera parameters
    free_cam_W = 640
    free_cam_H = 480
    free_cam_hfov_deg = 60
    free_cam_vfov_deg = 60

    free_cam_fx = fov2focal(np.deg2rad(free_cam_hfov_deg), free_cam_W)
    free_cam_fy = fov2focal(np.deg2rad(free_cam_vfov_deg), free_cam_H)

    free_cam_K_mat = np.eye(3)
    free_cam_K_mat[0,0] = free_cam_fx
    free_cam_K_mat[1,1] = free_cam_fy
    free_cam_K_mat[0,2] = free_cam_W // 2
    free_cam_K_mat[1,2] = free_cam_H // 2

    for cur_cam_name in cam_list: 
        # initialize lists for video
        rendered_rgb_cam_dict[cur_cam_name] = []
        rendered_depth_cam_dict[cur_cam_name] = []
        rendered_normal_cam_dict[cur_cam_name] = []

        # initialize o3d intrinsics and extrinsics

        cur_intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic()

        if use_free_view_camera:

            cur_K_mat = free_cam_K_mat
            cur_H = free_cam_H
            cur_W = free_cam_W
            cur_extrinsic = np.eye(4)
        
        else:
            cur_K_mat = dataset.K_mats[cur_cam_name]
            cur_H = dataset.cam_heights[cur_cam_name]
            cur_W = dataset.cam_widths[cur_cam_name]
            cur_extrinsic = dataset.T_c_l_mats[cur_cam_name]

        # this is for eval_down_rate = 0, if not 1, then you need to change K_mat accordingly
        cur_intrinsic_o3d.set_intrinsics(
                                    height=int(cur_H/eval_down_scale),
                                    width=int(cur_W/eval_down_scale),
                                    fx=cur_K_mat[0,0]/eval_down_scale,
                                    fy=cur_K_mat[1,1]/eval_down_scale,
                                    cx=cur_K_mat[0,2]/eval_down_scale,
                                    cy=cur_K_mat[1,2]/eval_down_scale)
        
        intrinsic_o3d_cam_dict[cur_cam_name] = cur_intrinsic_o3d
        extrinsic_o3d_cam_dict[cur_cam_name] = cur_extrinsic


    frame_count = len(view_poses)
    frame_end = min(frame_count, frame_end)

    for frame_id in tqdm(range(frame_begin, frame_end, frame_step), desc="Render views along the trajectory"):

        if skip_frame_ids is not None and frame_id in skip_frame_ids:
            continue  # held-out eval: this frame was a training keyframe

        dataset.read_frame_with_loader(frame_id, init_pose = False, use_image=True, monodepth_on=config.monodepth_on)

        if dataset.cur_cam_img is None:
            continue

        remove_gpu_cache()

        T_w_l_np = view_poses[frame_id] # for the usual case, we use LiDAR pose and a preset camera-LiDAR calibration
        T_w_l = torch.tensor(T_w_l_np, dtype=config.dtype, device=config.device)
        
        cur_frame_position_np = T_w_l_np[:3,3]
        cur_frame_position_torch = T_w_l[:3,3]

        if frame_id % 100 == 0:
            neural_points.recreate_hash(cur_frame_position_torch, kept_points=True, with_ts=False) # and at the same time reset local map
        else:
            neural_points.reset_local_map(cur_frame_position_torch, cur_ts=frame_id)

        neural_points_data, sorrounding_neural_points_data = neural_points.gather_local_data()
    
        sorrounding_spawn_results = spawn_gaussians(sorrounding_neural_points_data, 
                    decoders, None, cur_frame_position_torch,
                    dist_concat_on=config.dist_concat_on, 
                    view_concat_on=config.view_concat_on, 
                    scale_filter_on=True,
                    z_far=config.sorrounding_map_radius,
                    learn_color_residual=config.learn_color_residual,
                    gs_type=config.gs_type,
                    displacement_range_ratio=config.displacement_range_ratio,
                    max_scale_ratio=config.max_scale_ratio,
                    unit_scale_ratio=config.unit_scale_ratio)

        cur_frame_measured_pcd_o3d = None

        tran_in_frame = None

        if eval_on:
            dataset.filter_and_correct()

            # deskew and reset depth map
            if config.deskew and frame_id > 0:
                tran_in_frame = dataset.get_tran_in_frame(frame_id)
                dataset.deskew_at_frame(tran_in_frame)

            if not dataset.is_rgbd:
                dataset.project_pointcloud_to_cams(use_only_colorized_points=True, tran_in_frame=tran_in_frame) 
    
            if pc_cd_eval_on:
                cur_frame_measured_pcd_o3d = o3d.geometry.PointCloud()

                cur_frame_measured_xyz_np = (
                    dataset.cur_point_cloud_torch[:,:3].detach().cpu().numpy().astype(np.float64)
                )

                cur_frame_measured_color_np = (
                    dataset.cur_point_cloud_torch[:,3:].detach().cpu().numpy().astype(np.float64)
                )
                cur_frame_measured_pcd_o3d.points = o3d.utility.Vector3dVector(cur_frame_measured_xyz_np)
                cur_frame_measured_pcd_o3d.colors = o3d.utility.Vector3dVector(cur_frame_measured_color_np)

                cur_frame_measured_pcd_o3d.transform(T_w_l_np)

        cur_frame_rendered_pcd_o3d = None
        if recon_3d_on:
            cur_frame_rendered_pcd_o3d = o3d.geometry.PointCloud()

        rendered_rgb_list = []
        rendered_depth_list = []
        rendered_normal_list = []
            
        for cur_cam_name in cam_list: 
            
            gt_rgb_image = None
            gt_depth_image = None

            if use_free_view_camera and (not eval_on): # in this case, we do not do evaluation
                K_mat = free_cam_K_mat #  # as np.array
                T_w_c = T_w_l # the input is then directly the camera poses
                
                cur_view_cam = CamImage(frame_id, None, K_mat, 
                                        config.min_range*0.5, config.local_map_radius*1.1,
                                        cur_cam_name, device=config.device, cam_pose = T_w_c, 
                                        img_width = free_cam_W, img_height = free_cam_H)

            else:
                
                if cur_cam_name not in list(dataset.cur_cam_img.keys()):
                    continue

                K_mat = dataset.K_mats[cur_cam_name]
                T_c_l = torch.tensor(dataset.T_c_l_mats[cur_cam_name], dtype=config.dtype, device=config.device) 
                
                diff_pose_l_c_ts = torch.eye(4).to(T_w_l)
                if tran_in_frame is not None and dataset.cur_sensor_ts is not None:
                    cur_cam_ref_ts_ratio = dataset.get_cur_cam_ref_ts_ratio(cur_cam_name)
                    # print(cur_cam_ref_ts_ratio)
                    diff_pose_l_c_ts = slerp_pose(tran_in_frame, cur_cam_ref_ts_ratio, config.deskew_ref_ratio).to(T_w_l)

                T_w_l_cam_ts = T_w_l @ diff_pose_l_c_ts
                T_w_c = T_w_l_cam_ts @ torch.linalg.inv(T_c_l) # need to convert to cam frame

                # you need to also load the camera exposure coefficients here
                cur_view_cam: CamImage = dataset.cur_cam_img[cur_cam_name]
                cur_view_cam.set_pose(T_w_c)
        
                gt_rgb_image = cur_view_cam.rgb_image_list[eval_down_rate]

                if cur_view_cam.depth_on:
                    gt_depth_image = cur_view_cam.depth_image_list[eval_down_rate]
                    valid_depth_mask = (gt_depth_image > eval_depth_min) & (gt_depth_image < eval_depth_max)
            
            # here we would also refine the camera pose
            if eval_on and eval_cam_refine_on and gt_rgb_image is not None:

                opt = setup_optimizer(config, cams = [cur_view_cam])
                
                # consider increase gs_cam_refine_iter_count here
                for iter in tqdm(range(cam_refine_max_iter), disable=(not log_on), desc="Camera refinement"):    
                    
                    # current values
                    render_pkg = render(cur_view_cam, None, neural_points_data, 
                        decoders, sorrounding_spawn_results, background, 
                        down_rate=eval_down_rate, 
                        dist_concat_on=config.dist_concat_on, 
                        view_concat_on=config.view_concat_on, 
                        correct_exposure=config.exposure_correction_on, 
                        correct_exposure_affine = config.affine_exposure_correction,
                        learn_color_residual=config.learn_color_residual,
                        front_only_on=config.train_front_only,
                        gs_type=config.gs_type,
                        displacement_range_ratio=config.displacement_range_ratio,
                        max_scale_ratio=config.max_scale_ratio,
                        unit_scale_ratio=config.unit_scale_ratio,
                        verbose=False)
                    
                    # rendered results
                    rendered_rgb_image, rendered_depth, rendered_alpha = render_pkg["render"], render_pkg["surf_depth"], render_pkg["rend_alpha"] # 3, H, W / 1, H, W

                    rendered_rgb_image = torch.clamp(rendered_rgb_image, 0, 1)

                    loss_rgb_robust = tukey_loss(rendered_rgb_image, gt_rgb_image, c=0.5) # now just l1 loss

                    if config.lambda_ssim > 0.0:
                        ssim_value = fused_ssim(rendered_rgb_image.unsqueeze(0), gt_rgb_image.unsqueeze(0)) # have to be 4 dim
                        rgb_loss = (1.0 - config.lambda_ssim) * loss_rgb_robust + config.lambda_ssim * (1.0 - ssim_value)
                    else:
                        rgb_loss = loss_rgb_robust # l1 only, ssim might take a long time

                    depth_loss = 0.0 
                    if rendered_depth is not None and gt_depth_image is not None and config.lambda_depth > 0:
                        if rendered_alpha is not None:
                            accu_alpha_mask = rendered_alpha.detach() > config.depth_min_accu_alpha
                            valid_depth_mask = valid_depth_mask & accu_alpha_mask
                        depth_loss = l1_loss(gt_depth_image[valid_depth_mask], rendered_depth[valid_depth_mask])
                        depth_loss *= config.lambda_depth

                    total_loss = rgb_loss + depth_loss

                    # print("Camera refinement loss:", total_loss.item())

                    total_loss.backward()
                    
                    with torch.no_grad():
                        opt.step()
                        converged = update_pose(cur_view_cam)

                    opt.zero_grad(set_to_none=True) 

                    # if vis_on:
                    #     sleep(0.1)
                    #     if q_main2vis is not None:
                    #         # add the eval frame to vis
                    #         packet_to_vis= VisPacket(frame_id=frame_id,
                    #             current_frames=dataset.cur_cam_img, 
                    #             img_down_rate=eval_down_rate)

                    #         packet_to_vis.add_neural_points_data(neural_points, only_local_map=True)

                    #         q_main2vis.put(packet_to_vis)

                    if converged:
                        break     

            # current values
            if render_on:
                render_pkg = render(cur_view_cam, None, neural_points_data, 
                    decoders, sorrounding_spawn_results, background, 
                    down_rate=eval_down_rate, 
                    dist_concat_on=config.dist_concat_on, 
                    view_concat_on=config.view_concat_on, 
                    correct_exposure=config.exposure_correction_on, 
                    correct_exposure_affine = config.affine_exposure_correction,
                    learn_color_residual=config.learn_color_residual,
                    front_only_on=config.train_front_only,
                    gs_type=config.gs_type,
                    displacement_range_ratio=config.displacement_range_ratio,
                    max_scale_ratio=config.max_scale_ratio,
                    unit_scale_ratio=config.unit_scale_ratio,
                    verbose=False)
            else:
                render_pkg = None

            
            # rendered results
            if render_pkg is not None:
                rendered_rgb_image, rendered_depth, rendered_normal, rendered_alpha = render_pkg["render"], render_pkg["surf_depth"], render_pkg["rend_normal"], render_pkg["rend_alpha"] # 3, H, W / 1, H, W
                
                # rgb 
                rendered_rgb_image = torch.clamp(rendered_rgb_image, 0, 1)
                rendered_rgb_np = (rendered_rgb_image * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy().astype(np.uint8)  # value 0-255
                if save_video_on:
                    rendered_rgb_cam_dict[cur_cam_name].append(rendered_rgb_np)

                alpha_mask = None
                if rendered_alpha is not None:
                    alpha_mask = rendered_alpha > config.eval_depth_min_accu_alpha

                # # depth
                # if rendered_depth is not None:
                #     if alpha_mask is not None:
                #         rendered_depth[~alpha_mask] = 0.0
                        
                #     color_map_used = "inferno_r"
                #     rendered_depth_np = rendered_depth.detach().cpu().numpy().astype(np.float32) 
                #     rendered_depth_color_np = (colorize_depth_maps(rendered_depth_np, 0.1, config.max_range, cmap=color_map_used)[0]*255.0).astype(np.uint8) # 1, 3, H, W 
                #     rendered_depth_color_np = np.ascontiguousarray(np.transpose(rendered_depth_color_np, (1, 2, 0))) # H, W, 3
                #     if save_video_on:
                #         rendered_depth_cam_dict[cur_cam_name].append(rendered_depth_color_np)
                
                if rendered_normal is not None:
                    if normal_in_world_frame: 
                        rendered_normal = -1.0 * (rendered_normal.permute(1,2,0) @ (cur_view_cam.world_view_transform[:3,:3].T)).permute(2,0,1)

                    if normal_with_alpha:
                        normal_norm = rendered_normal.norm(2, dim=0) 
                        rendered_normal_show = 0.5 * (normal_norm - rendered_normal) #   # convert to the normal vis color
                    else:
                        rendered_normal = torch.nn.functional.normalize(rendered_normal, dim=0) # normalize to norm==1 # don't do this, for small opacity region, we just downweight its normal
                        rendered_normal_show = 0.5 * (1 - rendered_normal)
                    
                    rendered_normal_np = (rendered_normal_show.permute(1,2,0).detach().cpu().numpy() * 255.0).astype(np.uint8) 
                    rendered_normal_np = np.ascontiguousarray(rendered_normal_np)
                    if save_video_on:
                        rendered_normal_cam_dict[cur_cam_name].append(rendered_normal_np)

                if rendered_depth is not None and recon_3d_on:
                    
                    if tsdf_fusion_max_range > 0:
                        depth_trunc = tsdf_fusion_max_range     
                    else:
                        depth_trunc = config.max_range * 0.8 # default value

                    rgb_img_o3d = o3d.geometry.Image(rendered_rgb_np)

                    rendered_depth_np = np.transpose(rendered_depth_np, (1, 2, 0))

                    depth_img_o3d = o3d.geometry.Image(rendered_depth_np)

                    cur_rgbd_o3d = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_img_o3d, 
                                                                            depth_img_o3d, 
                                                                            depth_scale=1.0, 
                                                                            depth_trunc=depth_trunc, 
                                                                            convert_rgb_to_intensity=False)

                    cur_cam_pcd_o3d = o3d.geometry.PointCloud.create_from_rgbd_image(
                                cur_rgbd_o3d, 
                                intrinsic_o3d_cam_dict[cur_cam_name], 
                                extrinsic_o3d_cam_dict[cur_cam_name])

                    cur_frame_rendered_pcd_o3d += cur_cam_pcd_o3d 

                if eval_on:
                    # only for ipb car dataset to mask out the ego car
                    pixel_v_min = 0
                    pixel_v_max = -1
                    if dataset.cam_valid_v_ratios_minmax is not None:
                        img_height = rendered_rgb_image.shape[1]
                        valid_v_ratio = dataset.cam_valid_v_ratios_minmax[cur_cam_name]
                        pixel_v_min = int((1-valid_v_ratio[1])*img_height)
                        pixel_v_max = int((1-valid_v_ratio[0])*img_height)

                    eval_rgb_image = fit_affine_exposure(rendered_rgb_image, gt_rgb_image) if fit_exposure else rendered_rgb_image
                    rendered_rgb_image_for_eval = eval_rgb_image[:,pixel_v_min:pixel_v_max,:]
                    gt_rgb_image_for_eval = gt_rgb_image[:,pixel_v_min:pixel_v_max,:]

                    cur_psnr = psnr(rendered_rgb_image_for_eval, gt_rgb_image_for_eval).mean().item()
                    cur_ssim = fused_ssim(rendered_rgb_image_for_eval.unsqueeze(0), gt_rgb_image_for_eval.unsqueeze(0), train=False).item()

                    if lpips_eval_on:
                        cur_lpips = lpips(rendered_rgb_image_for_eval.unsqueeze(0), gt_rgb_image_for_eval.unsqueeze(0)).item()
                    else:
                        cur_lpips = -1.0 # not available

                    psnr_list.append(cur_psnr)
                    ssim_list.append(cur_ssim)
                    lpips_list.append(cur_lpips)

                    cam_metrics = per_cam_metrics.setdefault(cur_cam_name, {"psnr": [], "ssim": [], "lpips": []})
                    cam_metrics["psnr"].append(cur_psnr)
                    cam_metrics["ssim"].append(cur_ssim)
                    cam_metrics["lpips"].append(cur_lpips)
                    
                    if log_on:
                        print("Camera id: {}".format(cur_view_cam.uid))
                        print("Current view PSNR  ↑ :", f"{cur_psnr:.3f}")
                        print("Current view SSIM  ↑ :", f"{cur_ssim:.3f}")
                        print("Current view LPIPS ↓ :", f"{cur_lpips:.3f}")
                
                    if gt_depth_image is not None and rendered_depth is not None: 
                        gt_depth_image = cur_view_cam.depth_image_list[eval_down_rate] # torch.tensor
                        valid_depth_mask = (gt_depth_image > eval_depth_min) & (rendered_depth > eval_depth_min) & (gt_depth_image < eval_depth_max) & (rendered_depth < eval_depth_max)
                        
                        diff_depth = torch.abs(gt_depth_image - rendered_depth) # already abs
                        # diff_depth[~depth_valid_mask] = 0.0
                        diff_depth_masked = diff_depth[valid_depth_mask].detach().cpu().numpy()
                        cur_depth_l1 = np.mean(diff_depth_masked)
                        cur_depth_rmse = np.sqrt(np.mean(diff_depth_masked**2))
                        if log_on:
                            print("Current view Depth L1 (m) ↓ :", f"{cur_depth_l1:.3f}")
                            print("Current view Depth RMSE (m) ↓ :", f"{cur_depth_rmse:.3f}")

                        depthl1_list.append(cur_depth_l1)
                        depth_rmse_list.append(cur_depth_rmse)

            if recon_3d_on:
                
                cur_frame_rendered_pcd_o3d.transform(T_w_l_np) # convert to world frame

                # downsample a bit
                cur_frame_rendered_pcd_o3d = cur_frame_rendered_pcd_o3d.voxel_down_sample(config.vox_down_m)

                if pc_cd_eval_on: 
                    cd_metrics = eval_pair(cur_frame_rendered_pcd_o3d, cur_frame_measured_pcd_o3d, 
                        down_sample_res=0.05, threshold=0.1, 
                        truncation_acc=1.0, truncation_com=1.0)
                    
                    cur_cd = cd_metrics['Chamfer_L1 (m)']
                    cur_f1 = cd_metrics['F-score (%)']

                    if log_on:
                        print("Current eval frame Chamfer Distance L1 (m) ↓ :", f"{cur_cd:.3f}")
                        print("Current eval frame F1-score (%) ↑ :", f"{cur_f1:.3f}")

                    cd_list.append(cur_cd)
                    f1_list.append(cur_f1)

                if recon_3d_tsdf_on: 
                    # baseline: rerender depth and then do TSDF fusion
                    # better do the downsampling first, too time consuming here
                    if log_on:
                        print("Begin TSDF fusion")
                    vdb_volume.integrate(np.array(cur_frame_rendered_pcd_o3d.points), cur_frame_position_np)
                    if log_on:
                        print("TSDF fusion done")

        if vis_on:
            # add the eval frame to vis
            frames_to_vis = dataset.cur_cam_img

            packet_to_vis= VisPacket(frame_id=frame_id,
                current_frames=frames_to_vis, 
                img_down_rate=eval_down_rate)

            if update_neural_points_vis:
                packet_to_vis.add_neural_points_data(neural_points, only_local_map=(not vis_global_on))
            
            if cur_frame_measured_pcd_o3d is not None:
                packet_to_vis.add_scan(np.array(cur_frame_measured_pcd_o3d.points, dtype=np.float64), 
                                        np.array(cur_frame_measured_pcd_o3d.colors, dtype=np.float64))

            if cur_frame_rendered_pcd_o3d is not None:
                packet_to_vis.add_rendered_scan(np.array(cur_frame_rendered_pcd_o3d.points, dtype=np.float64), 
                                                np.array(cur_frame_rendered_pcd_o3d.colors, dtype=np.float64))
            
            # if mesh_to_vis is not None:
            #     packet_to_vis.add_mesh(np.array(mesh_to_vis.vertices, dtype=np.float64), 
            #                             np.array(mesh_to_vis.triangles), 
            #                             np.array(mesh_to_vis.vertex_colors, dtype=np.float64))
            
            poses_for_show = view_poses_np[frame_begin:frame_id+1]
            packet_to_vis.add_traj(slam_poses = poses_for_show)

            q_main2vis.put(packet_to_vis)

            if not q_vis2main.empty():
                control_packet: ControlPacket = get_latest_queue(q_vis2main)
                vis_global_on = control_packet.flag_global
                while control_packet.flag_pause:
                    time.sleep(0.1)
                    if not q_vis2main.empty():
                        control_packet = get_latest_queue(q_vis2main)
                        if not control_packet.flag_pause:
                            break

    if eval_on and gs_eval_output_csv_path is not None:

        # output eval results 
        
        cam_count = len(dataset.cam_names) # better to also compute for each cam

        eval_frame_count = len(psnr_list) 
        psnr_np = ssim_np = lpips_np = depthl1_np = depth_rmse_np = cd_np = f1_np = 0.0

        print(f"Calculated on {eval_frame_count} eval views")
        psnr_np = np.mean(np.array(psnr_list))
        ssim_np = np.mean(np.array(ssim_list))
        lpips_np = np.mean(np.array(lpips_list))

        print("Average eval view PSNR  ↑ :", f"{psnr_np:.3f}")
        print("Average eval view SSIM  ↑ :", f"{ssim_np:.3f}")
        print("Average eval view LPIPS ↓ :", f"{lpips_np:.3f}")

        if len(depthl1_list) > 0:
            depthl1_np = np.mean(np.array(depthl1_list))
            depth_rmse_np = np.mean(np.array(depth_rmse_list))

            print("Average eval view Depth L1 (m) ↓ :", f"{depthl1_np:.3f}")
            print("Average eval view Depth RMSE (m) ↓ :", f"{depth_rmse_np:.3f}")

        if len(cd_list) > 0:
            cd_np = np.mean(np.array(cd_list))
            f1_np = np.mean(np.array(f1_list))
            print("Average eval frame CD (m) ↓ :", f"{cd_np:.3f}")
            print("Average eval frame F1 (%) ↑ :", f"{f1_np:.3f}")

        for cam_name in sorted(per_cam_metrics.keys()):
            m = per_cam_metrics[cam_name]
            print(f"[{cam_name}] {len(m['psnr'])} views  "
                  f"PSNR {np.mean(m['psnr']):.3f}  SSIM {np.mean(m['ssim']):.3f}  "
                  f"LPIPS {np.mean(m['lpips']):.3f}")

        gs_csv_columns = [
                "Frame-Type",
                "PSNR↑",
                "SSIM↑",
                "LPIPS↓",
                "Depth-L1(m)↓",
                "Depth-RMSE(m)↓",
                "Recon-CD(m)↓",
                "Recon-F1(%)↑",
                "Frame-count",
        ]

        row_label = "eval_heldout" if skip_frame_ids is not None else "eval"
        if fit_exposure:
            row_label += "_expofit"
        gs_eval = [
            {
                gs_csv_columns[0]: row_label,
                gs_csv_columns[1]: psnr_np,
                gs_csv_columns[2]: ssim_np,
                gs_csv_columns[3]: lpips_np,
                gs_csv_columns[4]: depthl1_np,
                gs_csv_columns[5]: depth_rmse_np,
                gs_csv_columns[6]: cd_np,
                gs_csv_columns[7]: f1_np,
                gs_csv_columns[8]: eval_frame_count,
            }
        ]
        for cam_name in sorted(per_cam_metrics.keys()):
            m = per_cam_metrics[cam_name]
            gs_eval.append({
                gs_csv_columns[0]: f"{row_label}_{cam_name}",
                gs_csv_columns[1]: np.mean(m["psnr"]),
                gs_csv_columns[2]: np.mean(m["ssim"]),
                gs_csv_columns[3]: np.mean(m["lpips"]),
                gs_csv_columns[4]: 0.0,
                gs_csv_columns[5]: 0.0,
                gs_csv_columns[6]: 0.0,
                gs_csv_columns[7]: 0.0,
                gs_csv_columns[8]: len(m["psnr"]),
            })

        try:
            with open(gs_eval_output_csv_path, "a") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=gs_csv_columns)
                # writer.writeheader()
                for data in gs_eval:
                    writer.writerow(data)
        except IOError:
            print("I/O error")

        print("Write the evaluation to: ", gs_eval_output_csv_path)

    if recon_3d_tsdf_on:

        # Extract triangle mesh (numpy arrays)
        vert, tri = vdb_volume.extract_triangle_mesh()

        mesh_tsdf_fusion = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(vert),
            o3d.utility.Vector3iVector(tri),
        )

        mesh_tsdf_fusion = filter_isolated_vertices(mesh_tsdf_fusion, config.min_cluster_vertices)

        mesh_tsdf_fusion.compute_vertex_normals()

        # save the mesh
        if tsdf_mesh_save_base_path is not None:
            mesh_path = os.path.join(tsdf_mesh_save_base_path, "mesh_tsdf_fusion_{}cm.ply".format(str(round(tsdf_fusion_voxel_size*1e2))))
            o3d.io.write_triangle_mesh(mesh_path, mesh_tsdf_fusion)
            print("Save the mesh results from TSDF fusion to {}".format(mesh_path))

        # free memory
        vdb_volume = None
        mesh_tsdf_fusion = None
    
    # save rendered videos along a pose sequence
    if save_video_on:

        dt = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

        for cur_cam_name in cam_list: 
            cur_rendered_rgb_list = rendered_rgb_cam_dict[cur_cam_name]
            # cur_rendered_depth_list = rendered_depth_cam_dict[cur_cam_name]
            cur_rendered_normal_list = rendered_normal_cam_dict[cur_cam_name]

            if len(cur_rendered_rgb_list) > 0:
                cur_rgb_video_save_path = os.path.join(video_save_base_path, "rendered_rgb_{}_{}.mp4".format(cur_cam_name, dt))
                save_video_np(cur_rendered_rgb_list, cur_rgb_video_save_path)

            # if len(cur_rendered_depth_list) > 0:
            #     cur_depth_video_save_path = os.path.join(video_save_base_path, "rendered_depth_{}_{}.mp4".format(cur_cam_name, dt))
            #     save_video_np(cur_rendered_depth_list, cur_depth_video_save_path)

            if len(cur_rendered_normal_list) > 0:    
                cur_normal_video_save_path = os.path.join(video_save_base_path, "rendered_normal_{}_{}.mp4".format(cur_cam_name, dt))
                save_video_np(cur_rendered_normal_list, cur_normal_video_save_path)

            # free the lists
            rendered_rgb_cam_dict[cur_cam_name] = []
            # rendered_depth_cam_dict[cur_cam_name] = []
            rendered_normal_cam_dict[cur_cam_name] = []

    # NOTE: CPU memory might not be enough for all of these videos, maybe output it one by one

    
if __name__ == "__main__":
    app()
