#!/usr/bin/env python3
# @file      config.py
# @author    Yue Pan     [yue.pan@igg.uni-bonn.de]
# Copyright (c) 2024 Yue Pan, all rights reserved

import os

import torch
import yaml


class Config:
    def __init__(self):

        # Default values (most of the parameters would be kept as default or adaptive)

        # settings
        self.name: str = "dummy"  # experiment name
        self.run_name: str = self.name # this would also include an unique timestamp

        self.run_path: str = ""
        self.output_root: str = "experiments"  # output root folder
        self.pc_path: str = ""  # input point cloud folder
        self.pose_path: str = ""  # input pose file
        self.calib_path: str = ""  # input calib file (to sensor frame), optional
        self.label_path: str = "" # input point-wise label path, for semantic mapping (optional)

        self.use_dataloader: bool = True # use specific dataloaders (now this the only option for PINGS)
        self.data_loader_name: str = "generic"
        self.data_loader_seq: str = ""

        self.load_model: bool = False  # load the pre-trained model or not
        self.model_path: str = "/"  # pre-trained model path

        self.first_frame_ref: bool = False  # if false, we directly use the world
        # frame as the reference frame
        self.begin_frame: int = 0  # begin from this frame
        self.end_frame: int = 100000  # end at this frame
        self.step_frame: int = 1  # process every x frame

        self.seed: int = 42 # random seed for the experiment
        self.num_workers: int = 12 # number of worker for the dataloader
        self.device: str = "cuda"  # use "cuda" or "cpu"
        self.gpu_id: str = "0"  # used GPU id

        # dataset specific
        self.kitti_correction_on: bool = False # intrinsic vertical angle correction # issue 11
        self.correction_deg: float = 0.0
        self.stop_frame_thre: int = 5 # FIXME # determine if the robot is stopped when there's almost no motion in a time peroid # 20

        # motion undistortion
        self.deskew: bool = False
        self.lidar_type_guess: str = "hesai" # velodyne
        self.deskew_ref_ratio: float = 0.5 # deskew to a reference ts (ratio indicates the ratio in a frame duration, typically 0.1s)

        # preprocess
        # distance filter
        self.min_range: float = 2.5 # filter too-close points (and 0 artifacts)
        self.max_range: float = 60.0 # filter far-away points
        self.range_filter_2d: bool = True # do the range-based filter according to 2d (xy) distance or 3d (xyz) distance (FIXME, for rgbd dataset, better to use 3d version) 
        self.adaptive_range_on: bool = False # use an adpative range

        self.estimate_normal: bool = False

        # filter for z coordinates (unit: m)
        self.min_z: float = -5.0  
        self.max_z: float = 60.0

        self.rand_downsample: bool = False  # apply random or voxel downsampling to input original point clcoud
        self.vox_down_m: float = 0.05 # the voxel size if using voxel downsampling (unit: m)
        self.rand_down_r: float = 1.0 # the decimation ratio if using random downsampling (0-1)

        # semantic related
        self.semantic_on: bool = False # semantic shine mapping on [semantic]
        self.sem_class_count: int = 20 # semantic class count: 20 for semantic kitti
        self.sem_label_decimation: int = 1 # use only 1/${sem_label_decimation} of the available semantic labels for training (fitting)
        self.freespace_label_on: bool = False
        self.filter_moving_object: bool = True

        # color (intensity) related 
        self.color_map_on: bool = True # colorized mapping default on
        self.color_on: bool = False
        self.color_channel: int = 0 # For RGB, channel=3, For Lidar with intensity, channel=1

        # map-based dynamic filtering (observations in certain freespace are dynamic)
        self.dynamic_filter_on: bool = False
        self.dynamic_certainty_thre: float = 1.0 # 0.5 
        self.dynamic_sdf_ratio_thre: float = 0.8 # 1.5 # type1 dynamic
        self.dynamic_min_grad_norm_thre: float = 0.25 # type2 dynamic

        # neural points
        self.temporal_local_map_off: bool = False # default on
        self.voxel_size_m: float = 0.3 # we use the voxel hashing structure to maintain the neural points, the voxel size is set as this value      
        self.weighted_first: bool = True # weighted the neighborhood feature before decoding to sdf or do the weighting of the decoded sdf afterwards
        self.layer_norm_on: bool = False # apply layer norm to the features
        self.num_nei_cells: int = 2 # the neighbor searching padding voxel # NOTE: can even be set to 3 when the motion is dramastic
        self.query_nn_k: int = 6 # query the point's k nearest neural points
        self.use_mid_ts: bool = False # use the middle of the created and last updated timestamp for adjusting or just use the created timestamp
        self.search_alpha: float = 0.2 # the larger this value is, the larger neighborhood region would be, the more robust to the highly dynamic motion and also the more time-consuming
        self.idw_index: int = 2 # the index for IDW (inverse distance weighting), 2 means square inverse
        self.buffer_size: int = int(1e8) # buffer size for hashing, the smaller, the more likely to collision # TODO decrease this to save memory # 5e7

        # shared by both kinds of feature 
        self.feature_dim: int = 8  # length of the feature for each grid feature
        self.color_feature_dim: int = 8
        self.sem_feature_dim: int = 8
        self.feature_std: float = 0.0 # feature initialization standard deviation (zero initialization)

        # Use all the surface samples or just the exact measurements to build the neural points map
        # If True may lead to larger memory consumption, but is more robust while the reconstruction.
        self.from_sample_points: bool = True
        self.from_all_samples: bool = False  # even use the freespace samples (for better ESDF mapping at a cost of larger memory consumption)
        self.map_surface_ratio: float = 0.2 # ratio * surface sample range, use those samples for initializing neural points

        # local map
        self.diff_ts_local: float = 400.0 # deprecated (use travel distance instead)
        self.local_map_travel_dist_ratio: float = 5.0
        self.local_map_radius: float = 50.0
        self.sorrounding_map_radius: float = 100.0  # radius for the sorrounding map, for rendering the background (the gaussians in it are not optimizable) 

        # map management
        self.prune_map_on: bool = False
        self.max_prune_certainty: float = 3.0 # neural point pruning threshold
        self.prune_freq_frame: int = 100

        # training sampler
        # spilt into 3 parts for sampling: close-to-surface (+ exact beam endpoint) + front-surface-freespace + behind-surface-freespace
        self.surface_sample_range_m: float = 0.25 # better to be set according to the noise level (actually as the std for a gaussian distribution)
        self.surface_sample_n: int = 3
        self.free_sample_begin_ratio: float = 0.3 # minimum ray distance ratio in front of the surface 
        self.free_sample_end_dist_m: float = 1.0 # maximum distance behind the surface (unit: m)
        self.free_front_n: int = 2
        self.free_behind_n: int = 0

        # training data pool related (for replay)
        self.window_radius: float = 50.0 # unit: m
        self.pool_capacity: int = int(1e7)
        self.bs_new_sample: int = 2048 # number of the sample per batch for the current frame's data, half of all the data
        self.new_certainty_thre: float = 1.0
        self.pool_filter_freq: int = 10 
        self.pool_filter_with_dist: bool = False # filter sdf sample pools based on a given radius # FIXME
        
        # MLP decoder
        self.mlp_bias_on: bool = True
        self.mlp_leaky_relu: bool = False
        self.geo_mlp_level: int = 1
        self.geo_mlp_hidden_dim: int = 64
        self.sem_mlp_level: int = 1
        self.sem_mlp_hidden_dim: int = 64
        self.color_mlp_level: int = 1
        self.color_mlp_hidden_dim: int = 64

        self.gs_mlp_level: int = 1
        self.gs_mlp_hidden_dim: int = 64

        self.decoder_freezed: bool = False # change to true after self.freeze_after_frame
        self.freeze_after_frame: int = 30  # if the decoder model is not loaded, it would be trained and freezed after such frame number # FIXME

        # For GS MLPs
        self.dist_concat_on: bool = False
        self.view_concat_on: bool = False

        # training (mapping) loss
        # the main loss type, select from the sample sdf loss ('bce', 'l1', 'l2', 'zhong') 
        self.main_loss_type: str = 'bce'
        self.sigma_sigmoid_m: float = 0.1 # better to be set according to the noise level (used only for BCE loss as the sigmoid scale factor)
        self.logistic_gaussian_ratio: float = 0.55 # the factor ratio for approximize a Gaussian distribution using the derivative of logistic function

        self.proj_correction_on: bool = False # conduct projective distance correction based on the sdf gradient or not, True does not work well 
        self.loss_weight_on: bool = False  # if True, the weight would be given to the loss, if False, the weight would be used to change the sigmoid's shape
        self.behind_dropoff_on: bool = False  # behind surface drop off weight
        self.dist_weight_on: bool = True  # weight decrease linearly with the measured distance, reflecting the measurement noise
        self.dist_weight_scale: float = 0.8 # weight changing range [0.6, 1.4]
        
        self.numerical_grad: bool = True # use numerical SDF gradient as in the paper Neuralangelo for the Ekional regularization during mapping
        self.gradient_decimation: int = 10 # 6 # use just a part of the points for the ekional loss when using the numerical grad, save computing time
        self.num_grad_step_ratio: float = 0.2 # step as a ratio of the nerual point resolution, length = num_grad_step_ratio * voxel_size_m

        self.ekional_loss_on: bool = True # Ekional regularization (default on)
        self.ekional_add_to: str = 'all' # select from 'all', 'surface', 'freespace', the samples used for Ekional regularization
        self.weight_e: float = 0.5 

        self.weight_s: float = 1.0  # weight for semantic classification loss
        self.weight_i: float = 0.1  # weight for color or intensity regression loss

        # optimizer
        self.mapping_freq_frame: int = 1
        self.iters: int = 12 # training iterations per frame. to have a better reconstruction results, you need to set a larger iters, a smaller lr
        self.init_iter_ratio: int = 40 # train init_iter_ratio x iters for the first frame to kick the SLAM off
        self.opt_adam: bool = True  # use adam (default) or sgd as the gradient descent optimizer
        self.bs: int = 16384 # batch size
        self.lr_geo: float = 0.01 # learning rate for the neural point geometric feature
        self.lr_color: float = 0.01 # learning rate for the neural point color feature
        self.lr_mlp_base: float = 0.01
        self.lr_exposure: float = 0.001 # learning rate for camera exposure
        # default not used
        self.lr_cam_dr: float = 0.003 # 0.003 # learning rate for camera rotation
        self.lr_cam_dt: float = 0.001 # 0.001 # learning rate for camera translation
        
        # for the mlps of the gaussian parameters (not very crucial)
        self.lr_mlp_gs_xyz = 1e-3
        self.lr_mlp_gs_scale = 1e-3
        self.lr_mlp_gs_rot = 1e-3
        self.lr_mlp_gs_alpha = 1e-3
        self.lr_mlp_gs_color = 1e-2 # better to be larger, like 1e-2

        # for directly optimizing the raw gaussian parameters # same as the original 3DGS
        # deprecated for now (we spawn Gaussians now)
        self.lr_gs_position: float = 1.6e-4
        self.lr_gs_rotation: float = 1e-3
        self.lr_gs_scaling: float = 5e-3
        self.lr_gs_opacity: float = 5e-2
        self.lr_gs_features: float = 2.5e-3
        
        self.weight_decay: float = 0.0 # weight_decay is only applied to the latent codes for the l2 regularization
        self.adam_eps: float = 1e-15
        self.adaptive_iters: bool = False # adptive map optimization iterations on (train for fewer iterations when there's not much new information to learn)
        self.new_sample_ratio_less: float = 0.02 # if smaller than this ratio, we think there's not much new information collected, train less
        self.new_sample_ratio_more: float = 0.15 # if larger than this ratio, we think there are a lot new observations to learn, train more
        self.new_sample_ratio_restart: float = 0.3 # if larger than this ratio, we think maybe tracking is lost, train much more
        
        # gaussian splatting fitting 
        self.gs_on: bool = False

        self.gs_type: str = "gaussian_surfel" # now we support 3d_gs, gaussian_surfel, and 2d_gs

        self.monodepth_on: bool = False
        self.monodepth_gaussian_res: float = self.voxel_size_m * 4.0

        self.exposure_correction_on: bool = False
        self.affine_exposure_correction: bool = True

        self.cam_pose_train_on: bool = False # jointly optimize camera pose during training

        self.gs_invalid_check_on: bool = True
        self.min_valid_nn_k: int = 6

        self.bg_color = [1.0, 1.0, 1.0] # white 
        
        self.spawn_n_gaussian = 8 # how many gaussians being spawned per neural point

        self.gs_iters: int = 0
        self.nothing_new_count_thre: int = 5 # if there is no new frame for a while (nothing_new_count_thre consecutive frames), skip the training of gsdf
        
        self.gaussian_bs_ratio: float = 1.0 # gaussian_bs = bs * gaussian_bs_ratio
        
        # gs keyframes
        self.gs_keyframe_interval: int = 1
        self.gs_keyframe_accu_travel_dist: float = 0.1 # unit: m
        self.gs_keyframe_accu_travel_degree: float = 30.0 # unit: degree

        self.lastest_train_prob: float = 0.1 # the probabilibilty of sampling a cam from the lastest observation for training
        self.short_term_train_prob: float = 0.5 # the probabilibilty of sampling a cam from short-term memory for training
        self.long_term_train_down: bool = False # downsample the training image for long-term memory, faster, vague supervision in long term memory
        
        self.img_pool_size: int = 10 # #short-term training views
        self.long_term_pool_size: int = 80
        self.gs_down_rate: int = 0 # downsampling rate for rendering (0 means no downsampling)
        self.gs_vis_down_rate: int = 0 # for the visualization

        # for gaussian spawning, the unit length is the neural point resolution
        self.displacement_range_ratio: float = 2.0 # 5.0
        self.max_scale_ratio: float = 2.0 # 5.0
        self.unit_scale_ratio: float = 0.5 # 0.5

        # disable backface rendering
        self.train_front_only: bool = True

        self.min_visible_neural_point_ratio: float = 0.1 # only train when the visible local neural point in this frame is larger than this threshold
        
        self.inverse_depth_loss: bool = False # use inverse depth (disparity) L1 loss or not
        
        # losses weights
        self.lambda_ssim: float = 0.2 # weight for ssim
        self.lambda_depth: float = 0.0 # weight for depth rendering
        self.lambda_sdf_cons: float = 0.0 # gaussian center's sdf should be close to 0 (a part of the Gaussian SDF consistency loss)
        self.lambda_sdf_normal_cons: float = 0.0 # gaussian's normal should align with sdf's gradient direction (a part of the Gaussian SDF consistency loss)
        self.lambda_area: float = 0.0 # weight for area (volume) regularization
        self.lambda_opacity: float = 0.0 # prefer larger opacity value, the smaller this value, the more likely to have masked gaussians -> fewer valid gaussian number spawned by each neural point for rendering
        self.lambda_invalid_opacity: float = 0.0 # to let those part with not well constructed sdf to have a samller opacity (deprecated)
        self.lambda_opacity_ent: float = 0.0 # for the entropy loss, opacity --> 0 or 1 (deprecated)
        self.lambda_normal_depth_consist: float = 0.0 # normal depth consistency regularization weight
        self.lambda_normal_smooth: float = 0.0 # normal smoothness loss (deprecated)
        self.lambda_mono_normal: float = 0.0 # mono normal prior loss weight (deprecated)
        self.lambda_distort: float = 0.0 # distance distortion regularization weight (1000 for bounded scene, 100 for unbounded scene), this is used to concentrate the gaussians, decrease the distance between the splat-ray intersections (deprecated)
        self.lambda_sky: float = 0.01 # bce loss, let the sky gaussians has small opacity (deprecated, use only together with monodepth)
        self.lambda_isotropic: float = 0.0 # weight for scale isotropic loss (deprecated)
        self.lambda_sdf: float = 0.0 # pin map sdf fitting loss 

        self.gs_consist_shift_count: int = 0 # shift points from spawned Gaussians to enforce the Gaussian SDF consistency loss
        self.gs_consist_shift_range_m: float = 0.5 * self.voxel_size_m # maximum shift range for the consistency loss

        # consistency loss supervision direction 
        # cannot be all true
        self.gs_consist_depth_fixed: bool = False
        self.gs_consist_normal_fixed: bool = False # fixed normal to guide depth

        self.gs_contribution_threshold: float = 0.1 # 1.0

        self.learn_color_residual: bool = False # directly learn the color or learn the residual to a base color
        
        self.valid_grad_min_thre: float = 0.4 # would be similar to reg_min_grad_norm
        self.valid_grad_max_thre: float = 2.0 # would be similar to reg_max_grad_norm
        self.min_alpha: float = 0.05
        self.depth_min_accu_alpha: float = 0.2
        self.eval_depth_min_accu_alpha: float = 0.8

        # gs evaluation
        self.gs_eval_cam_refine_on: bool = False  
        self.gs_cam_refine_iter_count: int = 50 # should be larger than 0

        # tracking (odometry estimation)
        self.track_on: bool = False
        # default without color
        self.photometric_loss_on: bool = False # add the color (or intensity) [photometric loss] to the tracking loss
        self.photometric_loss_weight: float = 0.01 # weight for the photometric loss in tracking
        self.consist_wieght_on: bool = False # weight for color (intensity) consistency for the measured and queried value
        
        self.source_vox_down_m: float = 0.8 # downsample voxel resolution for source point cloud
        self.uniform_motion_on: bool = True # use uniform motion (constant velocity) model for the transformation inital guess
        self.reg_min_grad_norm: float = 0.4 # min norm of SDF gradient for valid source point
        self.reg_max_grad_norm: float = 2.5 # max norm of SDF gradient for valid source point
        # validity gate for a registration: at least this many valid source points AND this
        # valid/source ratio. The ratio can starve honest registrations when a scan is mostly
        # NEW territory (young map -> no valid gradients yet), e.g. big open warehouses.
        self.reg_min_valid_points: int = 30
        self.reg_min_valid_ratio: float = 0.1
        self.track_mask_query_nn_k: int = self.query_nn_k # during tracking, a point without nn_k neighbors would be regarded as invalid
        self.max_sdf_ratio: float = 5.0 # ratio * surface_sample sigma
        self.max_sdf_std_ratio: float = 1.0 # ratio * surface_sample sigma
        self.reg_dist_div_grad_norm: bool = False # divide the sdf by the sdf gradient's norm for fixing overshoting or not
        self.reg_GM_dist_m: float = 0.3 # GM scale for SDF residual 
        self.reg_GM_grad: float = 0.1 # GM scale for SDF gradient anomaly 
        # for GM scale, the smaller the value, the smaller the weight would be (give even smaller weight to the outliers)
        self.reg_lm_lambda: float = 1e-4 # lm damping factor
        self.reg_iter_n: int = 50 # maximum iteration number for registration
        self.reg_term_thre_deg: float = 0.01 # iteration termination criteria for rotation 
        self.reg_term_thre_m: float = 0.001  # iteration termination criteria for translation
        self.eigenvalue_check: bool = True # use eigen value of Hessian matrix for degenaracy check

        # loop closure detection
        self.global_loop_on: bool = True # global loop detection using context descriptor
        self.local_map_context: bool = False # use local map context or scan context for loop closure description
        self.loop_with_feature: bool = False # encode neural point feature in the context
        self.min_loop_travel_dist_ratio: float = 4.0 # accumulated travel distance should be larger than this ratio * local map radius to be considered as an valid candidate
        self.local_map_context_latency: int = 5 # only used for local map context, wait for local_map_context_latency before descriptor generation for enough training of the new observations
        self.loop_local_map_time_window: int = 100 # unit: frame
        self.local_loop_dist_thre: float = 2.0 # unit: m, find local loop within this distance
        self.context_shape = [20, 60] # ring, sector count for the descriptor
        self.npmc_max_dist: float = 60.0  # max distance for the neural point map context descriptor
        self.context_num_candidates: int = 1 # select the best K candidates after comparing the ring key for further checking
        self.context_cosdist_threshold: float = 0.2 # cosine distance threshold for a candidate loop
        self.context_virtual_side_count: int = 5 # augment context_virtual_side_count virtual sensor positions on each side of the actual sensor position
        self.context_virtual_step_m: float = 2.0 # voxel_size_m * 4.0 
        self.loop_z_check_on: bool = False # check the z axix difference of the found loop frames to deal with the potential abitary issue in a multi-floor building
        self.loop_dist_drift_ratio_thre: float = 2.0 # find the loop candidate only within the distance of (loop_dist_drift_ratio_thre * drift)
    
        # pose graph optimization
        self.pgo_on: bool = False
        self.pgo_freq: int = 30 # frame interval for detecting loop closure and conducting pose graph optimization after a successful loop correction 
        self.pgo_with_isam: bool = True # use isam incremental optimization or lm batch optimization
        self.pgo_max_iter: int = 50 # maximum number of iterations
        self.pgo_with_pose_prior: bool = False # use the pose prior or not during the pgo
        self.pgo_tran_std: float = 0.04 # m 
        self.pgo_rot_std: float = 0.01 # deg
        self.use_reg_cov_mat: bool = False # use the covariance matrix directly calculated by the registration for pgo edges or not
        self.pgo_error_thre_frame: float = 500.0 # the maximum error for rejecting a wrong edge (per frame)
        self.pgo_merge_map: bool = False # merge the map (neural points) or not after the pgo (or we keep all the history neural points) 
        self.rehash_with_time: bool = True # Do the rehashing based on smaller time difference or higher point stability

        # eval
        self.wandb_vis_on: bool = False # monitor the training on weight and bias or not
        self.rerun_vis_on: bool = False # visualize the process using rerun visualizer or not
        self.silence: bool = True # print log in the terminal or not
        self.o3d_vis_on: bool = False # visualize the mesh in-the-fly using o3d visualzier or not [press space to pasue/resume]
        self.o3d_vis_raw: bool = False # visualize the raw point cloud or the weight source point cloud
        self.log_freq_frame: int = 0 # save the result log per x frames
        self.mesh_default_on: bool = False
        self.mesh_freq_frame: int = 20  # do the reconstruction per x frames
        self.sdf_default_on: bool = False # visualize the SDF slice or not
        self.sdfslice_freq_frame: int = 1 # visualize the SDF slice per x frames
        self.vis_sdf_slice_v: bool = False # also visualize the vertical SDF slice or not (default only horizontal slice)
        self.sdf_slice_height: float = -1.0 # initial height of the horizontal SDF slice (m) in sensor frame
        self.vis_sdf_res_m: float = 0.2 # resolution for the SDF slice for visualization (m)
        self.eval_traj_align: bool = True # do the SE3 alignment of the trajectory when evaluating the absolute error
        
        # mesh reconstruction, marching cubes related
        self.mc_res_m: float = 0.3 # resolution for marching cubes
        self.pad_voxel: int = 3 # pad x voxels on each side
        self.skip_top_voxel: int = 2 # slip the top x voxels (mainly for visualization indoor, remove the roof)
        self.mc_mask_on: bool = True # use mask for marching cubes to avoid the artifacts
        self.mesh_min_nn: int = 8  # The minimum number of the neighbor neural points for a valid SDF prediction for meshing, too small would cause some artifacts (more complete but less accurate), too large would lead to lots of holes (more accurate but less complete)
        self.min_cluster_vertices: int = 500 # if a connected's vertices number is smaller than this value, it would get filtered (as a postprocessing to filter outliers)
        self.keep_local_mesh: bool = False # keep the local mesh in the visualizer or not (don't delete them could cause a too large memory consumption)
        self.infer_bs: int = 4096 # batch size for inference
        
        # for baseline only
        self.tsdf_fusion_voxel_size: float = 0.2 
        self.tsdf_fusion_space_carving_on: bool = False
        self.rerender_tsdf_fusion_on: bool = False # rerender the tsdf fusion mesh or not
        
        # o3d visualization
        self.mesh_vis_normal: bool = False # normal colorization
        self.vis_frame_axis_len: float = 0.8 # sensor frame axis length, for visualization, unit: m
        self.vis_point_size: int = 2 # point size for visualization in o3d
        self.sensor_cad_path = None # the path to the sensor cad file, "./cad/ipb_car.ply" for visualization
        self.cam_cad_path = "./cad/camera.ply"

        # GS visualizer
        self.local_map_default_on: bool = True
        self.neural_point_map_default_on: bool = True
        self.gs_vis_on: bool = True # gs visualizer
        self.visualizer_split_width_ratio: float = 0.7 # left 0.6, right 0.4
        self.vis_in_cv2: bool = False # visualize rendered view in cv2 visualizer or 3d visualizer
        self.neural_point_vis_down_rate: int = 7 # downrate for the neural point visualization

        # result saving settings
        self.save_map: bool = False # save the neural point map model and decoders or not
        self.save_merged_pc: bool = False # save the merged point cloud pc or not
        self.save_mesh: bool = False # save the reconstructed mesh map or not

        # GS evaluation
        self.gs_eval_on: bool = False 
        self.rendered_pc_eval_on: bool = False

        # ROS related 
        self.run_with_ros: bool = False
        self.publish_np_map: bool = True # only for Rviz visualization, publish neural point map
        self.publish_np_map_down_rate_list = [11, 23, 37, 53, 71, 89, 97, 113, 131, 151] # prime number list, downsampling for boosting neural point map pubishing speed 
        self.republish_raw_input: bool = False # publish the raw input point cloud or not
        self.timeout_duration_s: int = 30 # in seconds, exit after receiving no topic for x seconds 

    def setup_dtype(self):
        self.dtype = torch.float32 # default torch tensor data type
        self.tran_dtype = torch.float64 # dtype used for all the transformation and poses

    def load(self, config_file):
        config_args = yaml.safe_load(open(os.path.abspath(config_file)))

        # common settings
        if "setting" in config_args:
            self.name = config_args["setting"].get("name", "pin_slam")

            self.output_root = config_args["setting"].get("output_root", "./experiments")
            self.pc_path = config_args["setting"].get("pc_path", "") 
            self.pose_path = config_args["setting"].get("pose_path", "")
            self.calib_path = config_args["setting"].get("calib_path", "")

            # optional, when semantic mapping is on [semantic]
            self.semantic_on = config_args["setting"].get("semantic_on", self.semantic_on) 
            if self.semantic_on:
                self.label_path = config_args["setting"].get("label_path", "./demo_data/labels")

            self.color_map_on = config_args["setting"].get("color_map_on", self.color_map_on)
            self.color_channel = config_args["setting"].get("color_channel", 0)
            if (self.color_channel == 1 or self.color_channel == 3) and self.color_map_on:
                self.color_on = True
            else:
                self.color_on = False

            self.load_model = config_args["setting"].get("load_model", self.load_model)
            if self.load_model:
                self.model_path = config_args["setting"].get("model_path", "")
            
            self.first_frame_ref = config_args["setting"].get("first_frame_ref", self.first_frame_ref)
            self.begin_frame = config_args["setting"].get("begin_frame", 0)
            self.end_frame = config_args["setting"].get("end_frame", self.end_frame)
            self.step_frame = config_args["setting"].get("step_frame", 1)

            self.seed = config_args["setting"].get("random_seed", self.seed)
            self.device = config_args["setting"].get("device", "cuda") # or cpu, on cpu it's about 5 times slower 
            self.gpu_id = config_args["setting"].get("gpu_id", "0")

            self.kitti_correction_on = config_args["setting"].get("kitti_correct", self.kitti_correction_on)
            if self.kitti_correction_on:
                self.correction_deg = config_args["setting"].get("correct_deg", self.correction_deg)
            self.stop_frame_thre = config_args["setting"].get("stop_frame_thre", self.stop_frame_thre)

            self.deskew = config_args["setting"].get("deskew", self.deskew) # apply motion undistortion or not
            if self.step_frame > 1:
                self.deskew = False
            self.deskew_ref_ratio = config_args["setting"].get("deskew_ref_ratio", self.deskew_ref_ratio) 

        # process
        if "process" in config_args:
            self.min_range = config_args["process"].get("min_range_m", self.min_range)
            self.max_range = config_args["process"].get("max_range_m", self.max_range)
            self.range_filter_2d = config_args["process"].get("range_filter_2d", self.range_filter_2d)
            self.min_z = config_args["process"].get("min_z_m", self.min_z)
            self.max_z = config_args["process"].get("max_z_m", self.max_z)
            self.rand_downsample = config_args["process"].get("rand_downsample", self.rand_downsample)
            if self.rand_downsample:
                self.rand_down_r = config_args["process"].get("rand_down_r", self.rand_down_r)
            else:
                self.vox_down_m = config_args["process"].get("vox_down_m", self.max_range*1e-3)
            self.dynamic_filter_on = config_args["process"].get("dynamic_filter_on", self.dynamic_filter_on)
            self.dynamic_sdf_ratio_thre = config_args["process"].get("dynamic_sdf_ratio_thre", self.dynamic_sdf_ratio_thre)
            self.dynamic_certainty_thre = config_args["process"].get("dynamic_certainty_thre", self.dynamic_certainty_thre)
            self.dynamic_min_grad_norm_thre = config_args["process"].get("dynamic_min_grad_norm_thre", self.dynamic_min_grad_norm_thre)
            self.adaptive_range_on = config_args["process"].get("adaptive_range_on", self.adaptive_range_on)
            self.estimate_normal = config_args["process"].get("estimate_normal", self.estimate_normal)

        # sampler
        if "sampler" in config_args:
            self.surface_sample_range_m = config_args["sampler"].get("surface_sample_range_m", self.vox_down_m * 3.0) 
            self.free_sample_begin_ratio = config_args["sampler"].get("free_sample_begin_ratio", self.free_sample_begin_ratio)
            self.free_sample_end_dist_m = config_args["sampler"].get("free_sample_end_dist_m", self.surface_sample_range_m * 2.0) # this value should be at least 2 times of surface_sample_range_m
            self.surface_sample_n = config_args["sampler"].get("surface_sample_n", self.surface_sample_n)
            self.free_front_n = config_args["sampler"].get("free_front_sample_n", self.free_front_n)
            self.free_behind_n = config_args["sampler"].get("free_behind_sample_n", self.free_behind_n)

        # neural point map
        if "neuralpoints" in config_args:
            self.buffer_size = int(float(config_args["neuralpoints"].get("buffer_size", self.buffer_size)))
            self.temporal_local_map_off = config_args["neuralpoints"].get("temporal_local_map_off", self.temporal_local_map_off)

            self.voxel_size_m = config_args["neuralpoints"].get("voxel_size_m", self.vox_down_m * 5.0)
            self.query_nn_k = config_args["neuralpoints"].get("query_nn_k", self.query_nn_k)
            self.num_nei_cells = config_args["neuralpoints"].get("num_nei_cells", self.num_nei_cells)
            self.search_alpha = config_args["neuralpoints"].get("search_alpha", self.search_alpha)
            self.feature_dim = config_args["neuralpoints"].get("feature_dim", self.feature_dim)
            self.color_feature_dim = config_args["neuralpoints"].get("color_feature_dim", self.color_feature_dim)
            # weighted the neighborhood feature before decoding to sdf or do the weighting of the decoded 
            # sdf afterwards, weighted first is faster, but may have some problem during the neural point map update after pgo
            self.weighted_first = config_args["neuralpoints"].get("weighted_first", self.weighted_first) 
            # build the neural point map from the surface samples or only the measurement points
            self.from_sample_points = config_args["neuralpoints"].get("from_sample_points", self.from_sample_points)
            if self.from_sample_points:
                self.map_surface_ratio = config_args["neuralpoints"].get("map_surface_ratio", self.map_surface_ratio)
            self.prune_map_on = config_args["neuralpoints"].get("prune_map_on", self.prune_map_on)
            self.max_prune_certainty = config_args["neuralpoints"].get("max_prune_certainty", self.max_prune_certainty)
            self.use_mid_ts = config_args["neuralpoints"].get("use_mid_ts", self.use_mid_ts)
            self.local_map_travel_dist_ratio = config_args["neuralpoints"].get("local_map_travel_dist_ratio", self.local_map_travel_dist_ratio)
            

        # decoder
        if "decoder" in config_args: # only on if indicated
            # number of the level of the mlp decoder
            self.geo_mlp_level = config_args["decoder"].get("mlp_level", self.geo_mlp_level)
            # dimension of the mlp's hidden layer
            self.geo_mlp_hidden_dim = config_args["decoder"].get("mlp_hidden_dim", self.geo_mlp_hidden_dim) 
            
            self.gs_mlp_level = config_args["decoder"].get("gs_mlp_level", self.gs_mlp_level)
            self.gs_mlp_hidden_dim = config_args["decoder"].get("gs_mlp_hidden_dim", self.gs_mlp_hidden_dim)
            
            # freeze the decoder after runing for x frames (used for incremental mapping to avoid forgeting)
            self.freeze_after_frame = config_args["decoder"].get("freeze_after_frame", self.freeze_after_frame) 

        # FIXME, now set to the same as geo mlp, but actually can be different
        self.color_mlp_level = self.geo_mlp_level
        self.color_mlp_hidden_dim = self.geo_mlp_hidden_dim
        self.sem_mlp_level = self.geo_mlp_level
        self.sem_mlp_hidden_dim = self.geo_mlp_hidden_dim

        # loss
        if "loss" in config_args:
            self.main_loss_type = config_args["loss"].get("main_loss_type", "bce")
            self.sigma_sigmoid_m = config_args["loss"].get("sigma_sigmoid_m", self.vox_down_m)
            self.loss_weight_on = config_args["loss"].get("loss_weight_on", self.loss_weight_on)
            if self.loss_weight_on:
                self.dist_weight_scale = config_args["loss"].get("dist_weight_scale", self.dist_weight_scale)
                # apply "behind the surface" loss weight drop-off or not
                self.behind_dropoff_on = config_args["loss"].get("behind_dropoff_on", self.behind_dropoff_on)
            self.weight_i = float(config_args["loss"].get("weight_color", self.weight_i))
            
            self.ekional_loss_on = config_args["loss"].get("ekional_loss_on", self.ekional_loss_on) # use ekional loss (norm(gradient) = 1 loss)
            self.weight_e = float(config_args["loss"].get("weight_e", self.weight_e))
            self.numerical_grad = config_args["loss"].get("numerical_grad_on", self.numerical_grad)
            if not self.numerical_grad:
                self.gradient_decimation = 1
            else:
                self.gradient_decimation = config_args["loss"].get("grad_decimation", self.gradient_decimation)
                self.num_grad_step_ratio = config_args["loss"].get("num_grad_step_ratio", self.num_grad_step_ratio)

        # rehersal (replay) based method
        if "continual" in config_args:
            self.pool_capacity = int(float(config_args["continual"].get("pool_capacity", self.pool_capacity)))
            self.bs_new_sample = int(config_args["continual"].get("batch_size_new_sample", self.bs_new_sample))
            self.new_certainty_thre = float(config_args["continual"].get("new_certainty_thre", self.new_certainty_thre))
            self.pool_filter_freq = config_args["continual"].get("pool_filter_freq", 1)
            self.pool_filter_with_dist = config_args["continual"].get("pool_filter_with_dist", self.pool_filter_with_dist)
        
        # tracker
        if "tracker" in config_args:
            self.track_on = True
            if self.color_on:
                self.photometric_loss_on = config_args["tracker"].get("photo_loss", self.photometric_loss_on)
                if self.photometric_loss_on:
                    self.photometric_loss_weight = float(config_args["tracker"].get("photo_weight", self.photometric_loss_weight))
                self.consist_wieght_on = config_args["tracker"].get("consist_wieght", self.consist_wieght_on)
            self.uniform_motion_on = config_args["tracker"].get("uniform_motion_on", self.uniform_motion_on)
            self.source_vox_down_m = config_args["tracker"].get("source_vox_down_m", self.vox_down_m * 10.0)
            self.reg_iter_n = config_args["tracker"].get("iter_n", self.reg_iter_n)
            self.track_mask_query_nn_k = config_args["tracker"].get("valid_nn_k", self.track_mask_query_nn_k)
            self.reg_min_grad_norm = config_args["tracker"].get("min_grad_norm", self.reg_min_grad_norm)
            self.reg_max_grad_norm = config_args["tracker"].get("max_grad_norm", self.reg_max_grad_norm)
            self.reg_min_valid_points = int(config_args["tracker"].get("min_valid_points", self.reg_min_valid_points))
            self.reg_min_valid_ratio = float(config_args["tracker"].get("min_valid_ratio", self.reg_min_valid_ratio))
            self.reg_GM_grad = config_args["tracker"].get("GM_grad", self.reg_GM_grad)
            self.reg_GM_dist_m = config_args["tracker"].get("GM_dist", self.reg_GM_dist_m)
            self.reg_lm_lambda = float(config_args["tracker"].get("lm_lambda", self.reg_lm_lambda))
            self.reg_term_thre_deg = float(config_args["tracker"].get("term_deg", self.reg_term_thre_deg))
            self.reg_term_thre_m = float(config_args["tracker"].get("term_m", self.reg_term_thre_m))
            self.eigenvalue_check = config_args["tracker"].get("eigenvalue_check", self.eigenvalue_check)

        # pgo
        if self.track_on:
            if "pgo" in config_args:
                self.pgo_on = True
                self.local_map_context = config_args["pgo"].get("map_context", self.local_map_context)
                self.loop_with_feature = config_args["pgo"].get("loop_with_feature", self.loop_with_feature)
                self.local_map_context_latency = config_args["pgo"].get('local_map_latency', self.local_map_context_latency)
                self.context_virtual_side_count = config_args["pgo"].get("virtual_side_count", self.context_virtual_side_count)
                self.context_virtual_step_m = config_args["pgo"].get("virtual_step_m", self.voxel_size_m * 4.0)
                self.npmc_max_dist = config_args["pgo"].get("npmc_max_dist", self.max_range * 0.7)
                self.pgo_freq = config_args["pgo"].get("pgo_freq_frame", self.pgo_freq)
                self.pgo_with_pose_prior = config_args["pgo"].get("with_pose_prior", self.pgo_with_pose_prior)
                # default cov (constant for all the edges)
                self.pgo_tran_std = float(config_args["pgo"].get("tran_std", self.pgo_tran_std))
                self.pgo_rot_std = float(config_args["pgo"].get("rot_std", self.pgo_rot_std))
                # use default or estimated cov
                self.use_reg_cov_mat = config_args["pgo"].get("use_reg_cov", False)
                # merge the neural point map or not after the loop, merge the map may lead to some holes
                self.pgo_error_thre = float(config_args["pgo"].get("pgo_error_thre_frame", self.pgo_error_thre_frame))
                self.pgo_max_iter = config_args["pgo"].get("pgo_max_iter", self.pgo_max_iter) 
                self.pgo_merge_map = config_args["pgo"].get("merge_map", False) 
                self.context_cosdist_threshold = config_args["pgo"].get("context_cosdist", self.context_cosdist_threshold) 
                self.min_loop_travel_dist_ratio = config_args["pgo"].get("min_loop_travel_ratio", self.min_loop_travel_dist_ratio) 
                self.loop_dist_drift_ratio_thre = config_args["pgo"].get("max_loop_dist_ratio", self.loop_dist_drift_ratio_thre)
                self.local_loop_dist_thre = config_args["pgo"].get("local_loop_dist_thre", self.voxel_size_m * 5.0)
            
        # mapping optimizer
        if "optimizer" in config_args:
            self.mapping_freq_frame = config_args["optimizer"].get("mapping_freq_frame", 1)
            self.adaptive_iters = config_args["optimizer"].get("adaptive_iters", self.adaptive_iters)
            self.iters = config_args["optimizer"].get("iters", self.iters) # mapping iters per frame
            self.init_iter_ratio = config_args["optimizer"].get("init_iter_ratio", self.init_iter_ratio) # iteration count ratio for the first frame (a kind of warm-up) #iter = init_iter_ratio*iter
            self.bs = config_args["optimizer"].get("batch_size", self.bs)
            # learning rate for neural points
            self.lr_geo = float(config_args["optimizer"].get("learning_rate_geo", self.lr_geo))
            self.lr_color = float(config_args["optimizer"].get("learning_rate_color", self.lr_color))
            # for mlps
            self.lr_mlp_base = float(config_args["optimizer"].get("learning_rate_mlp_base", self.lr_mlp_base))
            self.lr_mlp_gs_xyz = float(config_args["optimizer"].get("learning_rate_mlp_gs_xyz", self.lr_mlp_gs_xyz))  
            self.lr_mlp_gs_alpha = float(config_args["optimizer"].get("learning_rate_mlp_gs_alpha", self.lr_mlp_gs_alpha))
            self.lr_mlp_gs_scale = float(config_args["optimizer"].get("learning_rate_mlp_gs_scale", self.lr_mlp_gs_scale))
            self.lr_mlp_gs_rot = float(config_args["optimizer"].get("learning_rate_mlp_gs_rot", self.lr_mlp_gs_rot))
            self.lr_mlp_gs_color = float(config_args["optimizer"].get("learning_rate_mlp_gs_color", self.lr_mlp_gs_color))
            # for exposures
            self.lr_exposure = float(config_args["optimizer"].get("learning_rate_exposure", self.lr_exposure))

            self.lr_cam_dr = float(config_args["optimizer"].get("learning_rate_cam_dr", self.lr_cam_dr)) # 0.003 # learning rate for camera rotation
            self.lr_cam_dt = float(config_args["optimizer"].get("learning_rate_cam_dt", self.lr_cam_dt))# 0.001 # learning rate for camera translation

        # gaussian splatting
        if "gs" in config_args:
            self.gs_on = True
            self.gs_type = config_args["gs"].get("gs_type", self.gs_type)

            self.exposure_correction_on = config_args["gs"].get("exposure_correction_on", self.exposure_correction_on)
            self.affine_exposure_correction = config_args["gs"].get("affine_exposure_correction", self.affine_exposure_correction)

            self.cam_pose_train_on = config_args["gs"].get("cam_pose_train_on", self.cam_pose_train_on)

            self.gs_invalid_check_on = config_args["gs"].get("invalid_check_on", self.gs_invalid_check_on) 
            self.min_valid_nn_k = config_args["gs"].get("min_valid_nn_k", self.min_valid_nn_k)

            self.monodepth_on = config_args["gs"].get("monodepth_on", self.monodepth_on)
            self.monodepth_gaussian_res = config_args["gs"].get("monodepth_gaussian_res", self.voxel_size_m * 5.0)

            # spawning related
            self.spawn_n_gaussian = config_args["gs"].get("n_gaussian", self.spawn_n_gaussian)
            self.dist_concat_on = config_args["gs"].get("dist_concat_on", self.dist_concat_on)
            self.view_concat_on = config_args["gs"].get("view_concat_on", self.view_concat_on)
            self.learn_color_residual = config_args["gs"].get("learn_color_residual", self.learn_color_residual)
            self.displacement_range_ratio = float(config_args["gs"].get("displacement_range_ratio", self.displacement_range_ratio))
            self.max_scale_ratio = float(config_args["gs"].get("max_scale_ratio", self.max_scale_ratio))
            self.unit_scale_ratio = float(config_args["gs"].get("unit_scale_ratio", self.unit_scale_ratio))

            self.train_front_only = config_args["gs"].get("train_front_only", self.train_front_only)

            self.gs_iters = config_args["gs"].get("gs_iters", self.gs_iters)
            self.nothing_new_count_thre = config_args["gs"].get("nothing_new_count_thre", self.nothing_new_count_thre)
            
            self.gaussian_bs_ratio = config_args["gs"].get("gaussian_bs_ratio", self.gaussian_bs_ratio) # gaussian count per iter (for gsdf consistency loss)
            
            self.gs_keyframe_interval = config_args["gs"].get("gs_keyframe_interval", self.gs_keyframe_interval)
            self.gs_keyframe_accu_travel_dist = config_args["gs"].get("gs_keyframe_accu_dist", self.max_range*0.02) # default value set to be self.max_range*0.03 # TODO: change this later
            self.gs_keyframe_accu_travel_degree = config_args["gs"].get("gs_keyframe_accu_degree", self.gs_keyframe_accu_travel_degree)

            self.img_pool_size = config_args["gs"].get("img_pool_size", self.img_pool_size)
            self.long_term_pool_size = config_args["gs"].get("long_term_img_pool_size", 2*self.img_pool_size)
            self.lastest_train_prob = config_args["gs"].get("lastest_train_prob", self.lastest_train_prob)
            self.short_term_train_prob = config_args["gs"].get("short_term_train_prob", self.short_term_train_prob)
            self.long_term_train_down = config_args["gs"].get("long_term_train_down", self.long_term_train_down)

            self.gs_down_rate = config_args["gs"].get("gs_down_rate", self.gs_down_rate)
            self.gs_vis_down_rate = config_args["gs"].get("gs_vis_down_rate", self.gs_vis_down_rate)
            self.min_visible_neural_point_ratio = config_args["gs"].get("min_visible_neural_point_ratio", self.min_visible_neural_point_ratio)
            
            self.inverse_depth_loss = config_args["gs"].get("inverse_depth_loss", self.inverse_depth_loss)
            
            self.lambda_ssim= float(config_args["gs"].get("lambda_ssim", self.lambda_ssim)) # weight for ssim, set to zero for faster training
            self.lambda_depth = float(config_args["gs"].get("lambda_depth", self.lambda_depth))
            self.lambda_distort = float(config_args["gs"].get("lambda_distort", self.lambda_distort)) # weight for the distance distortion loss
            self.lambda_normal_depth_consist = float(config_args["gs"].get("lambda_normal_depth", self.lambda_normal_depth_consist)) # weight for the distance/normal consistency loss
            self.lambda_normal_smooth = float(config_args["gs"].get("lambda_normal_smooth", self.lambda_normal_smooth)) # weight for the distance/normal consistency loss
            self.lambda_mono_normal = float(config_args["gs"].get("lambda_mono_normal", self.lambda_mono_normal))
            self.lambda_isotropic = float(config_args["gs"].get("lambda_isotropic", self.lambda_isotropic))
            self.lambda_area = float(config_args["gs"].get("lambda_area", self.lambda_area))
            self.lambda_opacity = float(config_args["gs"].get("lambda_opacity", self.lambda_opacity))
            self.lambda_opacity_ent = float(config_args["gs"].get("lambda_opacity_ent", self.lambda_opacity_ent))
            self.lambda_sky = float(config_args["gs"].get("lambda_sky", self.lambda_sky))
            self.lambda_sdf_cons = float(config_args["gs"].get("lambda_sdf_cons", self.lambda_sdf_cons))
            self.lambda_sdf_normal_cons = float(config_args["gs"].get("lambda_sdf_normal_cons", self.lambda_sdf_normal_cons))
            self.lambda_invalid_opacity = float(config_args["gs"].get("lambda_invalid_opacity", self.lambda_invalid_opacity)) # add this to better deal with dynamic objects
            self.lambda_sdf = float(config_args["gs"].get("lambda_sdf", self.lambda_sdf))

            self.gs_consist_shift_count = int(config_args["gs"].get("consist_shift_count", self.gs_consist_normal_fixed))
            self.gs_consist_shift_range_m = float(config_args["gs"].get("consist_shift_range_m", self.gs_consist_shift_range_m))

            self.gs_consist_normal_fixed = config_args["gs"].get("consist_normal_fixed", self.gs_consist_normal_fixed)
            self.gs_consist_depth_fixed = config_args["gs"].get("consist_depth_fixed", self.gs_consist_depth_fixed)

            self.gs_contribution_threshold = float(config_args["gs"].get("contribution_threshold", self.gs_contribution_threshold))

            self.min_alpha = config_args["gs"].get("min_alpha", self.min_alpha) # this is the per-gaussian alpha
            self.depth_min_accu_alpha = config_args["gs"].get("depth_min_accu_alpha", self.depth_min_accu_alpha) # this is the rendered accumulated alpha for valid depth
            self.eval_depth_min_accu_alpha = config_args["gs"].get("eval_depth_min_accu_alpha", self.eval_depth_min_accu_alpha) # this is the rendered accumulated alpha for valid depth

            self.gs_eval_cam_refine_on = config_args["gs"].get("eval_cam_refine_on", self.gs_eval_cam_refine_on) 
        
        # vis and eval
        if "eval" in config_args:
            # use weight and bias to monitor the experiment or not
            self.wandb_vis_on = config_args["eval"].get("wandb_vis_on", self.wandb_vis_on)
            self.silence = config_args["eval"].get("silence_log", self.silence)
            # turn on the open3d visualizer to visualize the mapping progress or not
            self.o3d_vis_on = config_args["eval"].get("o3d_vis_on", self.o3d_vis_on)
            # path to the sensor cad file
            self.sensor_cad_path = config_args["eval"].get('sensor_cad_path', None)
            
            # frequency for pose log (per x frame)
            self.log_freq_frame = config_args["eval"].get('log_freq_frame', 0)
            # frequency for mesh reconstruction (per x frame)
            self.mesh_freq_frame = config_args["eval"].get('mesh_freq_frame', self.mesh_freq_frame)
            # keep the previous reconstructed mesh in the visualizer or not
            self.keep_local_mesh = config_args["eval"].get('keep_local_mesh', self.keep_local_mesh)
            # frequency for sdf slice visualization (per x frame)
            self.sdfslice_freq_frame = config_args["eval"].get('sdf_freq_frame', 1)
            self.sdf_slice_height = config_args["eval"].get('sdf_slice_height', self.sdf_slice_height) # in sensor frame, unit: m
            
            # mesh reconstruction
            self.mesh_default_on = config_args["eval"].get('mesh_default_on', self.mesh_default_on)
            self.mesh_min_nn = config_args["eval"].get('mesh_min_nn', self.mesh_min_nn)
            self.skip_top_voxel = config_args["eval"].get('skip_top_voxel', self.skip_top_voxel)
            self.min_cluster_vertices = config_args["eval"].get('min_cluster_vertices', self.min_cluster_vertices)
            self.mc_res_m = config_args["eval"].get('mc_res_m', self.voxel_size_m*0.6) # initial marching cubes grid sampling interval (unit: m)
            
            # for baseline only
            self.tsdf_fusion_voxel_size = config_args["eval"].get('tsdf_fusion_voxel_size', self.tsdf_fusion_voxel_size)
            self.tsdf_fusion_space_carving_on = config_args["eval"].get('tsdf_fusion_space_carving_on', self.tsdf_fusion_space_carving_on)
            self.rerender_tsdf_fusion_on = config_args["eval"].get('rerender_tsdf_fusion_on', self.rerender_tsdf_fusion_on) # use the rerendered depth from radiance field for tsdf fusion as used in 2DGS

            # save the map or not
            self.save_map = config_args["eval"].get('save_map', self.save_map)
            self.save_merged_pc = config_args["eval"].get('save_merged_pc', self.save_merged_pc)
            self.save_mesh = config_args["eval"].get('save_mesh', self.save_mesh)

            # gs visualizer
            self.visualizer_split_width_ratio = config_args["eval"].get('visualizer_split_width_ratio', self.visualizer_split_width_ratio)
            
            self.gs_eval_on = config_args["eval"].get("gs_eval_on", self.gs_eval_on)
            self.rendered_pc_eval_on = config_args["eval"].get('rendered_pc_eval_on', self.rendered_pc_eval_on)

            self.neural_point_vis_down_rate = config_args["eval"].get('neural_point_vis_down_rate', self.neural_point_vis_down_rate)  
            
            self.vis_frame_axis_len = config_args["eval"].get('vis_frame_len', self.max_range / 50.0)

        # associated parameters
        self.infer_bs = self.bs * 8
        self.local_map_radius = min(self.max_range*1.05, self.max_range+5.0) # for the local neural points
        self.window_radius = max(self.local_map_radius+self.voxel_size_m*2, 6.0) # for the sampling data pool, should not be too small
        self.sorrounding_map_radius = self.local_map_radius * 1.4
