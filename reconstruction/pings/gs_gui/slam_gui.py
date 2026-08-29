# This file is adapted from the GUI of [MonoGS](https://github.com/muskie82/MonoGS)
# Yue Pan, 2025

# NOTE: if your computer has some issue working with OpenGL, set this to True, 
# then the visualization of Gaussian ellipsoid would be disabled.
gl_issue = True

from typing import Dict, List, Tuple
import threading
import time
from datetime import datetime

import cv2
import os

import matplotlib.cm as cm
import numpy as np
import copy
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import torch

from pickle import load, dump

# from brisque import BRISQUE

from gaussian_splatting.gaussian_renderer import render, spawn_gaussians
from gaussian_splatting.utils.graphics_utils import fov2focal, getWorld2View2
from gaussian_splatting.utils.image_utils import psnr
from gaussian_splatting.utils.cameras import CamImage
from gs_gui.gui_utils import (
    VisPacket,
    ControlPacket,
    create_frustum,
    cv_gl,
    get_latest_queue,   
)

if not gl_issue:
    from OpenGL import GL as gl
    import glfw
    from gs_gui.gl_render import util, util_gau
    from gs_gui.gl_render.render_ogl import OpenGLRenderer
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"

from utils.tools import colorize_depth_maps, seed_anything, get_time, remove_gpu_cache, find_closest_prime

# o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

RED = np.array([255, 0, 0]) / 255.0
PURPLE = np.array([238, 130, 238]) / 255.0
BLACK = np.array([0, 0, 0]) / 255.0
GOLDEN = np.array([255, 215, 0]) / 255.0
SILVER = np.array([192, 192, 192]) / 255.0
GREEN = np.array([0, 128, 0]) / 255.0
BLUE = np.array([0, 0, 128]) / 255.0
LIGHTBLUE = np.array([0, 166, 237]) / 255.0

ToGLCamera = np.array([
    [1,  0,  0,  0],
    [0,  -1,  0,  0],
    [0,  0,  -1,  0],
    [0,  0,  0,  1]
])
FromGLGamera = np.linalg.inv(ToGLCamera)

class SLAM_GUI:
    def __init__(self, params_gui=None):
        self.step = 0
        self.process_finished = False
        self.device = "cuda"

        self.frustum_dict = {}
        self.keyframe_dict = {}
        self.model_dict = {}

        self.q_main2vis = None
        self.cur_data_packet = None

        self.cur_base_gaussians = None # Dict: these are background gaussians stored in the visualizer

        self.decoders = None

        self.background = None
        self.config = None

        self.init = False
        self.kf_window = None
        self.render_img = None

        self.show_rendered_img = True

        self.brisque_score_on = False # this is deprecated

        self.neural_point_vis_down_rate = 1

        self.frustum_size = 0.05

        self.local_map_default_on = True
        self.mesh_default_on = False
        self.sdf_default_on = False
        self.neural_point_map_default_on = False
        self.robot_default_on = True
        self.neural_point_color_default_mode = 1
        self.neural_point_vis_down_rate = 1

        if params_gui is not None:
            self.decoders = params_gui.decoders
            self.background = params_gui.background
            # self.init = False
            self.q_main2vis = params_gui.q_main2vis
            self.q_vis2main = params_gui.q_vis2main
            self.config = params_gui.config
            self.gs_default_on = params_gui.gs_default_on
            self.robot_default_on = params_gui.robot_default_on
            self.neural_point_map_default_on = params_gui.neural_point_map_default_on
            self.mesh_default_on = params_gui.mesh_default_on
            self.neural_point_color_default_mode = params_gui.neural_point_color_default_mode
            self.is_rgbd = params_gui.is_rgbd
            self.neural_point_vis_down_rate = params_gui.neural_point_vis_down_rate
            self.frustum_size = params_gui.frustum_size
            self.local_map_default_on = params_gui.local_map_default_on
            self.still_view_default_on = params_gui.still_view_default_on
            self.follow_cam_default_on = params_gui.follow_cam_default_on
            
        if self.config is not None:
            seed_anything(self.config.seed)

        self.init_widget()

        self.cur_frame_id = -1

        self.gaussian_nums = []

        # self.brisque_scorer = BRISQUE(url=False)

        self.recorded_poses = []

        self.view_save_base_path = os.path.expanduser("~/.viewpoints")
        os.makedirs(self.view_save_base_path, 0o755, exist_ok=True)

        # these are only used for the elliopsoid rendering 

        # print("[GUI] gl_issue:", gl_issue)
        self.gl_issue_notified = False

        if not gl_issue:
      
            self.g_camera = util.Camera(self.window_h, self.window_w)
            self.window_gl = self.init_glfw() # this has no issue

            # something wrong here with the glfw (just crash) after I use mini-forge
            # solution:
            # os.environ["PYOPENGL_PLATFORM"] = "osmesa"
            # or set in your conda environment
            # export PYOPENGL_PLATFORM=osmesa
            # reference: 
            # https://github.com/facebookresearch/AnimatedDrawings/issues/99

            self.g_renderer = OpenGLRenderer(self.g_camera.w, self.g_camera.h)  

            # gl.glEnable(gl.GL_TEXTURE_2D)
            gl.glEnable(gl.GL_DEPTH_TEST)
            gl.glDepthFunc(gl.GL_LEQUAL)
            self.gaussians_gl = util_gau.GaussianData(0, 0, 0, 0, 0)

        # screenshot saving path
        save_path = os.path.join(self.config.run_path, "log")
        os.makedirs(save_path, 0o755, exist_ok=True)

        self.save_dir_2d_screenshots = os.path.join(save_path, "2d_screenshots")
        os.makedirs(self.save_dir_2d_screenshots, 0o755, exist_ok=True)

        self.save_dir_3d_screenshots = os.path.join(save_path, "3d_screenshots")
        os.makedirs(self.save_dir_3d_screenshots, 0o755, exist_ok=True)

        threading.Thread(target=self._update_thread).start()

    # has some issue here
    def init_widget(self):
        self.window_w, self.window_h = 1600, 900
        # self.window_w, self.window_h = 2560, 1600

        self.window = gui.Application.instance.create_window(
           "📌 PINGS Viewer", self.window_w, self.window_h
        )
        self.window.set_on_layout(self._on_layout)
        self.window.set_on_close(self._on_close)

        self.widget3d = gui.SceneWidget()
        self.widget3d.scene = rendering.Open3DScene(self.window.renderer)

        cg_settings = rendering.ColorGrading(
            rendering.ColorGrading.Quality.ULTRA,
            rendering.ColorGrading.ToneMapping.LINEAR,
        )
        self.widget3d.scene.view.set_color_grading(cg_settings)
        # self.widget3d.scene.show_skybox(False)

        self.window.add_child(self.widget3d)

        # not used now
        self.lit = rendering.MaterialRecord()
        self.lit.shader = "unlitLine"
        self.lit.line_width = 4 * self.window.scaling  # note that this is scaled with respect to pixels,

        self.lit_geo = rendering.MaterialRecord()
        self.lit_geo.shader = "defaultUnlit"

        # scan
        self.scan_render = rendering.MaterialRecord()
        self.scan_render.shader = "defaultLit" # "defaultUnlit", "normals", "depth"
        self.scan_render_init_size_unit = 2
        self.scan_render.point_size = self.scan_render_init_size_unit * self.window.scaling
        self.scan_render.base_color = [0.9, 0.9, 0.9, 0.8]

        # neural points
        self.neural_points_render = rendering.MaterialRecord()
        self.neural_points_render.shader = "defaultLit"
        self.neural_points_render_init_size_unit = 3
        self.neural_points_render.point_size = self.neural_points_render_init_size_unit * self.window.scaling
        self.neural_points_render.base_color = [0.9, 0.9, 0.9, 0.8]

        # sdf slice
        self.sdf_render = rendering.MaterialRecord()
        self.sdf_render.shader = "defaultLit"
        self.sdf_render.point_size = 10 * self.window.scaling
        self.sdf_render.base_color = [1.0, 1.0, 1.0, 1.0]

        # sdf sample pool
        self.sdf_pool_render = rendering.MaterialRecord()
        self.sdf_pool_render.shader = "defaultLit"
        self.sdf_pool_render.point_size = 1 * self.window.scaling
        self.sdf_pool_render.base_color = [1.0, 1.0, 1.0, 1.0]

        # mesh 
        self.mesh_render = rendering.MaterialRecord()
        # self.mesh_render.shader = "defaultLit"
        self.mesh_render.shader = "normals"

        # trajectory
        self.traj_render = rendering.MaterialRecord()
        self.traj_render.shader = "unlitLine"
        self.traj_render.line_width = 4 * self.window.scaling  # note that this is scaled with respect to pixels,

        # cur frame frustrum
        self.cur_frame_render = rendering.MaterialRecord()
        self.cur_frame_render.shader = "unlitLine"
        self.cur_frame_render.line_width = 3 * self.window.scaling

        # train frame frustrum
        self.train_frame_render = rendering.MaterialRecord()
        self.train_frame_render.shader = "unlitLine"
        self.train_frame_render.line_width = 2 * self.window.scaling

        # range ring
        self.ring_render = rendering.MaterialRecord()
        self.ring_render.shader = "unlitLine"
        self.ring_render.line_width = 2 * self.window.scaling  # note that this is scaled with respect to pixels,

        self.cad_render = rendering.MaterialRecord()
        self.cad_render.shader = "defaultLit"
        self.cad_render.base_color = [0.9, 0.9, 0.9, 1.0]

        # deprecated, coordinate frame
        self.axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.5, origin=[0, 0, 0]
        )

        # other geometry entities
        self.mesh = o3d.geometry.TriangleMesh()
        self.scan = o3d.geometry.PointCloud()
        self.rendered_scan = o3d.geometry.PointCloud()
        self.sdf_pool = o3d.geometry.PointCloud() # sample pool
        self.sdf_slice = o3d.geometry.PointCloud()
        self.neural_points = o3d.geometry.PointCloud()
        self.invalid_neural_points = o3d.geometry.PointCloud()
        self.sensor_cad = o3d.geometry.TriangleMesh()
        self.sensor_cad_origin = o3d.geometry.TriangleMesh()

        if self.config.sensor_cad_path is not None:
            self.sensor_cad_origin = o3d.io.read_triangle_mesh(self.config.sensor_cad_path)
            self.sensor_cad_origin.compute_vertex_normals()

        self.odom_traj = o3d.geometry.LineSet()
        self.slam_traj = o3d.geometry.LineSet()
        self.gt_traj = o3d.geometry.LineSet()
        self.loop_edges = o3d.geometry.LineSet()

        # range circles
        self.range_circle = o3d.geometry.LineSet()
        circle_points_1 = generate_circle(radius=self.config.max_range/2, num_points=100)
        lines1 = [[i, (i + 1) % len(circle_points_1)] for i in range(len(circle_points_1))]
        range_circle1 = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(circle_points_1),
            lines=o3d.utility.Vector2iVector(lines1),
        )
        circle_points_2 = generate_circle(radius=self.config.max_range, num_points=100)
        lines2 = [[i, (i + 1) % len(circle_points_2)] for i in range(len(circle_points_2))]
        range_circle2 = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(circle_points_2),
            lines=o3d.utility.Vector2iVector(lines2),
        )
        self.range_circle_origin = range_circle1 + range_circle2
        self.range_circle_origin.paint_uniform_color(LIGHTBLUE)

        bounds = self.widget3d.scene.bounding_box
        self.widget3d.setup_camera(60.0, bounds, bounds.get_center())
        
        em = self.window.theme.font_size
        margin = 0.5 * em
        
        self.panel = gui.Vert(0.5 * em, gui.Margins(margin))

        tab_margins = gui.Margins(0, int(np.round(0.5 * em)), 0, 0)

        tabs0 = gui.TabControl()

        tab_setting = gui.Vert(0.2 * em, tab_margins)

        slider_line = gui.Horiz(1.0 * em, gui.Margins(margin))
        
        # these are not button, but rather switch
        self.slider_slam = gui.ToggleSwitch("Pause / Resume SLAM")
        self.slider_slam.is_on = True # default on
        self.slider_slam.set_on_clicked(self._on_slam_slider)
        slider_line.add_child(self.slider_slam)

        self.slider_render = gui.ToggleSwitch("Pause / Resume Rendering")
        self.slider_render.is_on = True # default on
        self.slider_render.set_on_clicked(self._on_vis_slider)
        slider_line.add_child(self.slider_render)

        # self.slider_recording = gui.ToggleSwitch("Pause / Resume Recording")
        # self.slider_recording.is_on = False # default off
        # slider_line.add_child(self.slider_recording)

        tab_setting.add_child(slider_line)

        # ------------------------------------------------------------
        # View Options
        collapse_view = gui.CollapsableVert("View Options", 0.2 * em,
                                       gui.Margins(margin))
        collapse_view.set_is_open(True) 

        viewpoint_tile = gui.Horiz(0.5 * em, gui.Margins(margin))

        ##Check boxes
        self.local_map_chbox = gui.Checkbox("Local")
        self.local_map_chbox.checked = self.local_map_default_on
        viewpoint_tile.add_child(self.local_map_chbox)
        
        self.followcam_chbox = gui.Checkbox("Follow")
        self.followcam_chbox.checked = self.follow_cam_default_on
        viewpoint_tile.add_child(self.followcam_chbox)

        self.staybehind_chbox = gui.Checkbox("Behind")
        self.staybehind_chbox.checked = True
        viewpoint_tile.add_child(self.staybehind_chbox)

        self.still_chbox = gui.Checkbox("Still")
        self.still_chbox.checked = self.still_view_default_on
        viewpoint_tile.add_child(self.still_chbox)

        self.fly_chbox = gui.Checkbox("Fly")
        # NOTE: in fly mode, you can control like a game using WASD,Q,Z,E,R, up, right, left, down
        self.fly_chbox.checked = False
        self.fly_chbox.set_on_checked(self._set_mouse_mode)
        viewpoint_tile.add_child(self.fly_chbox)

        # self.bev_chbox = gui.Checkbox("BEV")
        # self.bev_chbox.checked = False
        # viewpoint_tile.add_child(self.bev_chbox)
        
        collapse_view.add_child(viewpoint_tile)

        viewpoint_tile_2 = gui.Horiz(0.2 * em, tab_margins)

        ##Combo panels for current frames
        combo_tile = gui.Vert(0.0 * em, gui.Margins(margin))
        self.combo_cams = gui.Combobox()
        self.combo_cams.set_on_selection_changed(self._on_combo_cams)
        combo_tile.add_child(gui.Label("Cameras"))
        combo_tile.add_child(self.combo_cams)

        ##Combo panels for train frames
        combo_tile2 = gui.Vert(0.0 * em, gui.Margins(margin))
        self.combo_train_cams = gui.Combobox()
        self.combo_train_cams.set_on_selection_changed(self._on_combo_train_cams)
        combo_tile2.add_child(gui.Label("Train Frames"))
        combo_tile2.add_child(self.combo_train_cams)

        ##Combo panels for preset views 
        combo_tile3 = gui.Vert(0.0 * em, gui.Margins(margin))
        self.combo_preset_cams = gui.Combobox()
        for i in range(30):
            self.combo_preset_cams.add_item(str(i))

        # self.combo_preset_cams.set_on_selection_changed(self._on_combo_preset_cams) 
        combo_tile3.add_child(gui.Label("Preset"))
        combo_tile3.add_child(self.combo_preset_cams)

        self.save_view_btn = gui.Button("Save")
        self.save_view_btn.set_on_clicked(
            self._on_save_view_btn
        )  # set the callback function

        self.load_view_btn = gui.Button("Load")
        self.load_view_btn.set_on_clicked(
            self._on_load_view_btn
        )  # set the callback function

        self.reset_view_btn = gui.Button("Reset")
        self.reset_view_btn.set_on_clicked(
            self._on_reset_view_btn
        )  # set the callback function

        viewpoint_tile_2.add_child(combo_tile)
        viewpoint_tile_2.add_child(combo_tile2)
        viewpoint_tile_2.add_child(combo_tile3)

        viewpoint_tile_2.add_child(self.save_view_btn)
        viewpoint_tile_2.add_child(self.load_view_btn)
        viewpoint_tile_2.add_child(self.reset_view_btn)
        
        collapse_view.add_child(viewpoint_tile_2)

        tab_setting.add_child(collapse_view)

        # ------------------------------------------------------------  
        # 3D Entities
        collapse_3dobj = gui.CollapsableVert("3D Entities", 0.4 * em,
                                       gui.Margins(margin))
        collapse_3dobj.set_is_open(True)

        chbox_tile_3dobj = gui.Horiz(0.5 * em, gui.Margins(margin))

        self.gs_chbox = gui.Checkbox("GS")
        self.gs_chbox.checked = self.gs_default_on
        # self.gs_chbox.set_on_checked(self._on_gs_chbox)
        chbox_tile_3dobj.add_child(self.gs_chbox)

        self.cameras_chbox = gui.Checkbox("Cameras")
        self.cameras_chbox.checked = True
        self.cameras_chbox.set_on_checked(self._on_cameras_chbox)
        chbox_tile_3dobj.add_child(self.cameras_chbox)

        self.keyframe_chbox = gui.Checkbox("Train Cameras")
        self.keyframe_chbox.checked = False
        self.keyframe_chbox.set_on_checked(self._on_keyframes_chbox)
        chbox_tile_3dobj.add_child(self.keyframe_chbox)

        self.mesh_chbox = gui.Checkbox("Mesh")
        self.mesh_chbox.checked = self.mesh_default_on
        self.mesh_chbox.set_on_checked(self._on_mesh_chbox)
        chbox_tile_3dobj.add_child(self.mesh_chbox)
        self.mesh_name = "pin_mesh"

        self.scan_chbox = gui.Checkbox("Scan")
        self.scan_chbox.checked = True
        self.scan_chbox.set_on_checked(self._on_scan_chbox)
        chbox_tile_3dobj.add_child(self.scan_chbox)
        self.scan_name = "cur_scan"

        chbox_tile_3dobj_2 = gui.Horiz(0.5 * em, gui.Margins(margin))

        self.neural_point_chbox = gui.Checkbox("Neural Point Map")
        self.neural_point_chbox.checked = self.neural_point_map_default_on
        self.neural_point_chbox.set_on_checked(self._on_neural_point_chbox)
        chbox_tile_3dobj_2.add_child(self.neural_point_chbox)
        self.neural_point_name = "neural_points"

        self.invalid_neural_point_chbox = gui.Checkbox("Invalid Points")
        self.invalid_neural_point_chbox.checked = False
        self.invalid_neural_point_chbox.set_on_checked(self._on_invalid_neural_point_chbox)
        chbox_tile_3dobj_2.add_child(self.invalid_neural_point_chbox)
        self.invalid_neural_point_name = "invalid_neural_points"

        self.rendered_scan_chbox = gui.Checkbox("Rendered Points")
        self.rendered_scan_chbox.checked = False
        self.rendered_scan_chbox.set_on_checked(self._on_rendered_scan_chbox)
        chbox_tile_3dobj_2.add_child(self.rendered_scan_chbox)
        self.rendered_scan_name = "cur_rendered_scan"
        
        chbox_tile_3dobj_3 = gui.Horiz(0.5 * em, gui.Margins(margin))

        self.cad_chbox = gui.Checkbox("Robot")
        self.cad_chbox.checked = self.robot_default_on
        self.cad_chbox.set_on_checked(self._on_cad_chbox)
        chbox_tile_3dobj_3.add_child(self.cad_chbox)
        self.cad_name = "sensor_cad"

        self.gt_traj_chbox = gui.Checkbox("GT Traj.")
        self.gt_traj_chbox.checked = False
        self.gt_traj_chbox.set_on_checked(self._on_gt_traj_chbox)
        chbox_tile_3dobj_3.add_child(self.gt_traj_chbox)
        self.gt_traj_name = "gt_trajectory"

        self.slam_traj_chbox = gui.Checkbox("SLAM Traj.")
        self.slam_traj_chbox.checked = False
        self.slam_traj_chbox.set_on_checked(self._on_slam_traj_chbox)
        chbox_tile_3dobj_3.add_child(self.slam_traj_chbox)
        self.slam_traj_name = "slam_trajectory"

        self.odom_traj_chbox = gui.Checkbox("Odom Traj.")
        self.odom_traj_chbox.checked = False
        self.odom_traj_chbox.set_on_checked(self._on_odom_traj_chbox)
        chbox_tile_3dobj_3.add_child(self.odom_traj_chbox)
        self.odom_traj_name = "odom_trajectory"

        chbox_tile_3dobj_4 = gui.Horiz(0.5 * em, gui.Margins(margin))

        self.sdf_chbox = gui.Checkbox("SDF Slice")
        self.sdf_chbox.checked = False
        self.sdf_chbox.set_on_checked(self._on_sdf_chbox)
        chbox_tile_3dobj_4.add_child(self.sdf_chbox)
        self.sdf_name = "cur_sdf_slice"

        self.sdf_pool_chbox = gui.Checkbox("SDF Samples")
        self.sdf_pool_chbox.checked = False
        self.sdf_pool_chbox.set_on_checked(self._on_sdf_pool_chbox)
        chbox_tile_3dobj_4.add_child(self.sdf_pool_chbox)
        self.sdf_pool_name = "sdf_sample_pool"

        self.loop_edges_chbox = gui.Checkbox("Loop")
        self.loop_edges_chbox.checked = False
        chbox_tile_3dobj_4.add_child(self.loop_edges_chbox)
        self.loop_edges_name = "loop_edges"

        self.range_circle_chbox = gui.Checkbox("Range Rings")
        self.range_circle_chbox.checked = False
        self.range_circle_chbox.set_on_checked(self._on_range_circle_chbox)
        chbox_tile_3dobj_4.add_child(self.range_circle_chbox)
        self.range_circle_name = "range_circle"

        collapse_3dobj.add_child(chbox_tile_3dobj)
        collapse_3dobj.add_child(chbox_tile_3dobj_2)
        collapse_3dobj.add_child(chbox_tile_3dobj_3)
        collapse_3dobj.add_child(chbox_tile_3dobj_4)

        tab_setting.add_child(collapse_3dobj)

        # ------------------------------------------------------------
        # GS Rendering Options
        if self.config.gs_on:
            gs_vis_collapse = gui.CollapsableVert("GS Rendering Options", 0.2 * em,
                                        gui.Margins(margin))
            gs_vis_collapse.set_is_open(False)

            chbox_tile_gsrender_1 = gui.Horiz(0.5 * em, gui.Margins(margin))

            # these cannot be on at the same time
            self.depth_chbox = gui.Checkbox("Depth")
            self.depth_chbox.checked = False
            self.depth_chbox.set_on_checked(self._on_depth_chbox)
            chbox_tile_gsrender_1.add_child(self.depth_chbox)

            self.normal_chbox = gui.Checkbox("Normal")
            self.normal_chbox.checked = False
            self.normal_chbox.set_on_checked(self._on_normal_chbox)
            chbox_tile_gsrender_1.add_child(self.normal_chbox)

            self.d2n_chbox = gui.Checkbox("D2N")
            self.d2n_chbox.checked = False
            self.d2n_chbox.set_on_checked(self._on_d2n_chbox)
            chbox_tile_gsrender_1.add_child(self.d2n_chbox)

            self.opacity_chbox = gui.Checkbox("Opacity")
            self.opacity_chbox.checked = False
            self.opacity_chbox.set_on_checked(self._on_opacity_chbox)
            chbox_tile_gsrender_1.add_child(self.opacity_chbox)

            self.ellipsoid_chbox = gui.Checkbox("Ellipsoid")
            self.ellipsoid_chbox.checked = False
            self.ellipsoid_chbox.set_on_checked(self._on_ellipsoid_chbox)
            chbox_tile_gsrender_1.add_child(self.ellipsoid_chbox)

            chbox_tile_gsrender_2 = gui.Horiz(0.5 * em, gui.Margins(margin))

            # self.time_shader_chbox = gui.Checkbox("Time Shader")
            # self.time_shader_chbox.checked = False
            # chbox_tile_gsrender_2.add_child(self.time_shader_chbox)

            self.backface_chbox = gui.Checkbox("Backface")
            self.backface_chbox.checked = False
            # self.backface_chbox.set_on_checked(self._on_backface_chbox)
            chbox_tile_gsrender_2.add_child(self.backface_chbox)

            self.elliopsoid_2d_chbox = gui.Checkbox("Surfel Mode")
            if self.config.gs_type == "3d_gs":
                self.elliopsoid_2d_chbox.checked = False
            else:
                self.elliopsoid_2d_chbox.checked = True

            chbox_tile_gsrender_2.add_child(self.elliopsoid_2d_chbox)

            self.normal_in_world_chbox = gui.Checkbox("Normal in World")
            self.normal_in_world_chbox.checked = True
            chbox_tile_gsrender_2.add_child(self.normal_in_world_chbox)

            chbox_tile_gsrender_3 = gui.Horiz(0.5 * em, gui.Margins(margin))

            self.normal_with_alpha_chbox = gui.Checkbox("Normal with Alpha")
            self.normal_with_alpha_chbox.checked = True
            chbox_tile_gsrender_3.add_child(self.normal_with_alpha_chbox)

            self.depth_filter_with_alpha_chbox = gui.Checkbox("Depth with Alpha")
            self.depth_filter_with_alpha_chbox.checked = True
            chbox_tile_gsrender_3.add_child(self.depth_filter_with_alpha_chbox)

            gs_vis_collapse.add_child(chbox_tile_gsrender_1)
            gs_vis_collapse.add_child(chbox_tile_gsrender_2)
            gs_vis_collapse.add_child(chbox_tile_gsrender_3)

            slider_tile = gui.Horiz(0.5 * em, gui.Margins(margin))
            slider_label = gui.Label("Gaussian Visualization Scale (0.0-1.0)")
            self.scaling_slider = gui.Slider(gui.Slider.DOUBLE)
            # Scaling Modifier to control the size of the displayed Gaussians
            self.scaling_slider.set_limits(0.001, 1.0)
            self.scaling_slider.double_value = 1.0
            slider_tile.add_child(slider_label)
            slider_tile.add_child(self.scaling_slider)
            gs_vis_collapse.add_child(slider_tile)

            slider_tile_down_rate = gui.Horiz(0.5 * em, gui.Margins(margin))
            slider_label_down_rate = gui.Label("Render Image Downsample Rate (0-3)")
            self.scaling_slider_downrate = gui.Slider(gui.Slider.INT)
            self.scaling_slider_downrate.set_limits(0, 3)
            self.scaling_slider_downrate.int_value = self.config.gs_vis_down_rate
            slider_tile_down_rate.add_child(slider_label_down_rate)
            slider_tile_down_rate.add_child(self.scaling_slider_downrate)
            gs_vis_collapse.add_child(slider_tile_down_rate)

            tab_setting.add_child(gs_vis_collapse)

        # ------------------------------------------------------------
        # Scan Options
        scan_vis_collapse = gui.CollapsableVert("Scan Options", 0.2 * em,
                                       gui.Margins(margin))
        scan_vis_collapse.set_is_open(False)

        chbox_tile_scan_color = gui.Horiz(0.5 * em, gui.Margins(margin))

        # mode 1
        self.scan_color_chbox = gui.Checkbox("Color")
        self.scan_color_chbox.checked = True
        self.scan_color_chbox.set_on_checked(self._on_scan_color_chbox)
        chbox_tile_scan_color.add_child(self.scan_color_chbox)
        
        # mode 2
        self.scan_regis_color_chbox = gui.Checkbox("Registration Weight")
        self.scan_regis_color_chbox.checked = False
        self.scan_regis_color_chbox.set_on_checked(self._on_scan_regis_color_chbox)
        chbox_tile_scan_color.add_child(self.scan_regis_color_chbox)

        # mode 3
        self.scan_height_color_chbox = gui.Checkbox("Height")
        self.scan_height_color_chbox.checked = False
        self.scan_height_color_chbox.set_on_checked(self._on_scan_height_color_chbox)
        chbox_tile_scan_color.add_child(self.scan_height_color_chbox)

        scan_vis_collapse.add_child(chbox_tile_scan_color)
        
        scan_point_size_slider_tile = gui.Horiz(0.5 * em, gui.Margins(margin))
        scan_point_size_slider_label = gui.Label("Scan point size (1-6)   ")
        self.scan_point_size_slider = gui.Slider(gui.Slider.INT)
        self.scan_point_size_slider.set_limits(1, 6)
        self.scan_point_size_slider.int_value = self.scan_render_init_size_unit
        self.scan_point_size_slider.set_on_value_changed(self._on_scan_point_size_changed)
        scan_point_size_slider_tile.add_child(scan_point_size_slider_label)
        scan_point_size_slider_tile.add_child(self.scan_point_size_slider)
        scan_vis_collapse.add_child(scan_point_size_slider_tile)

        tab_setting.add_child(scan_vis_collapse)

        # ------------------------------------------------------------
        # Neural Point Options
        neural_point_vis_collapse = gui.CollapsableVert("Neural Point Options", 0.2 * em,
                                       gui.Margins(margin))
        neural_point_vis_collapse.set_is_open(False)

        chbox_tile_neuralpoint = gui.Horiz(0.5 * em, gui.Margins(margin))

        # default mode 0: original rgb color

        # mode 1
        self.neuralpoint_geofeature_chbox = gui.Checkbox("Geo. Feature")
        self.neuralpoint_geofeature_chbox.checked = (self.neural_point_color_default_mode==1)
        self.neuralpoint_geofeature_chbox.set_on_checked(self._on_neuralpoint_geofeature_chbox)
        chbox_tile_neuralpoint.add_child(self.neuralpoint_geofeature_chbox)

        # mode 2
        self.neuralpoint_colorfeature_chbox = gui.Checkbox("Photo. Feature")
        self.neuralpoint_colorfeature_chbox.checked = (self.neural_point_color_default_mode==2)
        self.neuralpoint_colorfeature_chbox.set_on_checked(self._on_neuralpoint_colorfeature_chbox)
        chbox_tile_neuralpoint.add_child(self.neuralpoint_colorfeature_chbox)

        # mode 3
        self.neuralpoint_ts_chbox = gui.Checkbox("Time")
        self.neuralpoint_ts_chbox.checked = (self.neural_point_color_default_mode==3)
        self.neuralpoint_ts_chbox.set_on_checked(self._on_neuralpoint_ts_chbox)
        chbox_tile_neuralpoint.add_child(self.neuralpoint_ts_chbox)

        # mode 4
        self.neuralpoint_height_chbox = gui.Checkbox("Height")
        self.neuralpoint_height_chbox.checked = (self.neural_point_color_default_mode==4)
        self.neuralpoint_height_chbox.set_on_checked(self._on_neuralpoint_height_chbox)
        chbox_tile_neuralpoint.add_child(self.neuralpoint_height_chbox)

        neural_point_vis_collapse.add_child(chbox_tile_neuralpoint)

        map_point_size_slider_tile = gui.Horiz(0.5 * em, gui.Margins(margin))
        map_point_size_slider_label = gui.Label("Neural point size (1-6)")
        self.map_point_size_slider = gui.Slider(gui.Slider.INT)
        self.map_point_size_slider.set_limits(1, 6)
        self.map_point_size_slider.int_value = self.neural_points_render_init_size_unit
        self.map_point_size_slider.set_on_value_changed(self._on_neural_point_point_size_changed)
        map_point_size_slider_tile.add_child(map_point_size_slider_label)
        map_point_size_slider_tile.add_child(self.map_point_size_slider)
        neural_point_vis_collapse.add_child(map_point_size_slider_tile)

        tab_setting.add_child(neural_point_vis_collapse)

        # ------------------------------------------------------------
        # Mesh Options
        mesh_vis_collapse = gui.CollapsableVert("Mesh Options", 0.2 * em,
                                       gui.Margins(margin))
        mesh_vis_collapse.set_is_open(False)    

        chbox_tile_mesh_color = gui.Horiz(0.5 * em, gui.Margins(margin))
        
        # mode 1
        self.mesh_normal_chbox = gui.Checkbox("Normal")
        self.mesh_normal_chbox.checked = True
        self.mesh_normal_chbox.set_on_checked(self._on_mesh_normal_chbox)
        chbox_tile_mesh_color.add_child(self.mesh_normal_chbox)

        # mode 2
        self.mesh_color_chbox = gui.Checkbox("Color")
        self.mesh_color_chbox.checked = False
        self.mesh_color_chbox.set_on_checked(self._on_mesh_color_chbox)
        chbox_tile_mesh_color.add_child(self.mesh_color_chbox)
        
        # mode 3
        self.mesh_height_chbox = gui.Checkbox("Height")
        self.mesh_height_chbox.checked = False
        self.mesh_height_chbox.set_on_checked(self._on_mesh_height_chbox)
        chbox_tile_mesh_color.add_child(self.mesh_height_chbox)

        mesh_vis_collapse.add_child(chbox_tile_mesh_color)

        mesh_freq_frame_slider_tile = gui.Horiz(0.5 * em, gui.Margins(margin))
        mesh_freq_frame_slider_label = gui.Label("Mesh update per X frames (1-100)")
        self.mesh_freq_frame_slider = gui.Slider(gui.Slider.INT)
        self.mesh_freq_frame_slider.set_limits(1, 100)
        self.mesh_freq_frame_slider.int_value = self.config.mesh_freq_frame
        mesh_freq_frame_slider_tile.add_child(mesh_freq_frame_slider_label)
        mesh_freq_frame_slider_tile.add_child(self.mesh_freq_frame_slider)
        mesh_vis_collapse.add_child(mesh_freq_frame_slider_tile)
        
        mesh_mc_res_slider_tile = gui.Horiz(0.5 * em, gui.Margins(margin))
        mesh_mc_res_slider_label = gui.Label("Mesh MC resolution (1cm-100cm)")
        self.mesh_mc_res_slider = gui.Slider(gui.Slider.INT)
        self.mesh_mc_res_slider.set_limits(1, 100)
        self.mesh_mc_res_slider.int_value = int(self.config.mc_res_m * 100)
        mesh_mc_res_slider_tile.add_child(mesh_mc_res_slider_label)
        mesh_mc_res_slider_tile.add_child(self.mesh_mc_res_slider)
        mesh_vis_collapse.add_child(mesh_mc_res_slider_tile)

        mesh_min_nn_slider_tile = gui.Horiz(0.5 * em, gui.Margins(margin))
        mesh_min_nn_slider_label = gui.Label("Mesh query min neighbors (5-25)  ")
        self.mesh_min_nn_slider = gui.Slider(gui.Slider.INT)
        self.mesh_min_nn_slider.set_limits(5, 25)
        self.mesh_min_nn_slider.int_value = self.config.mesh_min_nn
        mesh_min_nn_slider_tile.add_child(mesh_min_nn_slider_label)
        mesh_min_nn_slider_tile.add_child(self.mesh_min_nn_slider)
        mesh_vis_collapse.add_child(mesh_min_nn_slider_tile)

        tab_setting.add_child(mesh_vis_collapse)

        # ------------------------------------------------------------
        # SDF Slice Options
        sdf_slice_vis_collapse = gui.CollapsableVert("SDF Slice Options", 0.2 * em,
                                       gui.Margins(margin))
        sdf_slice_vis_collapse.set_is_open(False)

        sdf_freq_frame_slider_tile = gui.Horiz(0.5 * em, gui.Margins(margin))
        sdf_freq_frame_slider_label = gui.Label("SDF slice update per X frames (1-100) ")
        self.sdf_freq_frame_slider = gui.Slider(gui.Slider.INT)
        self.sdf_freq_frame_slider.set_limits(1, 100)
        self.sdf_freq_frame_slider.int_value = self.config.sdfslice_freq_frame
        sdf_freq_frame_slider_tile.add_child(sdf_freq_frame_slider_label)
        sdf_freq_frame_slider_tile.add_child(self.sdf_freq_frame_slider)
        sdf_slice_vis_collapse.add_child(sdf_freq_frame_slider_tile)

        sdf_slice_height_slider_tile = gui.Horiz(0.5 * em, gui.Margins(margin))
        sdf_slice_height_slider_label = gui.Label("SDF slice height (m)                                 ")
        self.sdf_slice_height_slider = gui.Slider(gui.Slider.DOUBLE)
        self.sdf_slice_height_slider.set_limits(-2.0, 3.0)
        self.sdf_slice_height_slider.double_value = self.config.sdf_slice_height
        sdf_slice_height_slider_tile.add_child(sdf_slice_height_slider_label)
        sdf_slice_height_slider_tile.add_child(self.sdf_slice_height_slider)
        sdf_slice_vis_collapse.add_child(sdf_slice_height_slider_tile)

        sdf_res_slider_tile = gui.Horiz(0.5 * em, gui.Margins(margin))
        sdf_res_slider_label = gui.Label("SDF slice resolution (5cm-30cm)          ")
        self.sdf_res_slider = gui.Slider(gui.Slider.INT)
        self.sdf_res_slider.set_limits(5, 30)
        self.sdf_res_slider.int_value = int(self.config.vis_sdf_res_m * 100)
        sdf_res_slider_tile.add_child(sdf_res_slider_label)
        sdf_res_slider_tile.add_child(self.sdf_res_slider)
        sdf_slice_vis_collapse.add_child(sdf_res_slider_tile)

        tab_setting.add_child(sdf_slice_vis_collapse)

        # ------------------------------------------------------------
        # Save Options
        chbox_save_tile = gui.Horiz(0.5 * em, gui.Margins(margin))

        # screenshot buttom
        self.screenshot_btn = gui.Button("2D Screenshot")
        self.screenshot_btn.set_on_clicked(
            self._on_screenshot_btn
        )  # set the callback function
        chbox_save_tile.add_child(self.screenshot_btn)

        self.screenshot_3d_btn = gui.Button("3D Screenshot")
        self.screenshot_3d_btn.set_on_clicked(
            self._on_screenshot_3d_btn
        )  # set the callback function
        chbox_save_tile.add_child(self.screenshot_3d_btn)

        # self.save_recording_btn = gui.Button("Save Recording")
        # self.save_recording_btn.set_on_clicked(
        #     self._on_save_recording_btn
        # ) 
        # chbox_save_tile.add_child(self.save_recording_btn)
        
        tab_setting.add_child(chbox_save_tile)

        tabs0.add_tab("Setting", tab_setting)   

        self.panel.add_child(tabs0)


        ## Info Tab
        tabs = gui.TabControl()
        tab_info = gui.Vert(0, tab_margins)

        run_name_info = gui.Label("Mission: {}".format(self.config.run_name))
        tab_info.add_child(run_name_info)

        self.frame_info = gui.Label("Frame: ")
        tab_info.add_child(self.frame_info)

        self.neural_points_info = gui.Label("# Neural points: ")
        tab_info.add_child(self.neural_points_info)

        if self.config.gs_on:
            self.gaussian_info = gui.Label("# Current view Gaussians: ")
            tab_info.add_child(self.gaussian_info)

            self.freq_info = gui.Label("Render FPS: ")
            tab_info.add_child(self.freq_info)
        
        if self.config.pgo_on:
            self.loop_info = gui.Label("# Loop Closures: 0")
            tab_info.add_child(self.loop_info)

        # self.brisque_score_info = gui.Label("Current view BRISQUE score: ")
        # tab_info.add_child(self.brisque_score_info)

        self.gpu_mem_info = gui.Label("GPU Memory Usage: 0.00 GB")
        tab_info.add_child(self.gpu_mem_info)

        tabs.add_tab("Info", tab_info)
        self.panel.add_child(tabs)


        ## Input/Eval Image Tab
        
        self.in_rgb_widget = gui.ImageWidget()
        self.in_depth_widget = gui.ImageWidget()
        self.in_normal_widget = gui.ImageWidget()

        self.rendered_rgb_widget = gui.ImageWidget()
        self.rendered_normal_widget = gui.ImageWidget()
        self.rendered_depth_widget = gui.ImageWidget()
        self.rendered_depth_error_widget = gui.ImageWidget()

        if self.config.gs_on:
            tabs2 = gui.TabControl()
        
            tab_input = gui.Vert(0, tab_margins)

            self.cur_view_info = gui.Label("Camera: ")
            tab_input.add_child(self.cur_view_info)

            # self.cur_view_psnr_info = gui.Label("PSNR: ")
            # tab_input.add_child(self.cur_view_psnr_info)

            # self.cur_view_depthl1_info = gui.Label("Depth L1 (m): ")
            # tab_input.add_child(self.cur_view_depthl1_info)

            # self.cur_exposure_info = gui.Label("Exposure: ")
            # tab_input.add_child(self.cur_exposure_info)
        
            # tab_input.add_child(view_info_tile)
            
            self.rgb_input_collapse = gui.CollapsableVert("RGB Image", 0.1 * em, gui.Margins(margin))
            self.rgb_input_collapse.set_is_open(False)
            self.rgb_input_collapse.add_child(self.in_rgb_widget)
            tab_input.add_child(self.rgb_input_collapse)

            if self.show_rendered_img:
                self.rendered_collapse = gui.CollapsableVert("Rendered Image", 0.1 * em, gui.Margins(margin))
                self.rendered_collapse.set_is_open(False)
                self.rendered_collapse.add_child(self.rendered_rgb_widget)
                tab_input.add_child(self.rendered_collapse)

                self.rendered_normal_collapse = gui.CollapsableVert("Rendered Normal", 0.1 * em, gui.Margins(margin))
                self.rendered_normal_collapse.set_is_open(False)
                self.rendered_normal_collapse.add_child(self.rendered_normal_widget)
                tab_input.add_child(self.rendered_normal_collapse)

                self.rendered_depth_collapse = gui.CollapsableVert("Rendered Depth", 0.1 * em, gui.Margins(margin))
                self.rendered_depth_collapse.set_is_open(False)
                self.rendered_depth_collapse.add_child(self.rendered_depth_widget)
                tab_input.add_child(self.rendered_depth_collapse)
            
            self.depth_input_collapse = gui.CollapsableVert("Depth Projection", 0.1 * em, gui.Margins(margin))
            self.depth_input_collapse.set_is_open(False)
            self.depth_input_collapse.add_child(self.in_depth_widget)

            pixel_size_slider_tile = gui.Horiz(0.5 * em, gui.Margins(margin))
            pixel_size_slider_label = gui.Label("Projected pixel size (1-10)   ")
            self.pixel_size_slider = gui.Slider(gui.Slider.INT)
            self.pixel_size_slider.set_limits(1, 10)
            self.pixel_size_slider.int_value = 5
            pixel_size_slider_tile.add_child(pixel_size_slider_label)
            pixel_size_slider_tile.add_child(self.pixel_size_slider)
            self.depth_input_collapse.add_child(pixel_size_slider_tile)

            tab_input.add_child(self.depth_input_collapse)

            # tab_input.add_child(self.rendered_depth_error_widget)
            
            # tab_input.add_child(self.in_normal_widget)

            tabs2.add_tab("Input", tab_input)
            self.panel.add_child(tabs2)

        self.window.add_child(self.panel)


    def init_glfw(self):
        window_name = "headless rendering"

        if not glfw.init():
            exit(1)

        # check by: glxinfo | grep "OpenGL version"
        # set opengl version hint (FIXME)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 6)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)

        window = glfw.create_window(
            self.window_w, self.window_h, window_name, None, None
        ) 

        glfw.make_context_current(window)
        glfw.swap_interval(0)

        if not window:
            glfw.terminate()
            exit(1)
        return window

    def update_activated_renderer_state(self, gaus, rend_mode=-4):
        self.g_renderer.update_gaussian_data(gaus)
        self.g_renderer.sort_and_update(self.g_camera)
        self.g_renderer.set_scale_modifier(self.scaling_slider.double_value)
        self.g_renderer.set_render_mod(rend_mode) #  // > 0 render 0-ith SH dim, -1 depth, -2 bill board, -3 flat ball, -4 gaussian ball
        self.g_renderer.update_camera_pose(self.g_camera) 
        self.g_renderer.update_camera_intrin(self.g_camera)
        self.g_renderer.set_render_reso(self.g_camera.w, self.g_camera.h)

    def add_camera(self, camera, name, color=[0, 1, 0], size=0.01):
        # only the cam geometry
        # img are not added

        W2C = getWorld2View2(camera.R, camera.T)

        W2C = W2C.cpu().numpy()
        C2W = np.linalg.inv(W2C)
        frustum = create_frustum(C2W, color, size=size)
        
        if name not in self.frustum_dict.keys():
            # frustum = create_frustum(C2W, color, size=size)
            self.combo_cams.add_item(name)
        
        frustum.update_pose(C2W)
        self.frustum_dict[name] = frustum
        self.widget3d.scene.add_geometry(name, frustum.line_set, self.cur_frame_render) # add camera frame to visualizer
        
        # frustum = self.frustum_dict[name]
        # frustum.update_pose(C2W)
        # self.widget3d.scene.set_geometry_transform(name, C2W.astype(np.float64))
        self.widget3d.scene.show_geometry(name, self.cameras_chbox.checked)
        return frustum

    def add_keyframe(self, camera, name, color=[0, 1, 0], size=0.01):
        # only the cam geometry
        # img are not added

        # here keyframe are actually the train frames

        W2C = getWorld2View2(camera.R, camera.T)
        W2C = W2C.cpu().numpy()
        C2W = np.linalg.inv(W2C)
        frustum = create_frustum(C2W, color, size=size)
        if name not in self.keyframe_dict.keys():
            # frustum = create_frustum(C2W, color, size=size)
            self.combo_train_cams.add_item(name) # TODO
            self.keyframe_dict[name] = frustum
            frustum.update_pose(C2W)
            self.widget3d.scene.add_geometry(name, frustum.line_set, self.train_frame_render) # add camera frame to visualizer
        # frustum = self.keyframe_dict[name]
        # frustum.update_pose(C2W)
        # self.widget3d.scene.set_geometry_transform(name, C2W.astype(np.float64))
        self.widget3d.scene.show_geometry(name, self.keyframe_chbox.checked)
        return frustum

    def _on_layout(self, layout_context):
        contentRect = self.window.content_rect
        #self.widget3d_width_ratio = 0.6 # 0.7 # FIXME
        self.widget3d_width_ratio = self.config.visualizer_split_width_ratio
        self.widget3d_width = int(
            self.window.size.width * self.widget3d_width_ratio
        )  # 15 ems wide
        self.widget3d.frame = gui.Rect(
            contentRect.x, contentRect.y, self.widget3d_width, contentRect.height
        )
        self.panel.frame = gui.Rect(
            self.widget3d.frame.get_right(),
            contentRect.y,
            contentRect.width - self.widget3d_width,
            contentRect.height,
        )

    def _on_close(self):
        self.is_done = True

        print("[GUI] Received terminate signal")
        # clean up the pipe
        while not self.q_main2vis.empty():
            self.q_main2vis.get()
        while not self.q_vis2main.empty():
            self.q_vis2main.get()
        self.q_vis2main = None
        self.q_main2vis = None
        self.process_finished = True

        return True  # False would cancel the close

    def _on_combo_model(self, new_val, new_idx):
        model_idx = self.model_dict[new_val]
        self.global_map.active_map_idx = model_idx

    def _on_combo_cams(self, new_val, new_idx):
        frustum = self.frustum_dict[new_val]
        viewpoint = (
                    frustum.view_dir_behind
                    if self.staybehind_chbox.checked
                    else frustum.view_dir
                )
        if not self.still_chbox.checked:
            self.widget3d.look_at(viewpoint[0], viewpoint[1], viewpoint[2])

        self.update_img_show(new_val)

    def _on_combo_train_cams(self, new_val, new_idx):
        frustum = self.keyframe_dict[new_val]
        viewpoint = (
                    frustum.view_dir_behind
                    if self.staybehind_chbox.checked
                    else frustum.view_dir
                )
        if not self.still_chbox.checked:
            self.widget3d.look_at(viewpoint[0], viewpoint[1], viewpoint[2])

        self.update_img_show(new_val, from_cur_frame=False)

    # def _on_gs_chbox(self, is_checked, name=None):
    #     names = self.frustum_dict.keys() if name is None else [name]
    #     for name in names:
    #         self.widget3d.scene.show_geometry(name, is_checked)

    def _on_cameras_chbox(self, is_checked, name=None):
        names = self.frustum_dict.keys() if name is None else [name]
        for name in names:
            self.widget3d.scene.show_geometry(name, is_checked)

    def _on_keyframes_chbox(self, is_checked, name=None):
        names = self.keyframe_dict.keys() if name is None else [name]
        for name in names:
            self.widget3d.scene.show_geometry(name, is_checked)

    def _on_cad_chbox(self, is_checked):
        if is_checked:
            self.widget3d.scene.remove_geometry(self.cad_name)
            self.widget3d.scene.add_geometry(self.cad_name, self.sensor_cad, self.cad_render)
        else:
            self.widget3d.scene.remove_geometry(self.cad_name)
    
    def _on_neural_point_chbox(self, is_checked):
        self.widget3d.scene.show_geometry(self.neural_point_name, is_checked)

    def _on_invalid_neural_point_chbox(self, is_checked):
        self.widget3d.scene.show_geometry(self.invalid_neural_point_name, is_checked)

    def _on_mesh_chbox(self, is_checked):
        self.widget3d.scene.show_geometry(self.mesh_name, is_checked)

    def _on_scan_chbox(self, is_checked):
        self.widget3d.scene.show_geometry(self.scan_name, is_checked)

    def _on_rendered_scan_chbox(self, is_checked):
        self.widget3d.scene.show_geometry(self.rendered_scan_name, is_checked)
    
    def _on_sdf_pool_chbox(self, is_checked):
        self.widget3d.scene.show_geometry(self.sdf_pool_name, is_checked)
       
    # sdf slice
    def _on_sdf_chbox(self, is_checked):
        self.widget3d.scene.show_geometry(self.sdf_name, is_checked)

    def _on_gt_traj_chbox(self, is_checked):
        if is_checked:
            self.widget3d.scene.remove_geometry(self.gt_traj_name)
            self.widget3d.scene.add_geometry(self.gt_traj_name, self.gt_traj, self.traj_render)
        else:
            self.widget3d.scene.remove_geometry(self.gt_traj_name)

    def _on_slam_traj_chbox(self, is_checked):
        if is_checked:
            self.widget3d.scene.remove_geometry(self.slam_traj_name)
            self.widget3d.scene.add_geometry(self.slam_traj_name, self.slam_traj, self.traj_render)
        else:
            self.widget3d.scene.remove_geometry(self.slam_traj_name)

    def _on_odom_traj_chbox(self, is_checked):
        if is_checked:
            self.widget3d.scene.remove_geometry(self.odom_traj_name)
            self.widget3d.scene.add_geometry(self.odom_traj_name, self.odom_traj, self.traj_render)
        else:
            self.widget3d.scene.remove_geometry(self.odom_traj_name)

    def _on_range_circle_chbox(self, is_checked):
        if is_checked:
            self.widget3d.scene.remove_geometry(self.range_circle_name)
            self.widget3d.scene.add_geometry(self.range_circle_name, self.range_circle, self.ring_render)
        else:
            self.widget3d.scene.remove_geometry(self.range_circle_name)

    def _on_scan_point_size_changed(self, value):
        self.scan_render.point_size = value * self.window.scaling

        self.widget3d.scene.remove_geometry(self.scan_name)
        self.widget3d.scene.add_geometry(self.scan_name, self.scan, self.scan_render)
        self.widget3d.scene.show_geometry(self.scan_name, self.scan_chbox.checked)

    def _on_neural_point_point_size_changed(self, value):
        self.neural_points_render.point_size = value * self.window.scaling

        self.widget3d.scene.remove_geometry(self.neural_point_name)
        self.widget3d.scene.add_geometry(self.neural_point_name, self.neural_points, self.neural_points_render)
        self.widget3d.scene.show_geometry(self.neural_point_name, self.neural_point_chbox.checked)

        self.widget3d.scene.remove_geometry(self.invalid_neural_point_name)
        self.widget3d.scene.add_geometry(self.invalid_neural_point_name, self.invalid_neural_points, self.neural_points_render)
        self.widget3d.scene.show_geometry(self.invalid_neural_point_name, self.invalid_neural_point_chbox.checked)

    # only one can be selected at the same time
    def _on_ellipsoid_chbox(self, is_checked):
        if is_checked:
            self.depth_chbox.checked = False
            self.normal_chbox.checked = False
            self.d2n_chbox.checked = False
            self.opacity_chbox.checked = False

    def _on_depth_chbox(self, is_checked):
        if is_checked:
            self.ellipsoid_chbox.checked = False
            self.normal_chbox.checked = False
            self.d2n_chbox.checked = False
            self.opacity_chbox.checked = False

    def _on_normal_chbox(self, is_checked):
        if is_checked:
            self.ellipsoid_chbox.checked = False
            self.depth_chbox.checked = False
            self.d2n_chbox.checked = False
            self.opacity_chbox.checked = False

            # self.cad_render.shader = "normals"

    def _on_d2n_chbox(self, is_checked):
        if is_checked:
            self.ellipsoid_chbox.checked = False
            self.normal_chbox.checked = False
            self.depth_chbox.checked = False
            self.opacity_chbox.checked = False

    def _on_opacity_chbox(self, is_checked):
        if is_checked:
            self.ellipsoid_chbox.checked = False
            self.normal_chbox.checked = False
            self.d2n_chbox.checked = False
            self.depth_chbox.checked = False

    def _on_scan_color_chbox(self, is_checked):
        if is_checked:
            self.scan_height_color_chbox.checked = False
            self.scan_regis_color_chbox.checked = False
        self.visualize_scan()
    
    def _on_scan_regis_color_chbox(self, is_checked):
        if is_checked:
            self.scan_height_color_chbox.checked = False
            self.scan_color_chbox.checked = False
        self.visualize_scan()

    def _on_scan_height_color_chbox(self, is_checked):
        if is_checked:
            self.scan_color_chbox.checked = False
            self.scan_regis_color_chbox.checked = False
        self.visualize_scan()

    def _on_neuralpoint_geofeature_chbox(self, is_checked):
        if is_checked:
            self.neuralpoint_colorfeature_chbox.checked = False
            self.neuralpoint_height_chbox.checked = False
            self.neuralpoint_ts_chbox.checked = False
        self.visualize_neural_points()

    def _on_neuralpoint_colorfeature_chbox(self, is_checked):
        if is_checked:
            self.neuralpoint_geofeature_chbox.checked = False
            self.neuralpoint_height_chbox.checked = False
            self.neuralpoint_ts_chbox.checked = False
        self.visualize_neural_points()

    def _on_neuralpoint_ts_chbox(self, is_checked):
        if is_checked:
            self.neuralpoint_geofeature_chbox.checked = False
            self.neuralpoint_height_chbox.checked = False
            self.neuralpoint_colorfeature_chbox.checked = False
        self.visualize_neural_points()

    def _on_neuralpoint_height_chbox(self, is_checked):
        if is_checked:
            self.neuralpoint_geofeature_chbox.checked = False
            self.neuralpoint_ts_chbox.checked = False
            self.neuralpoint_colorfeature_chbox.checked = False
        self.visualize_neural_points()

    def _on_mesh_normal_chbox(self, is_checked):
        if is_checked:
            self.mesh_render.shader = "normals"
            self.mesh_color_chbox.checked = False
            self.mesh_height_chbox.checked = False
        self.visualize_mesh()

    def _on_mesh_color_chbox(self, is_checked):
        if is_checked:
            self.mesh_render.shader = "defaultLit"
            self.mesh_normal_chbox.checked = False
            self.mesh_height_chbox.checked = False
        self.visualize_mesh()

    def _on_mesh_height_chbox(self, is_checked):
        if is_checked:
            self.mesh_render.shader = "defaultLit"
            self.mesh_normal_chbox.checked = False
            self.mesh_color_chbox.checked = False
        self.visualize_mesh()

    def _on_sky_chbox(self, is_checked):
        self.widget3d.scene.show_skybox(is_checked)

    # def _on_backface_chbox(self, is_checked):
    #     self.widget3d.enable_back_face_culling(is_checked) 
    #     # self.mesh_render.show_back_face = is_checked

    def _on_slam_slider(self, is_on):
        if is_on:
            print("[GUI] SLAM resumed")
        else:
            print("[GUI] SLAM paused")
    
    def _on_vis_slider(self, is_on):
        if is_on:
            print("[GUI] Visualization resumed")
        else:
            print("[GUI] Visualization paused")

    def _on_screenshot_btn(self):
        dt = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        filename = os.path.join(self.save_dir_2d_screenshots, f"{dt}-gui.png")
        height = self.window.size.height
        width = self.widget3d_width
        app = o3d.visualization.gui.Application.instance
        img = np.asarray(app.render_to_image(self.widget3d.scene, width, height))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(filename, img)

        print("[GUI] 2D Screenshot save at {}".format(filename))


    def _on_screenshot_3d_btn(self):

        if self.sdf_pool.has_points() and self.sdf_pool_chbox.checked:
            data_pool_pc_name = str(self.cur_frame_id) + "_training_sdf_pool"
            data_pool_pc_path = os.path.join(self.save_dir_3d_screenshots, data_pool_pc_name)
            o3d.io.write_point_cloud(data_pool_pc_path, self.sdf_pool)
            print("[GUI] Output current SDF training pool to: ", data_pool_pc_path)
        if self.scan.has_points() and self.scan_chbox.checked:
            scan_pc_name = str(self.cur_frame_id) + "_scan"
            scan_pc_name += ".ply"
            scan_pc_path = os.path.join(self.save_dir_3d_screenshots, scan_pc_name)
            o3d.io.write_point_cloud(scan_pc_path, self.scan)
            print("[GUI] Output current scan to: ", scan_pc_path)
        if self.neural_points.has_points() and self.neural_point_chbox.checked:
            neural_point_name = str(self.cur_frame_id) + "_neural_point_map"
            if self.local_map_chbox.checked:
                neural_point_name += "_local"
            neural_point_name += ".ply"
            neural_point_path = os.path.join(self.save_dir_3d_screenshots, neural_point_name)
            o3d.io.write_point_cloud(neural_point_path, self.neural_points)
            print("[GUI] Output current neural point map to: ", neural_point_path)
        if self.sdf_slice.has_points() and self.sdf_chbox.checked:
            sdf_slice_name = str(self.cur_frame_id) + "_sdf_slice"
            sdf_slice_name += ".ply"
            sdf_slice_path = os.path.join(self.save_dir_3d_screenshots, sdf_slice_name)
            o3d.io.write_point_cloud(sdf_slice_path, self.sdf_slice)
            print("[GUI] Output current SDF slice to: ", sdf_slice_path)
        if self.mesh.has_triangles() and self.mesh_chbox.checked:
            mesh_name = str(self.cur_frame_id) + "_mesh_vis"
            if self.local_map_chbox.checked:
                mesh_name += "_local"
            mesh_name += ".ply"
            mesh_path = os.path.join(self.save_dir_3d_screenshots, mesh_name)
            o3d.io.write_triangle_mesh(mesh_path, self.mesh)
            print("[GUI] Output current mesh to: ", mesh_path)
        if self.sensor_cad.has_triangles() and self.cad_chbox.checked:
            cad_name = str(self.cur_frame_id) + "_sensor_vis"
            cad_name += ".ply"
            cad_path = os.path.join(self.save_dir_3d_screenshots, cad_name)
            o3d.io.write_triangle_mesh(cad_path, self.sensor_cad)
            print("[GUI] Output current sensor model to: ", cad_path)

    def _on_reset_view_btn(self):
        self.center_bev()
        self.fly_chbox.checked = False
        self.widget3d.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA_SPHERE)
    
    def _on_save_view_btn(self):
        save_view_file_name = 'saved_view_{}.pkl'.format(self.combo_preset_cams.selected_text)
        save_view_file_path = os.path.join(self.view_save_base_path, save_view_file_name)
        if self.save_view(save_view_file_path):
            print("[GUI] Camera view {} saved".format(self.combo_preset_cams.selected_text))
    
    def _on_load_view_btn(self):
        load_view_file_name = 'saved_view_{}.pkl'.format(self.combo_preset_cams.selected_text)
        load_view_file_path = os.path.join(self.view_save_base_path, load_view_file_name)
        if self.load_view(load_view_file_path):
            print("[GUI] Camera view {} loaded".format(self.combo_preset_cams.selected_text))

    def _on_save_recording_btn(self):
        dt = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        save_dir = self.save_path / "recording" / dt
        save_dir.mkdir(parents=True, exist_ok=True)
        # create the filename
        filename = save_dir / "recorded_pose"

        if len(self.recorded_poses) > 0:
            write_kitti_format_poses(filename, self.recorded_poses)
            print("[GUI] Recorded poses save at {}.txt".format(filename))

        self.recorded_poses = [] # clear the poses


    def _set_mouse_mode(self, is_on):
        if is_on:
            self.widget3d.set_view_controls(gui.SceneWidget.Controls.FLY)
        else:
            self.widget3d.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA_SPHERE)

    @staticmethod
    def resize_img(img, width):
        height = int(width * img.shape[0] / img.shape[1])
        return cv2.resize(img, (width, height))

    # disable this now
    # def add_ids(self):
    #     indices = (
    #         torch.unique(self.cur_data_packet.unique_kfIDs).cpu().numpy().astype(int)
    #     ).tolist()
    #     for idx in indices:
    #         if idx in self.gaussian_id_dict.keys():
    #             continue

    #         self.gaussian_id_dict[idx] = 0
    #         self.combo_gaussian_id.add_item(str(idx))

    def save_view(self, fname='.saved_view.pkl'):
        try:
            model_matrix = np.asarray(self.widget3d.scene.camera.get_model_matrix())
            extrinsic = model_matrix_to_extrinsic_matrix(model_matrix)
            height, width = int(self.window.size.height), int(self.widget3d_width)
            intrinsic = create_camera_intrinsic_from_size(width, height)
            saved_view = dict(extrinsic=extrinsic, intrinsic=intrinsic, width=width, height=height)
            with open(fname, 'wb') as pickle_file:
                dump(saved_view, pickle_file)
            return True
        except Exception as e:
            print("[GUI]", e)
            return False

    def load_view(self, fname=".saved_view.pkl"):
        try:
            with open(fname, 'rb') as pickle_file:
                saved_view = load(pickle_file)
            self.widget3d.setup_camera(saved_view['intrinsic'], saved_view['extrinsic'], saved_view['width'], saved_view['height'], self.widget3d.scene.bounding_box)
            # Looks like the ground plane gets messed up, no idea how to fix
            return True
        except Exception as e:
            print("[GUI] Can't find file", e)
            return False
        
    def send_data(self):
        packet = ControlPacket()
        packet.flag_pause = not self.slider_slam.is_on
        packet.flag_vis = self.slider_render.is_on
        packet.flag_source = self.scan_regis_color_chbox.checked
        packet.flag_mesh = self.mesh_chbox.checked
        packet.flag_sdf = self.sdf_chbox.checked
        packet.flag_global = not self.local_map_chbox.checked
        packet.mc_res_m = self.mesh_mc_res_slider.int_value / 100.0
        packet.mesh_min_nn = self.mesh_min_nn_slider.int_value
        packet.mesh_freq_frame = self.mesh_freq_frame_slider.int_value
        packet.sdf_freq_frame = self.sdf_freq_frame_slider.int_value
        packet.sdf_slice_height = self.sdf_slice_height_slider.double_value
        packet.sdf_res_m = self.sdf_res_slider.int_value / 100.0
        packet.cur_frame_id = self.cur_frame_id

        self.q_vis2main.put(packet)
    

    def receive_data(self, q):
        if q is None:
            return

        data_packet = get_latest_queue(q)

        if data_packet is None:
            return

        # if data_packet.frame_id != self.cur_frame_id:
            # only update with new data (once)

        if True:    
            self.cur_frame_id = data_packet.frame_id

            self.cur_data_packet = data_packet

            if data_packet.frame_id is not None:
                self.frame_info.text = "Frame: {}".format(data_packet.frame_id)
                    
            self.visualize_neural_points()
               
            if data_packet.has_sorrounding_points:
                # print("Spawn surrounding points")

                cur_center_position = data_packet.sorrounding_neural_points_data["center"]
                
                # spawn gaussians for the sorrounding map
                # self.cur_base_gaussians are stored in GPU, might take some memory

                self.cur_base_gaussians = spawn_gaussians(data_packet.sorrounding_neural_points_data, 
                    self.decoders, None, cur_center_position,
                    dist_concat_on=self.config.dist_concat_on, 
                    view_concat_on=self.config.view_concat_on, 
                    scale_filter_on=False,  # for visualization, do not filter (TODO)
                    z_far=self.config.sorrounding_map_radius,
                    learn_color_residual=self.config.learn_color_residual,
                    gs_type=self.config.gs_type,
                    displacement_range_ratio=self.config.displacement_range_ratio,
                    max_scale_ratio=self.config.max_scale_ratio,
                    unit_scale_ratio=self.config.unit_scale_ratio)

            # load cameras
            if data_packet.current_frames is not None and len(data_packet.cam_list)>0: # as Camera class
                
                for cam_name in list(self.frustum_dict.keys()): 
                    self.widget3d.scene.remove_geometry(cam_name)

                for cam in data_packet.cam_list:
                    frustum = self.add_camera(
                        data_packet.current_frames[cam], name=cam, color=[0, 1, 0], size=self.frustum_size
                    )
                    # print("Cam added")
                if self.followcam_chbox.checked:
                    selected_cam = self.combo_cams.selected_text
                    selected_frustum = self.frustum_dict[selected_cam]
                    viewpoint = (
                        selected_frustum.view_dir_behind
                        if self.staybehind_chbox.checked
                        else selected_frustum.view_dir
                    )
                    if not self.still_chbox.checked:
                        self.widget3d.look_at(viewpoint[0], viewpoint[1], viewpoint[2])

                    # show rgb / depth imgs (also the rendered rgb / depth error, etc.)
                    self.update_img_show(selected_cam)                           

            if data_packet.keyframes is not None: # as Camera class
                
                # remove old stuff from last frame
                for keyframe_name in list(self.keyframe_dict.keys()): 
                    self.widget3d.scene.remove_geometry(keyframe_name)

                self.keyframe_dict = {} # set back to empty
                self.combo_train_cams.clear_items() # set back to empty

                # add new stuff from this frame
                for cam in data_packet.keyframe_list:
                    cur_keyframe =  data_packet.keyframes[cam]
                    if cur_keyframe.in_long_term_memory:
                        frustum_color = [0.5, 0.5, 0]
                    else:
                        frustum_color = [1, 1, 0]
                    frustum = self.add_keyframe(
                        cur_keyframe, name=cur_keyframe.uid, color=frustum_color, size=self.frustum_size
                    ) 

            if data_packet.gpu_mem_usage_gb is not None:
                self.gpu_mem_info.text = f"GPU Memory Usage: {data_packet.gpu_mem_usage_gb:.2f} GB"

            self.visualize_scan(data_packet)

            self.visualize_mesh(data_packet)

            self.visualize_sdf_slice(data_packet)

            self.visualize_sdf_pool(data_packet)

            self.visualize_rendered_scan(data_packet)

            if data_packet.gt_poses is not None:
                gt_position_np = data_packet.gt_poses[:, :3, 3]
                if gt_position_np.shape[0] > 1:
                    self.gt_traj.points = o3d.utility.Vector3dVector(gt_position_np)
                    gt_edges = np.array([[i, i + 1] for i in range(gt_position_np.shape[0] - 1)])
                    self.gt_traj.lines = o3d.utility.Vector2iVector(gt_edges)
                    self.gt_traj.paint_uniform_color(BLACK) # Black
                
                if self.gt_traj_chbox.checked:
                    self.widget3d.scene.remove_geometry(self.gt_traj_name)
                    self.widget3d.scene.add_geometry(self.gt_traj_name, self.gt_traj, self.traj_render)

                if data_packet.slam_poses is None:
                    
                    self.sensor_cad = copy.deepcopy(self.sensor_cad_origin)
                    self.sensor_cad.transform(data_packet.gt_poses[-1])

                    self.range_circle = copy.deepcopy(self.range_circle_origin)
                    self.range_circle.transform(data_packet.gt_poses[-1])  
                    
                    if self.cad_chbox.checked:
                        self.widget3d.scene.remove_geometry(self.cad_name)
                        self.widget3d.scene.add_geometry(self.cad_name, self.sensor_cad, self.cad_render)

                    if self.range_circle_chbox.checked: 
                        self.widget3d.scene.remove_geometry(self.range_circle_name)
                        self.widget3d.scene.add_geometry(self.range_circle_name, self.range_circle, self.ring_render)

            if data_packet.slam_poses is not None:
                
                slam_position_np = data_packet.slam_poses[:, :3, 3]
                if slam_position_np.shape[0] > 1:
                    self.slam_traj.points = o3d.utility.Vector3dVector(slam_position_np)
                    slam_edges = np.array([[i, i + 1] for i in range(slam_position_np.shape[0] - 1)])
                    self.slam_traj.lines = o3d.utility.Vector2iVector(slam_edges)
                    self.slam_traj.paint_uniform_color(RED)

                if self.slam_traj_chbox.checked:
                    self.widget3d.scene.remove_geometry(self.slam_traj_name)
                    self.widget3d.scene.add_geometry(self.slam_traj_name, self.slam_traj, self.traj_render)
                
                self.sensor_cad = copy.deepcopy(self.sensor_cad_origin)
                self.sensor_cad.transform(data_packet.slam_poses[-1])

                self.range_circle = copy.deepcopy(self.range_circle_origin)
                self.range_circle.transform(data_packet.slam_poses[-1])

                if self.cad_chbox.checked:
                    self.widget3d.scene.remove_geometry(self.cad_name)
                    self.widget3d.scene.add_geometry(self.cad_name, self.sensor_cad, self.cad_render)
                
                if self.range_circle_chbox.checked: 
                    self.widget3d.scene.remove_geometry(self.range_circle_name)
                    self.widget3d.scene.add_geometry(self.range_circle_name, self.range_circle, self.ring_render)
                
                if data_packet.loop_edges is not None:
                    loop_count = len(data_packet.loop_edges)
                    self.loop_info.text = f"# Loop Closures: {loop_count}"

                    if loop_count > 0:
                        self.loop_edges.points = o3d.utility.Vector3dVector(slam_position_np)
                        self.loop_edges.lines = o3d.utility.Vector2iVector(np.array(data_packet.loop_edges))
                        self.loop_edges.paint_uniform_color(GREEN)

                        # if self.ego_chbox.checked:
                        #     self.loop_edges.transform(np.linalg.inv(self.cur_pose))

                        self.widget3d.scene.remove_geometry(self.loop_edges_name)
                        self.widget3d.scene.add_geometry(self.loop_edges_name, self.loop_edges, self.traj_render)
                        self.widget3d.scene.show_geometry(self.loop_edges_name, self.loop_edges_chbox.checked)
        
            if data_packet.odom_poses is not None:
            
                odom_position_np = data_packet.odom_poses[:, :3, 3]
                if odom_position_np.shape[0] > 1:
                    self.odom_traj.points = o3d.utility.Vector3dVector(odom_position_np)
                    odom_edges = np.array([[i, i + 1] for i in range(odom_position_np.shape[0] - 1)])
                    self.odom_traj.lines = o3d.utility.Vector2iVector(odom_edges)
                    self.odom_traj.paint_uniform_color(BLUE)

                if self.odom_traj_chbox.checked:
                    self.widget3d.scene.remove_geometry(self.odom_traj_name)
                    self.widget3d.scene.add_geometry(self.odom_traj_name, self.odom_traj, self.traj_render)

        # set up inital camera # no camera
        if not self.init:
            self.center_bev()

        self.init = True

        if data_packet.finish:
            print("[GUI] Received terminate signal")
            # clean up the pipe
            while not self.q_main2vis.empty():
                self.q_main2vis.get()
            while not self.q_vis2main.empty():
                self.q_vis2main.get()
            self.q_vis2main = None
            self.q_main2vis = None
            self.process_finished = True

    def center_bev(self):
        # set the view point to BEV of the current 3d objects
        bounds = self.widget3d.scene.bounding_box
        self.widget3d.setup_camera(45, bounds, bounds.get_center())  # field of view, bound, center
    
    def center_bev_local(self):
        bounds = self.range_circle.get_axis_aligned_bounding_box()
        self.widget3d.setup_camera(45, bounds, bounds.get_center())  # field of view, bound, center

    def update_img_show(self, cam_name, 
                        from_cur_frame: bool = True,
                        online_eval_on: bool = True, 
                        show_depth_error: bool = False,
                        alpha_foreground: float = 0.7):
        
        if not self.config.gs_on:
            return

        if self.cur_data_packet.current_frames is None:
            return 

        if cam_name in list(self.cur_data_packet.gtcolor.keys()):
            selected_gtcolor = self.cur_data_packet.gtcolor[cam_name]
            selected_gtdepth = self.cur_data_packet.gtdepth[cam_name]
            selected_gtnormal = self.cur_data_packet.gtnormal[cam_name]
        else:
            selected_gtcolor = selected_gtdepth = selected_gtnormal = None

        if selected_gtcolor is not None:
            rgb = torch.clamp(selected_gtcolor, min=0, max=1.0) 
            rgb_np = (rgb * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy()
            rgb_o3d = o3d.geometry.Image(rgb_np)
            self.in_rgb_widget.update_image(rgb_o3d)
            if not self.init:
                self.rgb_input_collapse.set_is_open(True)

        if selected_gtdepth is not None:
            depth_np = selected_gtdepth.contiguous().cpu().numpy() 
            depth_color_np = (colorize_depth_maps(depth_np, 0.1, self.config.max_range*0.8)*255.0).astype(np.uint8)
            depth_color_np = np.transpose(depth_color_np[0], (1, 2, 0))

            if self.is_rgbd:
                depth_color_np = self.overlaid_img(depth_color_np, rgb_np, alpha_foreground) 
            
            else:
                # find valid u,v
                valid_indices = np.argwhere(depth_np[0] > 0.1)
                v_coords, u_coords = valid_indices[:, 0], valid_indices[:, 1]
            
                uv_coords = np.stack((u_coords, v_coords), axis=1)

                overlay_image = rgb_np.copy()
                depth_projection_radius = self.pixel_size_slider.int_value
                
                for point in uv_coords:
                    u, v = point.astype(int)  # Convert coordinates to integer
                    depth_color = depth_color_np[v, u]
                    depth_color_tuple = (int(depth_color[0]), int(depth_color[1]), int(depth_color[2]))
                    cv2.circle(overlay_image, (u, v), radius=depth_projection_radius, color=depth_color_tuple, thickness=-1)

                overlay_image = cv2.addWeighted(overlay_image, alpha_foreground, rgb_np, 1 - alpha_foreground, 0)

                depth_color_np = np.array(overlay_image)
            
            depth_color_np = np.ascontiguousarray(depth_color_np)
            depth_color_o3d = o3d.geometry.Image(depth_color_np)
            self.in_depth_widget.update_image(depth_color_o3d)

            if not self.init:
                self.depth_input_collapse.set_is_open(True)

        if selected_gtnormal is not None:
            normal = selected_gtnormal.contiguous().cpu().numpy() 
            normal_color = 0.5 - normal * 0.5
            normal_color = np.transpose(normal_color, (1, 2, 0))
            normal_color = np.ascontiguousarray(normal_color)
            normal_color_o3d = o3d.geometry.Image(normal_color)
            self.in_normal_widget.update_image(normal_color_o3d)

        cur_psnr = None
        cur_depthl1 = None

        if from_cur_frame:
            if cam_name in list(self.cur_data_packet.current_frames.keys()):
                cur_frame_cam: CamImage = self.cur_data_packet.current_frames[cam_name]
            else:
                return
        else:
            cur_frame_cam: CamImage = self.cur_data_packet.keyframes[cam_name]

        down_rate_used = max(self.config.gs_vis_down_rate, cur_frame_cam.cur_best_level)

        if online_eval_on:

            with torch.no_grad():
                render_results = render(cur_frame_cam, 
                    None, self.cur_data_packet.neural_points_data, 
                    self.decoders, self.cur_base_gaussians, self.background,
                    scaling_modifier=self.scaling_slider.double_value, 
                    down_rate=down_rate_used,
                    dist_concat_on=self.config.dist_concat_on, 
                    view_concat_on=self.config.view_concat_on, 
                    correct_exposure=self.config.exposure_correction_on,
                    correct_exposure_affine=self.config.affine_exposure_correction,
                    learn_color_residual=self.config.learn_color_residual,
                    front_only_on=(not self.backface_chbox.checked),
                    d2n_on=False,
                    gs_type=self.config.gs_type,
                    displacement_range_ratio=self.config.displacement_range_ratio,
                    max_scale_ratio=self.config.max_scale_ratio,
                    unit_scale_ratio=self.config.unit_scale_ratio)
                    

            if render_results is not None:
                
                rendered_rgb = torch.clamp(render_results["render"], 0.0, 1.0)

                rendered_rgb_np = (
                    (rendered_rgb * 255)
                    .byte()
                    .permute(1, 2, 0)
                    .contiguous()
                    .cpu()
                    .numpy()
                )
                rendered_rgb_o3d = o3d.geometry.Image(rendered_rgb_np)
                self.rendered_rgb_widget.update_image(rendered_rgb_o3d)

                # cur_psnr = psnr(rendered_rgb, rgb).mean().item()

                eval_depth_max = self.config.max_range * 0.8
                eval_depth_min = self.config.min_range
                diff_depth_max_show = eval_depth_max * 0.05 # unit: m

                rendered_depth = render_results["surf_depth"]
                
                rendered_normal = render_results["rend_normal"]
                if rendered_normal is not None:
                    rendered_normal = torch.nn.functional.normalize(rendered_normal, dim=0) # normalize to norm==1 # don't do this, for small opacity region, we just downweight its normal
                    rendered_normal_color = 0.5 * (1 - rendered_normal)
                    rendered_normal_color_np = np.ascontiguousarray((rendered_normal_color.permute(1,2,0).detach().cpu().numpy() * 255.0).astype(np.uint8))
                    rendered_normal_o3d = o3d.geometry.Image(rendered_normal_color_np)
                    self.rendered_normal_widget.update_image(rendered_normal_o3d)

                cur_gt_depth = selected_gtdepth
                if rendered_depth is not None and cur_gt_depth is not None:
                    
                    depth_valid_mask = (rendered_depth > eval_depth_min) & (cur_gt_depth > eval_depth_min) & (cur_gt_depth < eval_depth_max) & (rendered_depth < eval_depth_max)
                    if render_results["rend_alpha"] is not None:
                        depth_valid_mask = depth_valid_mask & (render_results["rend_alpha"] > self.config.eval_depth_min_accu_alpha)
                    
                    depth_color_np = (colorize_depth_maps(rendered_depth.detach().cpu().numpy(), 0.1, self.config.max_range*0.8)[0]*255.0).astype(np.uint8) # 1, 3, H, W 
                    depth_color_np = np.ascontiguousarray(np.transpose(depth_color_np, (1, 2, 0))) # H, W, 3
                    depth_color_o3d = o3d.geometry.Image(depth_color_np)
                    self.rendered_depth_widget.update_image(depth_color_o3d)

                    diff_depth = torch.abs(rendered_depth - cur_gt_depth)
                    diff_depth_masked = diff_depth[depth_valid_mask].detach().cpu().numpy()
                    cur_depthl1 = np.mean(diff_depth_masked)

                    if show_depth_error: # TODO: use scatters

                        diff_depth[~depth_valid_mask] = 0.0
                        diff_depth_np = diff_depth.detach().cpu().numpy()
                        
                        diff_depth_color_np = (colorize_depth_maps(diff_depth_np, 0.0, diff_depth_max_show)[0]*255.0).astype(np.uint8)
                        diff_depth_color_np = np.transpose(diff_depth_color_np, (1, 2, 0)) # H, W, 3
                        diff_depth_color_np = np.ascontiguousarray(diff_depth_color_np)

                        if selected_gtcolor is not None:
                            diff_depth_color_np = self.overlaid_img(diff_depth_color_np, rgb_np) 

                        diff_depth_o3d = o3d.geometry.Image(diff_depth_color_np)

                        self.rendered_depth_error_widget.update_image(diff_depth_o3d)
        
        if cur_frame_cam.train_view:
            train_view_info = "train view"
        else:
            train_view_info = "test view"

        self.cur_view_info.text = "Camera: {} [{}]".format(cur_frame_cam.uid, train_view_info)
        # self.cur_exposure_info.text = "Exposure: ({:.3f} , {:.3f})".format(cur_frame_cam.exposure_a.item(), cur_frame_cam.exposure_b.item())
        
        # if cur_psnr is not None:
        #     self.cur_view_psnr_info.text = "PSNR: {:.3f}".format(cur_psnr)
        
        # if cur_depthl1 is not None:
        #     self.cur_view_depthl1_info.text = "Depth L1 (m): {:.3f}".format(cur_depthl1)

    
    def overlaid_img(self, foreground_img_np, background_img_np, alpha_foreground: float = 0.7):
        overlaid_img_np = (1 - alpha_foreground) * background_img_np + alpha_foreground * foreground_img_np
        overlaid_img_np = overlaid_img_np.astype(np.uint8)

        return overlaid_img_np

    @staticmethod
    def vfov_to_hfov(vfov_deg, height, width):
        # http://paulbourke.net/miscellaneous/lens/
        return np.rad2deg(
            2 * np.arctan(width * np.tan(np.deg2rad(vfov_deg) / 2) / height)
        )

    def get_current_cam(self):

        cur_view_mat = self.widget3d.scene.camera.get_view_matrix()

        has_nan = np.isnan(cur_view_mat).any()
        if has_nan:
            cur_view_mat = np.eye(4)

        w2c = cv_gl @ cur_view_mat # should not be NaN

        # print(w2c)

        image_gui = torch.zeros(
            (1, int(self.window.size.height), int(self.widget3d_width))
        )
        vfov_deg = self.widget3d.scene.camera.get_field_of_view() 

        hfov_deg = self.vfov_to_hfov(vfov_deg, image_gui.shape[1], image_gui.shape[2])
        FoVx = np.deg2rad(hfov_deg)
        FoVy = np.deg2rad(vfov_deg)
        fx = fov2focal(FoVx, image_gui.shape[2])
        fy = fov2focal(FoVy, image_gui.shape[1])
        H = image_gui.shape[1]
        W = image_gui.shape[2]
        cx = W // 2
        cy = H // 2
        T = torch.from_numpy(w2c) # T_cw

        K_mat = np.eye(3)
        K_mat[0,0] = fx
        K_mat[1,1] = fy
        K_mat[0,2] = cx
        K_mat[1,2] = cy

        current_cam = CamImage(-1, None, K_mat, self.config.min_range*0.2, self.config.local_map_radius*1.1,
            img_width=W, img_height=H, cam_pose=torch.linalg.inv(T)) # T_wc
                                                        
        return current_cam
    
    def visualize_neural_points(self, data_packet = None):
        if data_packet is None:
            data_packet = self.cur_data_packet
        
        if data_packet is None:
            return
        
        if (self.neural_point_chbox.checked or (not self.init)) and data_packet.has_neural_points:

            neural_point_vis_down_rate = self.neural_point_vis_down_rate

            dict_keys = list(data_packet.neural_points_data.keys())

            if "count" in dict_keys and "local_count" in dict_keys and "map_memory_mb" in dict_keys:
                self.neural_points_info.text = "# Neural points: {} (local {}) [PINGS Map size: {:.1f} MB]".format(
                        data_packet.neural_points_data["count"],
                        data_packet.neural_points_data["local_count"],
                        data_packet.neural_points_data["map_memory_mb"]
                    )

            local_mask = None
            # global map is being loaded here
            if "local_mask" in dict_keys:
                local_mask = data_packet.neural_points_data["local_mask"]
                # check if we need to downsample the global map a bit for fast visualization
            
            point_count = data_packet.neural_points_data["count"]
            if point_count > 300000 and not self.local_map_chbox.checked:
                neural_point_vis_down_rate = max(neural_point_vis_down_rate, find_closest_prime(point_count // 200000))

            if local_mask is not None and self.local_map_chbox.checked:
                neural_point_position = data_packet.neural_points_data["position"][local_mask]
            else:
                neural_point_position = data_packet.neural_points_data["position"]

            neural_point_position_np = neural_point_position[::neural_point_vis_down_rate, :].detach().cpu().numpy()
                    
            neural_point_colors_np = None
            
            if "color_pca_geo" in dict_keys and self.neuralpoint_geofeature_chbox.checked:
                if local_mask is not None and self.local_map_chbox.checked:
                    neural_point_colors = data_packet.neural_points_data["color_pca_geo"][local_mask]
                else:
                    neural_point_colors = data_packet.neural_points_data["color_pca_geo"]
                neural_point_colors_np = neural_point_colors[::neural_point_vis_down_rate, :].detach().cpu().numpy()
            elif "color_pca_color" in dict_keys and self.neuralpoint_colorfeature_chbox.checked:
                if local_mask is not None and self.local_map_chbox.checked:
                    neural_point_colors = data_packet.neural_points_data["color_pca_color"][local_mask]
                else:
                    neural_point_colors = data_packet.neural_points_data["color_pca_color"]
                neural_point_colors_np = neural_point_colors[::neural_point_vis_down_rate, :].detach().cpu().numpy()
            elif "ts" in dict_keys and self.neuralpoint_ts_chbox.checked:
                if local_mask is not None and self.local_map_chbox.checked:
                    ts_np = (data_packet.neural_points_data["ts"][local_mask])
                else:
                    ts_np = (data_packet.neural_points_data["ts"])
                ts_np = ts_np[::neural_point_vis_down_rate].detach().cpu().numpy()
                ts_np = ts_np / ts_np.max()
                color_map = cm.get_cmap("jet")
                neural_point_colors_np = color_map(ts_np)[:, :3].astype(np.float64)
            elif self.neuralpoint_height_chbox.checked:
                z_values = neural_point_position_np[:, 2]
                z_min, z_max = z_values.min(), z_values.max()
                z_normalized = (z_values - z_min) / (z_max - z_min + 1e-6)
                color_map = cm.get_cmap("jet")
                neural_point_colors_np = color_map(z_normalized)[:, :3].astype(np.float64)
            elif "color" in dict_keys:
                if local_mask is not None and self.local_map_chbox.checked:
                    neural_point_colors = data_packet.neural_points_data["color"][local_mask]
                else:
                    neural_point_colors = data_packet.neural_points_data["color"]
                neural_point_colors_np = neural_point_colors[::neural_point_vis_down_rate, :].detach().cpu().numpy()
            
            neural_point_valid_mask = None
            if "valid_mask" in dict_keys:
                if local_mask is not None and self.local_map_chbox.checked:
                    neural_point_valid_mask = data_packet.neural_points_data["valid_mask"][local_mask]
                else:
                    neural_point_valid_mask = data_packet.neural_points_data["valid_mask"]
                neural_point_valid_mask = neural_point_valid_mask[::neural_point_vis_down_rate].detach().cpu().numpy()

                valid_neural_point_position = neural_point_position_np[neural_point_valid_mask]                
                invalid_neural_point_position = neural_point_position_np[~neural_point_valid_mask]

                self.neural_points.points = o3d.utility.Vector3dVector(valid_neural_point_position)
                self.invalid_neural_points.points = o3d.utility.Vector3dVector(invalid_neural_point_position)
                
                if neural_point_colors_np is not None:
                    valid_neural_point_colors = neural_point_colors_np[neural_point_valid_mask]
                    invalid_neural_point_colors = neural_point_colors_np[~neural_point_valid_mask]
                    invalid_neural_point_colors[:,:] = (0, 0, 0) # invalid part set to black for vis

                    self.neural_points.colors = o3d.utility.Vector3dVector(valid_neural_point_colors)
                    self.invalid_neural_points.colors = o3d.utility.Vector3dVector(invalid_neural_point_colors)
                
            else:
                self.neural_points.points = o3d.utility.Vector3dVector(neural_point_position_np)
                if neural_point_colors_np is not None:
                    self.neural_points.colors = o3d.utility.Vector3dVector(neural_point_colors_np)
                self.invalid_neural_points = o3d.geometry.PointCloud()

            # if self.ego_chbox.checked:
            #     self.neural_points.transform(np.linalg.inv(self.cur_pose))

            self.widget3d.scene.remove_geometry(self.neural_point_name)
            self.widget3d.scene.add_geometry(self.neural_point_name, self.neural_points, self.neural_points_render)
            
            self.widget3d.scene.remove_geometry(self.invalid_neural_point_name)
            self.widget3d.scene.add_geometry(self.invalid_neural_point_name, self.invalid_neural_points, self.neural_points_render)

        self.widget3d.scene.show_geometry(self.neural_point_name, self.neural_point_chbox.checked)
        self.widget3d.scene.show_geometry(self.invalid_neural_point_name, self.invalid_neural_point_chbox.checked)
    
    def visualize_scan(self, data_packet = None):
        if data_packet is None:
            data_packet = self.cur_data_packet

        if data_packet is None:
            return

        if self.scan_chbox.checked and data_packet.current_pointcloud_xyz is not None:
            self.scan.points = o3d.utility.Vector3dVector(data_packet.current_pointcloud_xyz)
            if data_packet.current_pointcloud_rgb is not None:
                self.scan.colors = o3d.utility.Vector3dVector(data_packet.current_pointcloud_rgb)
        
            if not (self.config.color_on or self.config.semantic_on or self.scan_regis_color_chbox.checked):
                self.scan.paint_uniform_color(SILVER)

            if self.scan_height_color_chbox.checked:
                z_values = data_packet.current_pointcloud_xyz[:, 2]
                z_min, z_max = z_values.min(), z_values.max()
                z_normalized = (z_values - z_min) / (z_max - z_min + 1e-6)
                color_map = cm.get_cmap("jet")
                scan_colors_np = color_map(z_normalized)[:, :3].astype(np.float64)
                self.scan.colors = o3d.utility.Vector3dVector(scan_colors_np)

            # if self.ego_chbox.checked:
            #     self.scan.transform(np.linalg.inv(self.cur_pose))

            self.widget3d.scene.remove_geometry(self.scan_name)
            self.widget3d.scene.add_geometry(self.scan_name, self.scan, self.scan_render)

        self.widget3d.scene.show_geometry(self.scan_name, self.scan_chbox.checked)

    def visualize_mesh(self, data_packet = None):
        if data_packet is None:
            data_packet = self.cur_data_packet

        if data_packet is None:
            return

        if data_packet.mesh_verts is not None and data_packet.mesh_faces is not None:
            self.mesh = o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(data_packet.mesh_verts),
                o3d.utility.Vector3iVector(data_packet.mesh_faces),
                )
            self.mesh.compute_vertex_normals()

            if data_packet.mesh_verts_rgb is not None:
                self.mesh.vertex_colors = o3d.utility.Vector3dVector(data_packet.mesh_verts_rgb)
            
            if self.mesh_height_chbox.checked:
                z_values = np.array(self.mesh.vertices, dtype=np.float64)[:, 2]
                z_min, z_max = z_values.min(), z_values.max()
                z_normalized = (z_values - z_min) / (z_max - z_min + 1e-6)
                color_map = cm.get_cmap("jet")
                mesh_verts_colors_np = color_map(z_normalized)[:, :3].astype(np.float64)
                self.mesh.vertex_colors = o3d.utility.Vector3dVector(mesh_verts_colors_np)

            self.widget3d.scene.remove_geometry(self.mesh_name)
            self.widget3d.scene.add_geometry(self.mesh_name, self.mesh, self.mesh_render) 

        self.widget3d.scene.show_geometry(self.mesh_name, self.mesh_chbox.checked)

    def visualize_sdf_slice(self, data_packet = None):
        if data_packet is None:
            data_packet = self.cur_data_packet
        
        if data_packet is None:
            return

        if self.sdf_chbox.checked and data_packet.sdf_slice_xyz is not None and data_packet.sdf_slice_rgb is not None:
            self.sdf_slice.points = o3d.utility.Vector3dVector(data_packet.sdf_slice_xyz)
            self.sdf_slice.colors = o3d.utility.Vector3dVector(data_packet.sdf_slice_rgb)
            self.widget3d.scene.remove_geometry(self.sdf_name)
            self.widget3d.scene.add_geometry(self.sdf_name, self.sdf_slice, self.sdf_render)

        self.widget3d.scene.show_geometry(self.sdf_name, self.sdf_chbox.checked)

    def visualize_sdf_pool(self, data_packet = None):
        if data_packet is None:
            data_packet = self.cur_data_packet

        if data_packet is None:
            return

        if self.sdf_pool_chbox.checked and data_packet.sdf_pool_xyz is not None and data_packet.sdf_pool_rgb is not None:
            self.sdf_pool.points = o3d.utility.Vector3dVector(data_packet.sdf_pool_xyz)
            self.sdf_pool.colors = o3d.utility.Vector3dVector(data_packet.sdf_pool_rgb)
            self.widget3d.scene.remove_geometry(self.sdf_pool_name)
            self.widget3d.scene.add_geometry(self.sdf_pool_name, self.sdf_pool, self.sdf_pool_render)

        self.widget3d.scene.show_geometry(self.sdf_pool_name, self.sdf_pool_chbox.checked)
    
    def visualize_rendered_scan(self, data_packet = None):
        if data_packet is None:
            data_packet = self.cur_data_packet

        if data_packet is None:
            return
        
        if data_packet.current_rendered_xyz is not None:
            self.rendered_scan.points = o3d.utility.Vector3dVector(data_packet.current_rendered_xyz)
            if data_packet.current_rendered_rgb is not None:
                self.rendered_scan.colors = o3d.utility.Vector3dVector(data_packet.current_rendered_rgb)
            if self.rendered_scan_chbox.checked:
                self.widget3d.scene.remove_geometry(self.rendered_scan_name)
                self.widget3d.scene.add_geometry(self.rendered_scan_name, self.rendered_scan, self.scan_render)
    
    # main rendering function for the 3D visualizer
    def render_o3d_image(self, results, current_cam, normal_in_world_frame: bool = True, normal_with_alpha: bool = True):

        if (not self.gs_chbox.checked) or (not self.config.gs_on):
            return None # don't show gs rendering results

        rgb = (
                (torch.clamp(results["render"], min=0, max=1.0) * 255)
                .byte()
                .permute(1, 2, 0)
                .contiguous()
                .cpu()
                .numpy()
            )

        # if self.step % 300 == 0 and self.brisque_score_on:  # 3 second
        #     cur_brisque_score = self.brisque_scorer.score(img=rgb)
        #     self.brisque_score_info.text = ("Current view BRISQUE score: {:.3f}".format(cur_brisque_score))
        
        if self.depth_chbox.checked:
            depth = results["surf_depth"]
            if depth is None:
                return None # don't show gs rendering results
            
            if results["rend_alpha"] is not None and self.depth_filter_with_alpha_chbox.checked:
                valid_depth_mask = (results["rend_alpha"] > self.config.eval_depth_min_accu_alpha)
                depth[~valid_depth_mask] = 0.0

            depth = depth.detach().cpu().numpy()
            # max_depth = np.max(depth)
            depth_color = (colorize_depth_maps(depth, 0.1, self.config.max_range*0.8)[0]*255.0).astype(np.uint8) # 1, 3, H, W 
            depth_color = np.transpose(depth_color, (1, 2, 0)) # H, W, 3
            depth_color = np.ascontiguousarray(depth_color)
            render_img = o3d.geometry.Image(depth_color)

        elif self.normal_chbox.checked:
            normal = results["rend_normal"]
            if normal is None:
                return None # don't show gs rendering results

            if normal_in_world_frame: 
            # transform to world frame
                normal = -1.0 * (normal.permute(1,2,0) @ (current_cam.world_view_transform[:3,:3].T)).permute(2,0,1)

            if normal_with_alpha:
                normal_norm = normal.norm(2, dim=0) 
                normal_color = 0.5 * (normal_norm - normal) #   # convert to the normal vis color
            else:
                normal = torch.nn.functional.normalize(normal, dim=0) # normalize to norm==1 # don't do this, for small opacity region, we just downweight its normal
                normal_color = 0.5 * (1 - normal)

            normal_color = (normal_color.permute(1,2,0).detach().cpu().numpy() * 255.0).astype(np.uint8) 
            normal_color = np.ascontiguousarray(normal_color)
            render_img = o3d.geometry.Image(normal_color)

        elif self.d2n_chbox.checked:
            d2n = results["surf_normal"]
            if d2n is None:
                return None # don't show gs rendering results
            
            if normal_in_world_frame: 
            # transform to world frame
                d2n = -1.0 * (d2n.permute(1,2,0) @ (current_cam.world_view_transform[:3,:3].T)).permute(2,0,1)

            if normal_with_alpha:
                d2n_norm = d2n.norm(2, dim=0) 
                d2n_color =  0.5 * (d2n_norm - d2n) # convert to the normal vis color
            else:
                d2n = torch.nn.functional.normalize(d2n, dim=0) # normalize to norm==1
                d2n_color = 0.5 * (1 - d2n)
            
            d2n_color = (d2n_color.permute(1,2,0).detach().cpu().numpy() * 255.0).astype(np.uint8) 
            d2n_color = np.ascontiguousarray(d2n_color)
            render_img = o3d.geometry.Image(d2n_color)

        elif self.opacity_chbox.checked:
            
            opacity = results["rend_alpha"]
            
            if opacity is None:
                return None # don't show gs rendering results

            opacity = opacity.detach().cpu().numpy()

            opacity_color = (colorize_depth_maps(opacity, 0.0, 1.0, cmap="jet")[0]*255.0).astype(np.uint8)

            opacity_color = np.transpose(opacity_color, (1, 2, 0)) # H, W, 3
            opacity_color = np.ascontiguousarray(opacity_color)
            
            render_img = o3d.geometry.Image(opacity_color)

        elif self.ellipsoid_chbox.checked:

            if gl_issue:

                if not self.gl_issue_notified:
                    print("[GUI] GL issue, disable ellipsoid rendering, set gl_issue to False to enable it")
                    self.gl_issue_notified = True
                
                return

            if self.cur_data_packet is None:
                return

            glfw.poll_events()
            # gl.glClearColor(0, 0, 0, 1.0)
            gl.glClearColor(1.0, 1.0, 1.0, 0.0)
            gl.glClear(
                gl.GL_COLOR_BUFFER_BIT
                | gl.GL_DEPTH_BUFFER_BIT
                | gl.GL_STENCIL_BUFFER_BIT
            )

            w = int(self.window.size.width * self.widget3d_width_ratio)
            glfw.set_window_size(self.window_gl, w, self.window.size.height)
            self.g_camera.fovy = current_cam.FoVy
            self.g_camera.update_resolution(self.window.size.height, w)
            self.g_renderer.set_render_reso(w, self.window.size.height)
            frustum = create_frustum(
                np.linalg.inv(cv_gl @ self.widget3d.scene.camera.get_view_matrix())
            )

            self.g_camera.position = frustum.eye.astype(np.float32)
            self.g_camera.target = frustum.center.astype(np.float32)
            self.g_camera.up = frustum.up.astype(np.float32)

            # neural gaussian version
            self.gaussians_gl.xyz = results["gaussian_xyz"].cpu().numpy()

            gaussian_scale = results["gaussian_scale"]
            if self.config.gs_type == "2d_gs":
                thin_dim_scale = torch.full((gaussian_scale.shape[0], 1), 1e-7).to(gaussian_scale) # already after activation, last dim, very thin
                gaussian_scale = torch.cat((gaussian_scale, thin_dim_scale), dim=1) # NK, 3

            self.gaussians_gl.scale = gaussian_scale.cpu().numpy()
            self.gaussians_gl.rot = results["gaussian_rot"].cpu().numpy()
            self.gaussians_gl.opacity = results["gaussian_alpha"].cpu().numpy()
            gaussians_gl_rgb = results["gaussian_color"].cpu().numpy()
            self.gaussians_gl.sh = (gaussians_gl_rgb - 0.5) / 0.28209479177387814

            if self.elliopsoid_2d_chbox.checked:
                render_mode = -3 # 2D surfel
            else:
                render_mode = -4 # 3D elliopsoid

            self.update_activated_renderer_state(self.gaussians_gl, render_mode) # > 0 render 0-ith SH dim, -1 depth, -2 bill board, -3 flat ball (better fit with Gaussian Surfels), -4 gaussian ball
            self.g_renderer.sort_and_update(self.g_camera)
            width, height = glfw.get_framebuffer_size(self.window_gl)
            self.g_renderer.draw()
            bufferdata = gl.glReadPixels(
                0, 0, width, height, gl.GL_RGB, gl.GL_UNSIGNED_BYTE
            )
            img = np.frombuffer(bufferdata, np.uint8, -1).reshape(height, width, 3)
            img = cv2.flip(img, 0)
            render_img = o3d.geometry.Image(img)
            glfw.swap_buffers(self.window_gl)
        
        else:
            render_img = o3d.geometry.Image(rgb)

        return render_img

    def rasterise(self, current_cam):

        if self.cur_data_packet is None:
            return None

        if self.cur_data_packet.neural_points_data is None:
            return None

        # print("# Local neural point:", self.neural_points.local_count())

        # print(self.decoders["gauss_xyz"])

        with torch.no_grad():

            render_tic = get_time()

            rendering_data = render(current_cam, None, 
                self.cur_data_packet.neural_points_data, 
                self.decoders, self.cur_base_gaussians, self.background, 
                scaling_modifier=self.scaling_slider.double_value, 
                down_rate=self.scaling_slider_downrate.int_value, 
                dist_concat_on=self.config.dist_concat_on, 
                view_concat_on=self.config.view_concat_on, 
                correct_exposure=False,
                learn_color_residual=self.config.learn_color_residual,
                front_only_on=(not self.backface_chbox.checked),
                d2n_on=self.d2n_chbox.checked,
                gs_type=self.config.gs_type,
                displacement_range_ratio=self.config.displacement_range_ratio,
                max_scale_ratio=self.config.max_scale_ratio,
                unit_scale_ratio=self.config.unit_scale_ratio)
            
            render_toc = get_time()

            if rendering_data is not None:
                if "local_view_gaussian_count" in list(rendering_data.keys()):
                    gaussians_all_count = rendering_data["alpha_all"].shape[0]
                    gaussians_valid_count = rendering_data["local_view_gaussian_count"]
                    valid_ratio = 1.0 * gaussians_valid_count / gaussians_all_count
                    mean_valid_count = valid_ratio * self.config.spawn_n_gaussian
                    cur_visble_gaussian_count = torch.sum(rendering_data["visibility_filter"]).item()
                    self.gaussian_info.text = "# Current view Gaussians: {} (valid: {:.1f} / {})".format(cur_visble_gaussian_count, mean_valid_count, self.config.spawn_n_gaussian)

            render_time = render_toc - render_tic # s
            render_freq = 1.0/render_time
            
            if self.step % 10 == 0:
                self.freq_info.text = "Render FPS: {:.1f}".format(render_freq)
        
        return rendering_data

        # return None

    # main function here for render gs
    def render_gui(self):
        if not self.init:
            return

        current_cam = self.get_current_cam()

        if (not self.gs_chbox.checked) or (not self.config.gs_on):
            if self.render_img is None:
                return
            self.render_img = None
        else: # gs_chbox checked
            results = self.rasterise(current_cam)
            if results is None:
                return
            self.render_img = self.render_o3d_image(results, current_cam, 
                    self.normal_in_world_chbox.checked, 
                    self.normal_with_alpha_chbox.checked)
            results = {} # free memory 
        ## self.widget3d.scene.set_background([0, 0, 0, 1], self.render_img)
        self.widget3d.scene.set_background([1, 1, 1, 1], self.render_img)


    # this is used
    def _update_thread(self):
        while True:
            time.sleep(0.01)
            self.step += 1
            if self.process_finished:
                o3d.visualization.gui.Application.instance.quit()
                print("[GUI] Closing Visualization")
                break
            
            # print(self.step)

            def update():
                if self.slider_render.is_on:
                    # print("UPDATE scene")
                    if self.step % 3 == 0: # per 0.03s # 30 Hz
                        self.render_gui() # stucked here

                        # if self.slider_recording.is_on:
                        #     model_matrix = np.asarray(self.widget3d.scene.camera.get_model_matrix())
                        #     cur_extrinsic = model_matrix_to_extrinsic_matrix(model_matrix)
                        #     cam_pose = np.linalg.inv(cur_extrinsic)
                        #     self.recorded_poses.append(cam_pose)

                    if self.step % 10 == 0: # per 0.2s # 5 Hz # receive latest data
                        self.receive_data(self.q_main2vis) # this is also slow

                    if self.step % 20 == 0:
                        self.send_data()

                    if self.init and self.followcam_chbox.checked and not self.config.gs_on and not self.still_chbox.checked: 
                        if self.local_map_chbox.checked:
                            self.center_bev_local()
                        else:
                            self.center_bev()

                    if self.step % 100 == 0: 
                        remove_gpu_cache() # remove cache regularly
                
                else:
                    while not self.q_main2vis.empty(): # free the queue
                        self.q_main2vis.get()

                if self.step >= 1e9:
                    self.step = 0

            gui.Application.instance.post_to_main_thread(self.window, update)


def run(params_gui=None):
    app = o3d.visualization.gui.Application.instance
    app.initialize()
    win = SLAM_GUI(params_gui)
    app.run()


def main():
    app = o3d.visualization.gui.Application.instance
    app.initialize()
    win = SLAM_GUI()
    app.run()

def generate_circle(radius=1.0, num_points=100):
    angles = np.linspace(0, 2 * np.pi, num_points)
    # Circle in the XY plane
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    z = np.zeros(num_points)  # Z-coordinates are 0 for a flat circle in XY-plane
    circle_points = np.vstack((x, y, z)).T  # Shape (num_points, 3)
    return circle_points

def model_matrix_to_extrinsic_matrix(model_matrix):
    return np.linalg.inv(model_matrix @ FromGLGamera)

def create_camera_intrinsic_from_size(width=1024, height=768, hfov=60.0, vfov=60.0):
    fx = (width / 2.0)  / np.tan(np.radians(hfov)/2)
    fy = (height / 2.0)  / np.tan(np.radians(vfov)/2)
    fx = fy # not sure why, but it looks like fx should be governed/limited by fy
    return np.array(
        [[fx, 0, width / 2.0],
         [0, fy, height / 2.0],
         [0, 0,  1]])

# copyright: Nacho et al. KISS-ICP
def write_kitti_format_poses(filename: str, poses: List[np.ndarray]):
    def _to_kitti_format(poses: np.ndarray) -> np.ndarray:
        return np.array([np.concatenate((pose[0], pose[1], pose[2])) for pose in poses])

    np.savetxt(fname=f"{filename}.txt", X=_to_kitti_format(poses))


if __name__ == "__main__":
    main()
