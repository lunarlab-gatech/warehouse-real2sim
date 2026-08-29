#!/usr/bin/env python3
# @file      neural_gaussians.py
# @author    Yue Pan     [yue.pan@igg.uni-bonn.de]
# Copyright (c) 2025 Yue Pan, all rights reserved

import sys

import matplotlib.cm as cm
import numpy as np
import open3d as o3d
import torch
import torch.nn as nn
import torch.nn.functional as F
from rich import print

from utils.config import Config
from utils.tools import (
    apply_quaternion_rotation,
    get_time,
    quat_multiply,
    rotmat_to_quat,
    transform_batch_torch,
    voxel_down_sample_min_value_torch,
    voxel_down_sample_torch,
    feature_pca_torch,
)

class NeuralPoints(nn.Module):
    def __init__(self, config: Config) -> None:

        super().__init__()

        self.config = config
        self.silence = config.silence

        self.geo_feature_dim = config.feature_dim
        self.geo_feature_std = config.feature_std

        self.color_feature_dim = config.color_feature_dim
        self.color_feature_std = config.feature_std

        self.mean_grid_sampling = False  # NOTE: sample the gravity center of the points inside the voxel or keep the point that is closest to the voxel center

        self.device = config.device
        self.dtype = config.dtype
        self.idx_dtype = torch.int64
        # torch.int64/32 does not have much speed difference

        self.resolution = config.voxel_size_m

        self.buffer_size = config.buffer_size

        self.sorrounding_map_radius = config.sorrounding_map_radius

        self.temporal_local_map_on = True

        if not self.config.track_on and self.config.temporal_local_map_off:
            # print("Not using temporal local map")
            self.temporal_local_map_on = False

        self.diff_travel_dist_local = (
            self.config.local_map_radius * self.config.local_map_travel_dist_ratio
        )

        self.diff_ts_local = (
            self.config.diff_ts_local
        )  # not used now, switch to travel distance

        self.local_orientation = torch.eye(3, dtype=self.dtype, device=self.device)
        self.local_position = torch.zeros(3, dtype=self.dtype, device=self.device)

        self.cur_ts = 0  # current frame No. or the current timestamp
        self.max_ts = 0

        self.travel_dist = None  # for determine the local map, update from the dataset class for each frame
        self.est_poses = None
        self.after_pgo = False

        # for hashing (large prime numbers)
        self.primes = torch.tensor(
            [73856093, 19349669, 83492791], dtype=self.idx_dtype, device=self.device
        )

        # initialization
        # the global map
        self.buffer_pt_index = torch.full(
            (self.buffer_size,), -1, dtype=self.idx_dtype, device=self.device
        )

        # we can actually also optimize the position, not necessary in the indexed voxel
        self.neural_points = torch.empty((0, 3), dtype=self.dtype, device=self.device)
        self.point_orientations = torch.empty(
            (0, 4), dtype=self.dtype, device=self.device
        )  # as quaternion
        self.geo_features = torch.empty(
            (1, self.geo_feature_dim), dtype=self.dtype, device=self.device
        )
        if self.config.color_on:
            self.color_on = True
            self.color_features = torch.empty(
                (1, self.color_feature_dim), dtype=self.dtype, device=self.device
            )
            self.point_colors = torch.empty((0, 3), dtype=self.dtype, device=self.device) # RGB [0-1]
        else:
            self.color_on = False
            self.color_features = None
            self.point_colors = None

        # feature pca (for visualization only)
        self.geo_feature_pca = self.color_feature_pca = None

        # here, the ts represent the actually processed frame id (not neccessarily the frame id of the dataset)
        self.point_ts_create = torch.empty(
            (0), device=self.device, dtype=torch.int
        )  # create ts
        self.point_ts_update = torch.empty(
            (0), device=self.device, dtype=torch.int
        )  # last update ts
        self.point_certainties = torch.empty((0), dtype=self.dtype, device=self.device)
        
        self.valid_color_mask = torch.empty(0, dtype=torch.bool, device=self.device) # N, 1 # bool
        self.valid_gs_mask = torch.empty(0, dtype=torch.bool, device=self.device) # N, 1 # bool 
        self.free_gs_mask = torch.empty(0, dtype=torch.bool, device=self.device) # deprecated

        # the local map
        self.local_neural_points = torch.empty(
            (0, 3), dtype=self.dtype, device=self.device
        )

        self.local_point_orientations = torch.empty(
            (0, 4), dtype=self.dtype, device=self.device
        )  # as quaternion [w,x,y,z]
        self.local_geo_features = nn.Parameter()
        self.local_color_features = nn.Parameter()
        self.local_point_certainties = torch.empty(
            (0), dtype=self.dtype, device=self.device
        )
        self.local_point_ts_update = torch.empty(
            (0), device=self.device, dtype=torch.int
        )

        self.local_point_colors = None

        self.local_mask = None
        self.global2local = None
        
        # we define another local map called sorrounding_map, which is actually outside the local map
        # with a certain radius.
        # this sorrounding map would be used for rendering (for those relative far away backgrounds)
        # however, the neural points in this sorrounding map would not be optimized during the 
        # training process
        self.sorrounding_mask = None

        # this is just for vis
        self.local_valid_color_mask = torch.empty(0, dtype=torch.bool, device=self.device) # current not used
        # this is for gs (as a kind of pruning)
        self.local_valid_gs_mask = torch.empty(0, dtype=torch.bool, device=self.device)
        # restricted by sdf or not (deprecated)
        self.local_free_gs_mask = torch.empty(0, dtype=torch.bool, device=self.device)

        # set neighborhood search region
        self.set_search_neighborhood(
            num_nei_cells=config.num_nei_cells, search_alpha=config.search_alpha
        )

        self.cur_memory_mb = 0.0
        self.memory_footprint = []

        self.to(self.device)

    def is_empty(self):
        return self.neural_points.shape[0] == 0

    def count(self, valid_gs_only = False):
        if valid_gs_only:
            return self.neural_points[self.valid_gs_mask].shape[0]
        else:
            return self.neural_points.shape[0]

    def local_count(self, valid_gs_only = False):
        if valid_gs_only:
            return self.local_neural_points[self.local_valid_gs_mask].shape[0]
        else:
            return self.local_neural_points.shape[0]

    def free_count(self):
        return torch.sum(self.free_gs_mask).int()

    def free_local_count(self):
        return torch.sum(self.local_free_gs_mask).int()

    @property
    def has_neural_points(self):
        return (self.count() > 0)

    def record_memory(self, verbose: bool = True, record_footprint: bool = False):
        """
        Record the memory consumption of the neural point map
        """
        if verbose:
            print("# Global neural point: %d (%d free, %d valid)" % (self.count(), self.free_count(), self.count(valid_gs_only=True)))
            print("# Local  neural point: %d (%d free, %d valid)" % (self.local_count(), self.free_local_count(), self.local_count(valid_gs_only=True)))
        neural_point_count = self.count()
        # feature plus neural point position and orientation
        point_dim = self.geo_feature_dim + 3 + 4    
        if self.color_features is not None:
            point_dim += self.color_feature_dim  # also include the color feature
        self.cur_memory_mb = neural_point_count * point_dim * 4 / 1024 / 1024  # as float32
        if verbose:
            print("Current map memory consumption: %f (MB)" % self.cur_memory_mb)
        if record_footprint:
            self.memory_footprint.append(self.cur_memory_mb)

    def update(
        self,
        points: torch.Tensor,
        colors: torch.Tensor,
        normals: torch.Tensor,
        sensor_position: torch.Tensor = None,
        sensor_orientation: torch.Tensor = None,
        cur_ts: int = 0,
        is_reliable: bool = True # if false, it means these neural points are initialized with the mono depth estimation, which are not accurate
    ):
        '''
        Update the neural point map using new observations
        '''

        cur_resolution = self.resolution
        # if self.mean_grid_sampling:
        #     sample_points = meanGridSampling(points, resolution=cur_resolution)
        # take the point that is the closest to the voxel center (now used)
        sample_idx = voxel_down_sample_torch(points, cur_resolution)
        sample_points = points[sample_idx]

        sample_colors = None
        if colors is not None:
            sample_colors = colors[sample_idx]

        sample_normals = None
        if normals is not None:
            sample_normals = normals[sample_idx]

        grid_coords = (sample_points / cur_resolution).floor().to(self.primes)
        buffer_size = int(self.buffer_size)
        hash_value = torch.fmod((grid_coords * self.primes).sum(-1), buffer_size)

        hash_idx = self.buffer_pt_index[hash_value]

        # not occupied before or is occupied but already far away (then it would be a hash collision)
        if not self.is_empty():
            vec_points = self.neural_points[hash_idx] - sample_points
            dist2 = torch.sum(vec_points**2, dim=-1)

            update_mask = (hash_idx == -1) | (dist2 > 3 * cur_resolution**2)

            if sample_colors is not None:
                # # only use the valid part (this can be used for the case that we use the full point cloud)
                sample_points_valid_color_mask = (sample_colors[:, 0] >= 0.0) # negative value indicate invalid
                # then this would be all True (this can be used for the case that we use only the colorized part of the point cloud)
        
                color_update_mask = (hash_idx > -1) & (self.valid_color_mask[hash_idx] == 0) & sample_points_valid_color_mask
                hash_idx_color_update = hash_idx[color_update_mask]
                self.point_colors[hash_idx_color_update] = sample_colors[color_update_mask] # N, 3

                self.valid_color_mask[hash_idx_color_update] = 1 # valid again now
                
            # print("# Color update count:", color_update_mask.sum().item()) 

            if self.temporal_local_map_on: # only done for the slam mode
                # the voxel is not occupied before or the case when hash collision happens
                # delta_t = (cur_ts - self.point_ts_create[hash_idx]) # use time diff
                delta_travel_dist = (
                    self.travel_dist[cur_ts]
                    - self.travel_dist[self.point_ts_update[hash_idx]]
                )  # use travel dist diff

                # the last time mask is necessary
                update_mask = update_mask | (delta_travel_dist > self.diff_travel_dist_local)
        else:
            update_mask = torch.ones(
                hash_idx.shape, dtype=torch.bool, device=self.device
            )

        added_pt = sample_points[update_mask]

        added_colors = None
        if sample_colors is not None:
            added_colors = sample_colors[update_mask]
            # print(added_colors) # color also between 0 and 1

        added_normals = None
        if sample_normals is not None:
            added_normals = sample_normals[update_mask]

        new_point_count = added_pt.shape[0]

        new_point_ratio = new_point_count / sample_points.shape[0]

        cur_pt_idx = self.buffer_pt_index[hash_value]
        # allocate new neural points
        cur_pt_count = self.neural_points.shape[0]
        cur_pt_idx[update_mask] = (
            torch.arange(new_point_count, dtype=self.idx_dtype, device=self.device)
            + cur_pt_count
        )

        # torch.cat could be slow for large map
        self.buffer_pt_index[hash_value] = cur_pt_idx
        self.neural_points = torch.cat((self.neural_points, added_pt), 0)

        added_orientations = [[1, 0, 0, 0]] * new_point_count
        added_orientations = torch.tensor(
            added_orientations, dtype=self.dtype, device=self.device
        )
        self.point_orientations = torch.cat(
            (self.point_orientations, added_orientations), 0
        )

        new_points_ts = (
            torch.ones(new_point_count, device=self.device, dtype=torch.int) * cur_ts
        )
        self.point_ts_create = torch.cat((self.point_ts_create, new_points_ts), 0)
        self.point_ts_update = torch.cat((self.point_ts_update, new_points_ts), 0)

        # with padding in the end
        new_fts = self.geo_feature_std * torch.randn(
            new_point_count + 1,
            self.geo_feature_dim,
            device=self.device,
            dtype=self.dtype,
        )
        self.geo_features = torch.cat((self.geo_features[:-1], new_fts), 0)

        # with padding in the end
        if self.color_features is not None:
            new_fts = self.color_feature_std * torch.randn(
                new_point_count + 1,
                self.color_feature_dim,
                device=self.device,
                dtype=self.dtype,
            )
            self.color_features = torch.cat((self.color_features[:-1], new_fts), 0)

        new_certainty = torch.zeros(
            new_point_count, device=self.device, dtype=self.dtype, requires_grad=False
        )
        self.point_certainties = torch.cat((self.point_certainties, new_certainty), 0)

        if is_reliable:
            new_free_mask = torch.zeros((new_point_count), dtype=bool, device=self.device)
        else:
            new_free_mask = torch.ones((new_point_count), dtype=bool, device=self.device)
        self.free_gs_mask = torch.cat((self.free_gs_mask, new_free_mask), 0)
        # free_gs_mask: free ones are casted from the mono depth estimation

        # update RGB color
        if added_colors is not None:
            self.point_colors = torch.cat((self.point_colors, added_colors), 0)
        
        
        ## Mask
        if added_colors is not None:
            new_valid_color_mask = added_colors[:, 0] >= 0.0 # negative value indicates invalid
            self.valid_color_mask = torch.cat((self.valid_color_mask, new_valid_color_mask), 0)
        else:
            self.valid_color_mask = torch.cat((self.valid_color_mask, torch.ones((new_point_count), dtype=bool, device=self.device)), 0)

        self.valid_gs_mask = torch.cat((self.valid_gs_mask, torch.ones((new_point_count), dtype=bool, device=self.device)), 0) # all True

        if sensor_position is not None:
            self.reset_local_map(
                sensor_position, sensor_orientation, cur_ts
            )  # no need to recreate hash

        return new_point_ratio

    # we also set the sorrounding map here
    def reset_local_map(
        self,
        sensor_position: torch.Tensor,
        sensor_orientation: torch.Tensor = None,
        cur_ts: int = 0,
        use_travel_dist: bool = True,
        diff_ts_local: int = 50,
    ):
        '''
        Reset the local map using the sensor position and frame timestep
        '''

        self.cur_ts = cur_ts
        self.max_ts = max(self.max_ts, cur_ts)

        if self.temporal_local_map_on:
            if self.config.use_mid_ts:
                point_ts_used = (
                    (self.point_ts_create + self.point_ts_update) / 2
                ).int()
            else:
                point_ts_used = self.point_ts_create

            if use_travel_dist: # self.travel_dist as torch tensor
                delta_travel_dist = torch.abs(
                    self.travel_dist[cur_ts] - self.travel_dist[point_ts_used]
                )
                time_mask = (delta_travel_dist < self.diff_travel_dist_local) # increase this value now
            else:  # use delta_t
                delta_t = torch.abs(cur_ts - point_ts_used)
                time_mask = (delta_t < diff_ts_local) 
            
            if torch.sum(time_mask) < 100: # not enough neural points in the temporal window, we set all true to avoid error
                time_mask = torch.ones(self.count(), dtype=torch.bool, device=self.device) # all true
        
        else:
            time_mask = torch.ones(self.count(), dtype=torch.bool, device=self.device) # all true

        # speed up by calulating distance only with the t filtered points
        masked_vec2sensor = self.neural_points[time_mask] - sensor_position

        if self.config.range_filter_2d:
            masked_dist2sensor = torch.sum(masked_vec2sensor[:,:2]**2, dim=-1)  # dist square
        else:
            masked_dist2sensor = torch.sum(masked_vec2sensor**2, dim=-1)  # dist square

        dist_mask = (masked_dist2sensor < self.config.local_map_radius**2)
        time_mask_idx = torch.nonzero(time_mask).squeeze() # True index
        local_mask_idx = time_mask_idx[dist_mask] # True index

        sorrounding_dist_mask = (masked_dist2sensor < (self.sorrounding_map_radius)**2)
    
        sorrounding_mask_idx = time_mask_idx[~dist_mask & sorrounding_dist_mask] # parts that are not in local map

        local_mask = torch.full((time_mask.shape), False, dtype=torch.bool, device=self.device)
        local_mask[local_mask_idx] = True 

        sorrounding_mask = torch.full((time_mask.shape), False, dtype=torch.bool, device=self.device)
        sorrounding_mask[sorrounding_mask_idx] = True
        sorrounding_mask = torch.cat(
            (sorrounding_mask, torch.tensor([True], device=self.device))
        )  # padding with one element in the end
        self.sorrounding_mask = sorrounding_mask

        self.local_neural_points = self.neural_points[local_mask]
        self.local_point_orientations = self.point_orientations[local_mask]
        self.local_point_certainties = self.point_certainties[local_mask]
        self.local_point_ts_update = self.point_ts_update[local_mask]
        if self.point_colors is not None:
            self.local_point_colors = self.point_colors[local_mask]


        self.local_valid_color_mask = self.valid_color_mask[local_mask]
        self.local_valid_gs_mask = self.valid_gs_mask[local_mask]
        self.local_free_gs_mask = self.free_gs_mask[local_mask]


        local_mask = torch.cat(
            (local_mask, torch.tensor([True], device=self.device))
        )  # padding with one element in the end
        self.local_mask = local_mask

        # if Flase (not in the local map), the mapping get an idx as -1
        global2local = torch.full_like(local_mask, -1).long()
        
        local_indices = torch.nonzero(local_mask).flatten()
        local_point_count = local_indices.size(0)
        global2local[local_indices] = torch.arange(
            local_point_count, device=self.device
        )
        global2local[-1] = -1  # invalid idx is still invalid after mapping

        self.global2local = global2local

        # here we reset the nn.parameter
        self.local_geo_features = nn.Parameter(self.geo_features[local_mask])
        if self.color_features is not None:
            self.local_color_features = nn.Parameter(self.color_features[local_mask])

        self.local_orientation = sensor_orientation  # not used
        self.local_position = sensor_position.float()



    def assign_local_to_global(self):
        '''
        Assign the local map back to the global map
        '''

        local_mask = self.local_mask
        self.point_certainties[local_mask[:-1]] = self.local_point_certainties
        self.point_ts_update[local_mask[:-1]] = self.local_point_ts_update
        
        self.geo_features[local_mask] = self.local_geo_features.data

        if self.color_features is not None:
            self.color_features[local_mask] = self.local_color_features.data

    def compute_feature_principle_components(self, down_rate: int = 1):
        """
        Compute the principle components of the feature
        """
        _, self.geo_feature_pca = feature_pca_torch((self.geo_features)[:-1], down_rate=down_rate, project_data=False)

        if self.color_features is not None:
            _, self.color_feature_pca = feature_pca_torch((self.color_features)[:-1], down_rate=down_rate, project_data=False)


    def query_feature(
        self,
        query_points: torch.Tensor,
        query_ts: torch.Tensor = None,
        accumulate_stability: bool = True,
        query_locally: bool = True,
        query_geo_feature: bool = True,
        query_color_feature: bool = False,
        use_only_measured_points: bool = True,
        use_only_valid_points: bool = False,
    ):
        '''
        Query and interpolate the feature from the neural point map
        '''
        
        if not query_geo_feature and not query_color_feature:
            sys.exit("you need to at least query one kind of feature")

        batch_size = query_points.shape[0]

        geo_features_vector = None
        color_features_vector = None

        nn_k = self.config.query_nn_k

        # T0 = get_time()

        # the slow part
        dists2, idx = self.radius_neighborhood_search(
            query_points, time_filtering=self.temporal_local_map_on and query_locally
        )

        # [N, K], [N, K]
        # if query globally, we do not have the time filtering

        # T10 = get_time()
        
        # Only use the reliable neural points for SDF mapping
        if use_only_measured_points:
            disabled_point_mask = self.free_gs_mask[idx] 
            idx[disabled_point_mask] = -1
        
        if use_only_valid_points:
            disabled_point_mask = ~self.valid_gs_mask[idx] 
            idx[disabled_point_mask] = -1

        # print("K=", idx.shape[-1]) # K
        if query_locally:
            idx = self.global2local[idx]
            # [N, K] # get the local idx using the global2local mapping

        nn_counts = (idx >= 0).sum(dim=-1)
        # then it could be larger than nn_k because this is before the sorting

        # T1 = get_time()

        dists2[idx == -1] = 9e3  # invalid, set to large distance

        # sort according to distance
        sorted_dist2, sorted_neigh_idx = torch.sort(dists2, dim=1)
        
        sorted_idx = idx.gather(1, sorted_neigh_idx)
        dists2 = sorted_dist2[:, :nn_k]  # only take the knn
        idx = sorted_idx[:, :nn_k]  # sorted local idx, only take the knn

        # dist2, idx are all with the shape [N, K]

        # T2 = get_time()

        valid_mask = idx >= 0  # [N, K]

        # valid_mask = (idx >= 0) & ()  # [N, K]

        if query_geo_feature:
            geo_features = torch.zeros(
                batch_size,
                nn_k,
                self.geo_feature_dim,
                device=self.device,
                dtype=self.dtype,
            )  # [N, K, F]
            if query_locally:
                geo_features[valid_mask] = self.local_geo_features[idx[valid_mask]]
            else:
                geo_features[valid_mask] = self.geo_features[idx[valid_mask]]
            if self.config.layer_norm_on:
                geo_features = F.layer_norm(geo_features, [self.geo_feature_dim])
        if query_color_feature and self.color_features is not None:
            color_features = torch.zeros(
                batch_size,
                nn_k,
                self.color_feature_dim,
                device=self.device,
                dtype=self.dtype,
            )  # [N, K, F]
            if query_locally:
                color_features[valid_mask] = self.local_color_features[idx[valid_mask]]
            else:
                color_features[valid_mask] = self.color_features[idx[valid_mask]]
            if self.config.layer_norm_on:
                color_features = F.layer_norm(color_features, [self.color_feature_dim])

        N, K = valid_mask.shape  # K = nn_k here

        # print(self.local_point_certainties)

        if query_locally:
            certainty = self.local_point_certainties[idx]  # [N, K]
            neighb_vector = (
                query_points.view(-1, 1, 3) - self.local_neural_points[idx]
            )  # [N, K, 3]
            quat = self.local_point_orientations[idx]  # [N, K, 4]
        else:
            certainty = self.point_certainties[idx]  # [N, K]
            neighb_vector = (
                query_points.view(-1, 1, 3) - self.neural_points[idx]
            )  # [N, K, 3]
            quat = self.point_orientations[idx]  # [N, K, 4]

        # quat[...,1:] *= -1. # inverse (not needed)
        # This has been doubly checked
        if self.after_pgo:
            neighb_vector = apply_quaternion_rotation(
                quat, neighb_vector
            )  # [N, K, 3] # passive rotation (axis rotation w.r.t point)
        neighb_vector[~valid_mask] = torch.zeros(
            1, 3, device=self.device, dtype=self.dtype
        )

        if query_geo_feature:
            geo_features_vector = torch.cat(
                (geo_features, neighb_vector), dim=2
            )  # [N, K, F+P]
        if query_color_feature and self.color_features is not None:
            color_features_vector = torch.cat(
                (color_features, neighb_vector), dim=2
            )  # [N, K, F+P]

        eps = 1e-15  # avoid nan (dividing by 0)

        weight_vector = 1.0 / (
            dists2 + eps
        )  # [N, K] # Inverse distance weighting (IDW), distance square

        weight_vector[~valid_mask] = 0.0  # pad for invalid voxels
        weight_vector[
            nn_counts == 0
        ] = eps  # all 0 would cause NaN during normalization

        # apply the normalization of weight
        weight_row_sums = torch.sum(weight_vector, dim=1).unsqueeze(1)
        weight_vector = torch.div(
            weight_vector, weight_row_sums
        )  # [N, K] # normalize the weight, to make the sum as 1

        # print(weight_vector)
        weight_vector[~valid_mask] = 0.0  # invalid has zero weight

        with torch.no_grad():
            # Certainty accumulation for each neural point according to the weight
            # Use scatter_add_ to accumulate the values for each index
            if accumulate_stability:  # only do it during the training mode
                idx[~valid_mask] = 0  # scatter_add don't accept -1 index
                if query_locally:
                    self.local_point_certainties.scatter_add_(
                        dim=0, index=idx.flatten(), src=weight_vector.flatten()
                    )
                    if (
                        query_ts is not None
                    ):  # update the last update ts for each neural point
                        idx_ts = query_ts.view(-1, 1).repeat(1, K)
                        idx_ts[~valid_mask] = 0
                        self.local_point_ts_update.scatter_reduce_(
                            dim=0,
                            index=idx.flatten(),
                            src=idx_ts.flatten(),
                            reduce="amax",
                            include_self=True,
                        )
                        # print(self.local_point_ts_update)
                else:
                    self.point_certainties.scatter_add_(
                        dim=0, index=idx.flatten(), src=weight_vector.flatten()
                    )
                # queried_certainty = None

                certainty[~valid_mask] = 0.0
                queried_certainty = torch.sum(certainty * weight_vector, dim=1)

            else:  # inference mode
                certainty[~valid_mask] = 0.0
                queried_certainty = torch.sum(certainty * weight_vector, dim=1)

        weight_vector = weight_vector.unsqueeze(-1)  # [N, K, 1]

        if self.config.weighted_first:
            if query_geo_feature:
                geo_features_vector = torch.sum(
                    geo_features_vector * weight_vector, dim=1
                )  # [N, F+P]

            if query_color_feature and self.color_features is not None:
                color_features_vector = torch.sum(
                    color_features_vector * weight_vector, dim=1
                )  # [N, F+P]

        # T3 = get_time()

        # in ms
        # print("time for nn     :", (T1-T0) * 1e3) # ////
        # print("time for sorting:", (T2-T1) * 1e3) # //
        # print("time for feature:", (T3-T2) * 1e3) # ///

        return (
            geo_features_vector,
            color_features_vector,
            weight_vector,
            nn_counts,
            queried_certainty,
        )
    
    def get_neural_points_o3d(
        self,
        query_global: bool = True,
        color_mode: int = -1,
        random_down_ratio: int = 1,
        cur_sensor_position = None,
        vis_normals = False,
        vis_free_gaussians = False,
        vis_invalid_gaussians = False,
    ):

        # TODO: visualize orientation as normal

        neural_pc_o3d = o3d.geometry.PointCloud()

        if query_global:
            neural_points_np = (
                self.neural_points[::random_down_ratio]
                .cpu()
                .detach()
                .numpy()
                .astype(np.float64)
            )
            # points_orientation_np = self.point_orientations[::random_down_ratio].cpu().detach().numpy().astype(np.float64)
        else:
            neural_points_np = (
                self.local_neural_points[::random_down_ratio]
                .cpu()
                .detach()
                .numpy()
                .astype(np.float64)
            )

        if color_mode == 0 and (self.point_colors is not None): # raw color
            if query_global:
                point_colors_np = (
                    self.point_colors[::random_down_ratio]
                    .cpu()
                    .detach()
                    .numpy()
                    .astype(np.float64)
                )
            else:
                point_colors_np = (
                    self.local_point_colors[::random_down_ratio]
                    .cpu()
                    .detach()
                    .numpy()
                    .astype(np.float64)
                )

            neural_pc_o3d.colors = o3d.utility.Vector3dVector(point_colors_np) 

        elif color_mode == 1 and self.geo_feature_pca is not None:  # "geo_feature"               
            if query_global:
                neural_features_vis = self.geo_features[:-1:random_down_ratio]
            else:
                neural_features_vis = self.local_geo_features[
                    :-1:random_down_ratio
                ].detach()

            geo_feature_3d, _ = feature_pca_torch(neural_features_vis, principal_components=self.geo_feature_pca) # [0,1]
            geo_feature_rgb = geo_feature_3d.cpu().numpy().astype(np.float64)
            neural_pc_o3d.colors = o3d.utility.Vector3dVector(geo_feature_rgb)

        elif color_mode == 2 and self.color_feature_pca is not None and self.color_features is not None:  # "color_feature"
            if query_global:
                neural_features_vis = self.color_features[:-1:random_down_ratio]
            else:
                neural_features_vis = self.local_color_features[
                    :-1:random_down_ratio
                ].detach()
            
            color_feature_3d, _ = feature_pca_torch(neural_features_vis, principal_components=self.color_feature_pca) # [0,1]
            color_feature_rgb = color_feature_3d.cpu().numpy().astype(np.float64)
            neural_pc_o3d.colors = o3d.utility.Vector3dVector(color_feature_rgb)

        elif color_mode == 3:  # "ts": # frame number (ts) as the color
            if query_global:
                if self.config.use_mid_ts:
                    show_ts = ((self.point_ts_create + self.point_ts_update) / 2).int()
                else:
                    show_ts = self.point_ts_create
                ts_np = (
                    show_ts[::random_down_ratio]
                    .cpu()
                    .detach()
                    .numpy()
                    .astype(np.float64)
                )
            else:
                ts_np = (
                    self.local_point_ts_update[::random_down_ratio]
                    .cpu()
                    .detach()
                    .numpy()
                    .astype(np.float64)
                )
            ts_np = np.clip(ts_np / self.max_ts, 0.0, 1.0)
            color_map = cm.get_cmap("jet")
            ts_color = color_map(ts_np)[:, :3].astype(np.float64)
            neural_pc_o3d.colors = o3d.utility.Vector3dVector(ts_color)

        elif color_mode == 4:  # "stability" # stability as color
            if query_global:
                certainty_np = (
                    1.0
                    - self.point_certainties[::random_down_ratio]
                    .cpu()
                    .detach()
                    .numpy()
                    .astype(np.float64)
                    / 1000.0
                )
            else:
                certainty_np = (
                    1.0
                    - self.local_point_certainties[::random_down_ratio]
                    .cpu()
                    .detach()
                    .numpy()
                    .astype(np.float64)
                    / 1000.0
                )
            certainty_np = np.clip(certainty_np, 0.0, 1.0) # clip between 0 and 1
            # print(self.local_point_certainties)
            certainty_color = np.repeat(certainty_np.reshape(-1, 1), 3, axis=1)
            neural_pc_o3d.colors = o3d.utility.Vector3dVector(certainty_color)

        elif color_mode == 5:  # "random" # random color
            random_color = np.random.rand(neural_points_np.shape[0], 3).astype(
                np.float64
            )
            neural_pc_o3d.colors = o3d.utility.Vector3dVector(random_color)

        # coordinate
        neural_pc_o3d.points = o3d.utility.Vector3dVector(neural_points_np)

        # if cur_sensor_position is not None and neural_pc_o3d.has_normals():
        #     neural_pc_o3d.orient_normals_towards_camera_location(cur_sensor_position) # np.array

        return neural_pc_o3d

    # prune inactive uncertain neural points
    def prune_map(self, prune_certainty_thre, min_prune_count = 500):

        diff_travel_dist = torch.abs(
            self.travel_dist[self.cur_ts] - self.travel_dist[self.point_ts_update]
        )
        inactive_mask = diff_travel_dist > self.diff_travel_dist_local

        prune_mask = inactive_mask & (
            self.point_certainties < prune_certainty_thre
        )  # True for prune

        prune_count = torch.sum(prune_mask).item()
        if prune_count > min_prune_count:
            if not self.silence:
                print("# Prune neural points: ", prune_count)

            self.neural_points = self.neural_points[~prune_mask]
            self.point_orientations = self.point_orientations[~prune_mask]
            self.point_ts_create = self.point_ts_create[~prune_mask]
            self.point_ts_update = self.point_ts_update[~prune_mask]
            self.point_certainties = self.point_certainties[~prune_mask]

            if self.point_colors is not None:
                self.point_colors = self.point_colors[~prune_mask]
        
            self.valid_color_mask = self.valid_color_mask[~prune_mask]
            self.valid_gs_mask = self.valid_gs_mask[~prune_mask]
            self.free_gs_mask = self.free_gs_mask[~prune_mask]

            # with padding
            prune_mask = torch.cat(
                (prune_mask, torch.tensor([False]).to(prune_mask)), dim=0
            )
            self.geo_features = self.geo_features[~prune_mask]
            if self.config.color_on and self.color_features is not None:
                self.color_features = self.color_features[~prune_mask]
            # recreate hash and local map then
            return True
        return False

    def adjust_map(self, pose_diff_torch):
        """
        For each neural point, use its ts to find the diff between old and new pose, 
        transform the position and rotate the orientation.
        We use the mid_ts for each neural point. 
        This is done after loop closure and PGO.
        """

        self.after_pgo = True

        if self.config.use_mid_ts:
            used_ts = (
                (self.point_ts_create + self.point_ts_update) / 2
            ).int() 
        else:
            used_ts = self.point_ts_create

        self.neural_points = transform_batch_torch(
            self.neural_points, pose_diff_torch[used_ts]
        )

        diff_quat_torch = rotmat_to_quat(pose_diff_torch[:, :3, :3])  # rotation part

        self.point_orientations = quat_multiply(
            diff_quat_torch[used_ts], self.point_orientations
        ).to(self.point_orientations) # [w,x,y,z]


    def recreate_hash(
        self,
        sensor_position: torch.Tensor,
        sensor_orientation: torch.Tensor = None,
        kept_points: bool = True,
        with_ts: bool = True,
        cur_ts=0,
    ):

        cur_resolution = self.resolution

        self.buffer_pt_index = torch.full(
            (self.buffer_size,), -1, dtype=self.idx_dtype, device=self.device
        )  # reset

        # take the point that is closer to the current timestamp (now used)
        # also update the timestep of neural points during merging
        if with_ts:
            if self.config.use_mid_ts:
                ts_used = (
                    (self.point_ts_create + self.point_ts_update) / 2
                ).int()
            else:
                ts_used = self.point_ts_create
            ts_diff = torch.abs(ts_used - cur_ts).float()
            sample_idx = voxel_down_sample_min_value_torch(
                self.neural_points, cur_resolution, ts_diff
            )
        else:
            # take the point that has a larger stability
            sample_idx = voxel_down_sample_min_value_torch(
                self.neural_points,
                cur_resolution,
                self.point_certainties.max() - self.point_certainties,
            )

        if kept_points:
            # don't filter the neural points (keep them, only merge when neccessary, figure out the better merging method later)
            sample_points = self.neural_points[sample_idx]
            grid_coords = (sample_points / cur_resolution).floor().to(self.primes)
            hash_value = torch.fmod(
                (grid_coords * self.primes).sum(-1), int(self.buffer_size)
            )
            self.buffer_pt_index[hash_value] = sample_idx

        else:
            if not self.silence:
                print("Filter duplicated neural points")

            # only kept those filtered
            self.neural_points = self.neural_points[sample_idx]
            self.point_orientations = self.point_orientations[
                sample_idx
            ]  # as quaternion
            self.point_ts_create = self.point_ts_create[sample_idx]
            self.point_ts_update = self.point_ts_update[sample_idx]
            self.point_certainties = self.point_certainties[sample_idx]
            
            if self.point_colors is not None:
                self.point_colors = self.point_colors[sample_idx]

            sample_idx_pad = torch.cat((sample_idx, torch.tensor([-1]).to(sample_idx)))
            # with padding in the end
            self.geo_features = self.geo_features[sample_idx_pad] 
            if self.color_features is not None:
                self.color_features = self.color_features[sample_idx_pad]

            self.valid_color_mask = self.valid_color_mask[sample_idx]
            self.valid_gs_mask = self.valid_gs_mask[sample_idx]
            self.free_gs_mask = self.free_gs_mask[sample_idx]

            new_point_count = self.neural_points.shape[0]

            grid_coords = (self.neural_points / cur_resolution).floor().to(self.primes)
            hash_value = torch.fmod(
                (grid_coords * self.primes).sum(-1), int(self.buffer_size)
            )
            self.buffer_pt_index[hash_value] = torch.arange(
                new_point_count, dtype=self.idx_dtype, device=self.device
            )

        if sensor_position is not None:
            self.reset_local_map(sensor_position, sensor_orientation, cur_ts)

        if not kept_points:  # merged
            self.record_memory(verbose=(not self.silence))  # show the updated memory after merging

    def set_search_neighborhood(
        self, num_nei_cells: int = 1, search_alpha: float = 1.0
    ):

        dx = torch.arange(
            -num_nei_cells,
            num_nei_cells + 1,
            device=self.primes.device,
            dtype=self.primes.dtype,
        )

        coords = torch.meshgrid(dx, dx, dx, indexing="ij")
        dx = torch.stack(coords, dim=-1).reshape(-1, 3)  # [K,3]

        dx2 = torch.sum(dx**2, dim=-1)
        self.neighbor_dx = dx[
            dx2 < (num_nei_cells + search_alpha) ** 2
        ]  # in the sphere --> smaller K --> faster training

        # when num_cells = 3
        # alpha 0.2, K = 147
        # alpha 0.5, K = 179
        # alpha 1.0, K = 251

        # when num_cells = 2
        # alpha 0.2, K = 33
        # alpha 0.3, K = 57
        # alpha 0.5, K = 81
        # alpha 1.0, K = 93
        # alpha 2.0, K = 125

        self.neighbor_K = self.neighbor_dx.shape[0]
        self.max_valid_dist2 = 3 * ((num_nei_cells + 1) * self.resolution) ** 2
        # print(self.neighbor_K)

    def radius_neighborhood_search(
        self, points: torch.Tensor, time_filtering: bool = False
    ):

        # T0 = get_time()
        cur_resolution = self.resolution
        cur_buffer_size = int(self.buffer_size)

        grid_coords = (points / cur_resolution).floor().to(self.primes)  # [N,3]

        neighbord_cells = (
            grid_coords[..., None, :] + self.neighbor_dx
        )  # [N,K,3] # int64

        # T1 = get_time()

        # hash = (neighbord_cells * self.primes).sum(-1) % cur_buffer_size  # [N,K] # no negative number
        hash_value = torch.fmod(
            (neighbord_cells * self.primes).sum(-1), cur_buffer_size
        )  # [N,K] # with negative number (but actually the same)

        # T12 = get_time()

        neighb_idx = self.buffer_pt_index[hash_value]

        # T2 = get_time()

        if time_filtering:  # now is actually travel distance filtering
            diff_travel_dist = torch.abs(
                self.travel_dist[self.cur_ts]
                - self.travel_dist[self.point_ts_create[neighb_idx]]
            )
            local_t_window_mask = diff_travel_dist < self.diff_travel_dist_local
            neighb_idx[~local_t_window_mask] = -1

        # T3 = get_time()

        neighb_pts = self.neural_points[neighb_idx]
        neighb_pts_sub = neighb_pts - points.view(-1, 1, 3)  # [N,K,3]

        dist2 = torch.sum(neighb_pts_sub**2, dim=-1)
        dist2[neighb_idx == -1] = self.max_valid_dist2

        # if the dist is too large (indicating a hash collision), also mask the index as invalid
        neighb_idx[dist2 > self.max_valid_dist2] = -1

        # T4 = get_time()

        # print("time for get neighbor idx:", (T1-T0) * 1e3)  # |
        # # print("time for hashing func    :", (T12-T1) * 1e3)
        # print("time for hashing         :", (T2-T1) * 1e3)  # ||||
        # print("time for time filtering  :", (T3-T2) * 1e3)  # |
        # print("time for distance        :", (T4-T3) * 1e3)  # |||

        return dist2, neighb_idx

    def query_certainty(
        self, query_points: torch.Tensor
    ):  # a faster way to get the certainty at a batch of query points

        _, idx = self.radius_neighborhood_search(query_points)  # only the self voxel

        # idx = self.global2local[0][idx] # [N, K] # get the local idx using the global2local mapping
        # certainty = self.local_hier_certainty[0][idx] # [N, K] # directly global search

        certainty = self.point_certainties[idx]
        certainty[idx < 0] = 0.0

        query_points_certainty = torch.max(certainty, dim=-1)[0]

        # print(query_points_certainty)

        return query_points_certainty

    def gather_local_data(self, with_sorroundings: bool = True):

        """
        Gather the local and sorrounding neural points data, this is then used as the input to the rendering function and our GUI
        """

        neural_points_data = {}
        neural_points_data["position"] = self.local_neural_points

        # print(neural_points_data["position"])

        neural_points_data["orientation"] = self.local_point_orientations
        neural_points_data["color"] = self.local_point_colors
        neural_points_data["geo_feature"] = self.local_geo_features
        neural_points_data["color_feature"] = self.local_color_features
        neural_points_data["resolution"] = self.resolution
        neural_points_data["free_mask"] = self.local_free_gs_mask
        neural_points_data["valid_mask"] = self.local_valid_gs_mask
        neural_points_data["stability"] = self.local_point_certainties

        sorrounding_neural_points_data = None
        if with_sorroundings:
            sorrounding_neural_points_data = {}
            sorrounding_mask = self.sorrounding_mask
            sorrounding_mask_a = sorrounding_mask[:-1]
            sorrounding_neural_points_data["position"] = self.neural_points[sorrounding_mask_a]
            sorrounding_neural_points_data["orientation"] = self.point_orientations[sorrounding_mask_a]
            sorrounding_neural_points_data["geo_feature"] = self.geo_features[sorrounding_mask]
            
            if self.point_colors is not None:
                sorrounding_neural_points_data["color"] = self.point_colors[sorrounding_mask_a]
                sorrounding_neural_points_data["color_feature"] = self.color_features[sorrounding_mask]
            
            sorrounding_neural_points_data["resolution"] = self.resolution
            sorrounding_neural_points_data["free_mask"] = self.free_gs_mask[sorrounding_mask_a] # but now this is actually per neural point
            sorrounding_neural_points_data["valid_mask"] = self.valid_gs_mask[sorrounding_mask_a]
            sorrounding_neural_points_data["stability"] = self.point_certainties[sorrounding_mask_a]

        return neural_points_data, sorrounding_neural_points_data


    def clear_temp(self, clean_more: bool = False):
        """
        Clear the temp data that is not needed
        """
        self.buffer_pt_index = None
        self.local_neural_points = None
        self.local_point_orientations = None
        self.local_geo_features = nn.Parameter()
        self.local_color_features = nn.Parameter()
        self.local_point_certainties = None
        self.local_point_ts_update = None
        self.local_point_colors = None
        
        self.local_valid_color_mask = None
        self.local_valid_gs_mask = None
        self.local_free_gs_mask = None
        
        self.local_mask = None
        self.global2local = None

        # Also only used for debugging, can be removed
        if clean_more:
            self.point_ts_create = None
            self.point_ts_update = None
            self.point_certainties = None
            self.point_colors = None

    def get_map_o3d_bbx(self):
        map_min, _ = torch.min(self.neural_points, dim=0)
        map_max, _ = torch.max(self.neural_points, dim=0)

        # print(map_min)

        o3d_bbx = o3d.geometry.AxisAlignedBoundingBox(
            map_min.cpu().detach().numpy(), map_max.cpu().detach().numpy()
        )

        return o3d_bbx