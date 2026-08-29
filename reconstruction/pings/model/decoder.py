#!/usr/bin/env python3
# @file      decoder.py
# @author    Yue Pan     [yue.pan@igg.uni-bonn.de]
# Copyright (c) 2024 Yue Pan, all rights reserved

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.config import Config


class Decoder(nn.Module):
    def __init__(
        self,
        config: Config,
        input_feature_dim,
        hidden_dim,
        hidden_level,
        out_dim,
        out_k=1, 
        pos_dim=3, # for gs decoder, pos_dim=0
        is_time_conditioned=False,
    ):

        super().__init__()

        self.mlp_out_dim = out_dim * out_k
        self.out_dim_per = out_dim
        self.out_k = out_k # for gs, this denotes the gaussian count per neural point
        self.use_leaky_relu = config.mlp_leaky_relu
        self.bs = config.infer_bs # bs or infer_bs, for batch mlp

        bias_on = config.mlp_bias_on

        input_dim = input_feature_dim + pos_dim

        # default not used
        if is_time_conditioned:
            input_dim += 1

        # predict sdf (now it anyway only predict sdf without further sigmoid
        # Initializa the structure of shared MLP
        layers = []
        for i in range(hidden_level):
            if i == 0:
                layers.append(nn.Linear(input_dim, hidden_dim, bias_on))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim, bias_on))
        self.layers = nn.ModuleList(layers)
        self.lout = nn.Linear(hidden_dim, self.mlp_out_dim, bias_on)

        self.sdf_scale = 1.0
        if config.main_loss_type == "bce":
            self.sdf_scale = config.logistic_gaussian_ratio * config.sigma_sigmoid_m

        self.to(config.device)
        # torch.cuda.empty_cache()

    def mlp(self, features):
        # linear (feature_dim -> hidden_dim)
        # relu
        # linear (hidden_dim -> hidden_dim)
        # relu
        # linear (hidden_dim -> 1)
        for k, l in enumerate(self.layers):
            if k == 0:
                if self.use_leaky_relu:
                    h = F.leaky_relu(l(features))
                else:
                    h = F.relu(l(features))
            else:
                if self.use_leaky_relu:
                    h = F.leaky_relu(l(h))
                else:
                    h = F.relu(l(h))
        out = self.lout(h)
        # no relu, for last linear one

        return out

    def mlp_batch(self, features):

        count = features.shape[0]
        iter_n = math.ceil(count / self.bs)
        
        out = torch.zeros(count, self.mlp_out_dim).to(features)

        for n in range(iter_n):
            head = n * self.bs
            tail = min((n + 1) * self.bs, count)
            batch_features = features[head:tail, :]
            batch_out = self.mlp(batch_features)
            out[head:tail,:] = batch_out

        return out

    # predict the sdf (opposite sign to the actual sdf)
    # unit is already m
    def sdf(self, features):
        out = self.mlp(features).squeeze(1) * self.sdf_scale
        return out
    
    # deprecated
    def time_conditionded_sdf(self, features, ts):
        nn_k = features.shape[1]
        ts_nn_k = ts.repeat(nn_k).view(-1, nn_k, 1)
        time_conditioned_feature = torch.cat((features, ts_nn_k), dim=-1)
        out = self.sdf(time_conditioned_feature)
        return out

    # predict the occupancy probability
    def occupancy(self, features):
        out = torch.sigmoid(self.sdf(features) / -self.sdf_scale)  # to [0, 1]
        return out

    # predict the probabilty of each semantic label
    def sem_label_prob(self, features):
        out = F.log_softmax(self.mlp(features), dim=-1)
        return out

    def sem_label(self, features):
        out = torch.argmax(self.sem_label_prob(features), dim=1)
        return out

    # deprecated
    # def regress_color(self, features):
    #     out = torch.clamp(self.mlp(features), 0.0, 1.0) # clamp result
    #     return out

    def regress_color(self, features):
        out = torch.sigmoid(self.mlp(features)) # sigmoid map to [0,1]
        return out