import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import tinycudann as tcnn

class HashGrid(nn.Module):
    def __init__(self, in_channels,
        otype, n_levels, n_features_per_level, log2_hashmap_size, base_resolution, # the same as in tinycudann
        max_resolution, # NOTE need to compute per_level_scale ,
        dtype=torch.float32 # half precision might lead to NaN
    ):
        
        super().__init__()

        self.otype = otype
        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.base_resolution = base_resolution
        self.max_resolution = max_resolution
        self.per_level_scale = self.get_per_level_scale()

        self.config = {
            "otype": self.otype,
            "n_levels": self.n_levels,
            "n_features_per_level": self.n_features_per_level,
            "log2_hashmap_size": self.log2_hashmap_size,
            "base_resolution": self.base_resolution,
            "per_level_scale": self.per_level_scale
        }
        self.hashgrid = tcnn.Encoding(in_channels, self.config, dtype=dtype)

    def get_per_level_scale(self):
        return np.power(self.max_resolution / self.base_resolution, 1 / self.n_levels)
    
    def forward(self, inputs):
        return self.hashgrid(inputs)