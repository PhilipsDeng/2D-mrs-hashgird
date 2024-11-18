import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import tinycudann as tcnn

class HashGrid(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.n_levels = config['n_levels']
        self.n_features_per_level = config['n_features_per_level']
        self.log2_hashmap_size = config['log2_hashmap_size']
        self.base_resolution = config['base_resolution']
        self.max_resolution = config['max_resolution']
        self.dtype = torch.float32 if config['dtype'] == 'full' else torch.float16
        self.per_level_scale = self.get_per_level_scale()

        self.config = {
            "otype": "HashGrid",
            "n_levels": self.n_levels,
            "n_features_per_level": self.n_features_per_level,
            "log2_hashmap_size": self.log2_hashmap_size,
            "base_resolution": self.base_resolution,
            "per_level_scale": self.per_level_scale

        }

        self.hashgrid = tcnn.Encoding(2, self.config, dtype=self.dtype) #2是坐标(x,y)的维度

    def get_per_level_scale(self):
        return np.power(self.max_resolution / self.base_resolution, 1 / self.n_levels)
    
    def forward(self, inputs):
        return self.hashgrid(inputs)


