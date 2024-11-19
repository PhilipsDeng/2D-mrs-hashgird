import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.modules import MLP, FusedMLP

class Renderer(nn.Module):
    def hash_renderer(config):
        renderer = MLP(
            dim_in=config['dim_in'],
            dim_out=config['dim_out'],
            dim_hidden=config['dim_hidden'],
            num_layers=config['num_layers'],
            dtype=torch.float32 

        )
        return renderer
    
    def hash_fusedrenderer(config):
        renderer = FusedMLP(
            dim_in=config['dim_in'],
            dim_out=config['dim_out'],
            dim_hidden=config['dim_hidden'],
            num_layers=config['num_layers'],
        )
        return renderer