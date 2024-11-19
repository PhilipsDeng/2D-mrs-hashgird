import torch
import torch.nn as nn
import torch.nn.functional as F
import tinycudann as tcnn
import numpy as np


class MLP(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden, num_layers, bias=True, dtype=torch.float32):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers

        net = []
        for l in range(num_layers):
            net.append(nn.Linear(self.dim_in if l == 0 else self.dim_hidden,
                                 self.dim_out if l == num_layers - 1 else self.dim_hidden, bias=bias, dtype=dtype))

        self.net = nn.ModuleList(net)

    def forward(self, x):
        for l in range(self.num_layers):
            x = self.net[l](x)
            if l != self.num_layers - 1:
                x = F.relu(x, inplace=True)
            else:
                x = torch.sigmoid(x)
        return x
    
class FusedMLP(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden, num_layers):
        super().__init__()
        

        self.config = {
            "otype": "CutlassMLP",    
            "activation": "ReLU",        
            "output_activation": "Sigmoid", 
            "n_neurons": dim_hidden,            
            "n_hidden_layers": num_layers,
        }


        self.fusedMLP = tcnn.Network(dim_in, dim_out, self.config) 

    def forward(self, inputs):
        return self.fusedMLP(inputs)