import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class MLP(nn.Module):

    def __init__(self, input_shape, n_layers, n_units):
        super().__init__()
        self._layers = []
        n_in = int(np.prod(np.array(input_shape)))
        for i in range(n_layers):
            layer = nn.Linear(n_in, n_units)
            self.add_module('hidden_layer_{}'.format(i+1), layer)
            n_in = n_units
            self._layers.append(layer)

    def forward(self, x):
        h = x.reshape(x.shape[0], -1)
        for layer in self._layers:
            h = F.relu(layer(h))
        return h


class ReprNetMLP(nn.Module):

    def __init__(self, input_shape, n_layers, n_units, d):
        super().__init__()
        self.mlp = MLP(input_shape, n_layers, n_units)
        if n_layers >= 1:
            n_in = n_units
        else:
            n_in = int(np.prod(np.array(input_shape)))
        self.out_layer = nn.Linear(n_in, d)

    def forward(self, x):
        h = self.mlp(x)
        o = self.out_layer(h)
        return o


class DiscreteQNetMLP(nn.Module):

    def __init__(self, input_shape, n_actions,
            n_layers, n_units):
        super().__init__()
        self.mlp = MLP(input_shape, n_layers, n_units)
        if n_layers >= 1:
            n_in = n_units
        else:
            n_in = int(np.prod(np.array(input_shape)))
        self.out_layer = nn.Linear(n_in, n_actions)

    def forward(self, x):
        h = self.mlp(x)
        o = self.out_layer(h)
        return o

