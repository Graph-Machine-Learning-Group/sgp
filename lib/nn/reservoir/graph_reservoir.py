"""

Code extensively inspired by https://github.com/stefanonardo/pytorch-esn

"""
import numpy as np
import torch
import torch.nn as nn
import torch.sparse
from torch.nn import functional as F
from torch_geometric.nn import MessagePassing
from torch_sparse import matmul
from tsl.nn.blocks.encoders.gcrnn import _GraphRNN
from tsl.nn.utils import get_functional_activation

from lib.utils import self_normalizing_activation


class GESNLayer(MessagePassing):
    def __init__(self,
                 input_size,
                 hidden_size,
                 spectral_radius=0.9,
                 leaking_rate=0.9,
                 bias=True,
                 density=0.9,
                 in_scaling=1.,
                 bias_scale=1.,
                 activation='tanh',
                 aggr='add'):
        super(GESNLayer, self).__init__(aggr=aggr)
        self.w_ih_scale = in_scaling
        self.b_scale = bias_scale
        self.density = density
        self.hidden_size = hidden_size
        self.alpha = leaking_rate
        self.spectral_radius = spectral_radius

        assert activation in ['tanh', 'relu', 'self_norm', 'identity']
        if activation == 'self_norm':
            self.activation = self_normalizing_activation
        else:
            self.activation = get_functional_activation(activation)

        self.w_ih = nn.Parameter(torch.Tensor(hidden_size, input_size),
                                 requires_grad=False)
        self.w_hh = nn.Parameter(torch.Tensor(hidden_size, hidden_size),
                                 requires_grad=False)
        if bias is not None:
            self.b_ih = nn.Parameter(torch.Tensor(hidden_size),
                                     requires_grad=False)
        else:
            self.register_parameter('b_ih', None)
        self.reset_parameters()

    def reset_parameters(self):
        self.w_ih.data.uniform_(-1, 1)
        self.w_ih.data.mul_(self.w_ih_scale)

        if self.b_ih is not None:
            self.b_ih.data.uniform_(-1, 1)
            self.b_ih.data.mul_(self.b_scale)

        # init recurrent weights
        self.w_hh.data.uniform_(-1, 1)

        if self.density < 1:
            n_units = self.hidden_size * self.hidden_size
            mask = self.w_hh.data.new_ones(n_units)
            masked_weights = torch.randperm(n_units)[
                             :int(n_units * (1 - self.density))]
            mask[masked_weights] = 0.
            self.w_hh.data.mul_(mask.view(self.hidden_size, self.hidden_size))

        # adjust spectral radius
        abs_eigs = torch.linalg.eigvals(self.w_hh.data).abs()
        self.w_hh.data.mul_(self.spectral_radius / torch.max(abs_eigs))

    def message(self, x_j, edge_weight):
        return edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t, x):
        return matmul(adj_t, x, reduce=self.aggr)

    def forward(self, x, h, edge_index, edge_weight=None):
        """This layer expects a normalized adjacency matrix either in
        edge_index or SparseTensor layout."""
        h_new = self.activation(F.linear(x, self.w_ih, self.b_ih) +
                                self.propagate(edge_index,
                                               x=F.linear(h, self.w_hh),
                                               edge_weight=edge_weight))
        h_new = (1 - self.alpha) * h + self.alpha * h_new
        return h_new


class GraphESN(_GraphRNN):
    _cat_states_layers = True

    def __init__(self,
                 input_size,
                 hidden_size,
                 input_scaling=1.,
                 num_layers=1,
                 leaking_rate=0.9,
                 spectral_radius=0.9,
                 density=0.9,
                 activation='tanh',
                 bias=True,
                 alpha_decay=False):
        super(GraphESN, self).__init__()
        self.mode = activation
        self.input_size = input_size
        self.input_scaling = input_scaling
        self.hidden_size = hidden_size
        self.n_layers = num_layers
        self.leaking_rate = leaking_rate
        self.spectral_radius = spectral_radius
        self.density = density
        self.bias = bias
        self.alpha_decay = alpha_decay

        layers = []
        alpha = leaking_rate
        for i in range(num_layers):
            layers.append(
                GESNLayer(
                    input_size=input_size if i == 0 else hidden_size,
                    hidden_size=hidden_size,
                    in_scaling=input_scaling,
                    density=density,
                    activation=activation,
                    spectral_radius=spectral_radius,
                    leaking_rate=alpha
                ))
            if self.alpha_decay:
                alpha = np.clip(alpha - 0.1, 0.1, 1.)

        self.rnn_cells = nn.ModuleList(layers)

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.rnn_cells:
            layer.reset_parameters()
