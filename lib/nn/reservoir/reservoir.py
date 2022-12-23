"""

Code extensively inspired by https://github.com/stefanonardo/pytorch-esn

"""

import numpy as np
import torch
import torch.nn as nn
import torch.sparse
from einops import rearrange
from torch.nn import functional as F
from tsl.nn.utils import get_functional_activation

from lib.utils import self_normalizing_activation


class ReservoirLayer(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 spectral_radius,
                 leaking_rate,
                 bias=True,
                 density=1.,
                 in_scaling=1.,
                 bias_scale=1.,
                 activation='tanh'):
        super(ReservoirLayer, self).__init__()
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

    def forward(self, x, h):
        h_new = self.activation(
            F.linear(x, self.w_ih, self.b_ih) + F.linear(h, self.w_hh))
        h_new = (1 - self.alpha) * h + self.alpha * h_new
        return h_new


class Reservoir(nn.Module):
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
        super(Reservoir, self).__init__()
        self.mode = activation
        self.input_size = input_size
        self.input_scaling = input_scaling
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.leaking_rate = leaking_rate
        self.spectral_radius = spectral_radius
        self.density = density
        self.bias = bias
        self.alpha_decay = alpha_decay

        layers = []
        alpha = leaking_rate
        for i in range(num_layers):
            layers.append(
                ReservoirLayer(
                    input_size=input_size if i == 0 else hidden_size,
                    hidden_size=hidden_size,
                    in_scaling=input_scaling,
                    density=density,
                    activation=activation,
                    spectral_radius=spectral_radius,
                    leaking_rate=alpha
                )
            )
            if self.alpha_decay:
                alpha = np.clip(alpha - 0.1, 0.1, 1.)

        self.reservoir_layers = nn.ModuleList(layers)

    def reset_parameters(self):
        for layer in self.reservoir_layers:
            layer.reset_parameters()

    def forward_prealloc(self, x, h0=None, return_last_state=False):
        # x : b s n f
        *batch_size, steps, nodes, _ = x.size()

        if x.ndim == 4:
            batch_size = x.size(0)
            x = rearrange(x, 'b s n f -> s (b n) f')

        out = torch.empty((steps, x.size(1),
                           len(self.reservoir_layers) * self.hidden_size),
                          dtype=x.dtype, device=x.device)
        out[0] = 0
        size = [slice(i * self.hidden_size, (i + 1) * self.hidden_size)
                for i in range(len(self.reservoir_layers))]
        # for each step, update the reservoir states for all layers
        for s in range(steps):
            # for all layers, observe input and compute updated states
            x_s = x[s]
            for i, layer in enumerate(self.reservoir_layers):
                x_s = layer(x_s, out[s, :, size[i]])
                out[s, :, size[i]] = x_s
        if isinstance(batch_size, int):
            out = rearrange(out, 's (b n) f -> b s n f', b=batch_size, n=nodes)
        if return_last_state:
            return out[:, -1]
        return out

    def forward(self, x, h0=None, return_last_state=False):
        # x : b s n f
        batch_size, steps, nodes, _ = x.size()

        if h0 is None:
            h0 = x.new_zeros(len(self.reservoir_layers), batch_size * nodes,
                             self.hidden_size, requires_grad=False)

        x = rearrange(x, 'b s n f -> s (b n) f')
        out = []
        h = h0
        # for each step, update the reservoir states for all layers
        for s in range(steps):
            h_s = []
            # for all layers, observe input and compute updated states
            x_s = x[s]
            for i, layer in enumerate(self.reservoir_layers):
                x_s = layer(x_s, h[i])
                h_s.append(x_s)
            # update all states
            h = torch.stack(h_s)
            # collect states
            out.append(h)
        out = torch.stack(out)  # [s, l, b, (n), f]
        out = rearrange(out, 's l (b n) f -> b s n (l f)', b=batch_size,
                        n=nodes)
        if return_last_state:
            return out[:, -1]
        return out
