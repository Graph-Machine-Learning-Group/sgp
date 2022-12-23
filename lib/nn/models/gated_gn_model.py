import math

import torch
from einops import rearrange
from torch import nn
from tsl.nn.models.stgn import GatedGraphNetworkModel
from tsl.nn.utils import utils
from tsl.utils.parser_utils import ArgParser
from tsl.utils.parser_utils import str_to_bool


class Conv1dResidual(nn.Module):
    def __init__(self, in_channels: int,
                 hidden_size: int = None,
                 activation=nn.SiLU()):
        super(Conv1dResidual, self).__init__()
        hidden_size = hidden_size or int(in_channels // 2)
        self.hidden_size = hidden_size
        self.conv1 = torch.nn.Conv1d(in_channels, hidden_size, kernel_size=1)
        self.conv2 = torch.nn.Conv1d(hidden_size, in_channels, kernel_size=1)
        self.activation = activation

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs
        x = self.activation(self.conv1(x))
        x = self.conv2(x)
        return inputs + x


class CNNResidual(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 input_window_size: int,
                 hidden_size: int = 64,
                 max_hidden_size: int = 256,
                 kernel_size: int = 5):
        super().__init__()
        self.n_layers = math.ceil(math.log(input_window_size, kernel_size))

        padding = int((-input_window_size) % kernel_size)
        layers = [nn.Sequential(
            nn.ConstantPad1d((padding, 0), 0),
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=hidden_size,
                kernel_size=kernel_size,
                stride=kernel_size,
            ),
            Conv1dResidual(in_channels=hidden_size)
        )]

        current_length = int((input_window_size + padding) / kernel_size)
        for i in range(1, self.n_layers):
            nf_prev = hidden_size
            hidden_size = min(hidden_size * 2, max_hidden_size)
            padding = int((-current_length) % kernel_size)
            layers.append(nn.Sequential(
                nn.ConstantPad1d((padding, 0), 0),
                nn.Conv1d(in_channels=nf_prev,
                          out_channels=hidden_size,
                          kernel_size=kernel_size,
                          stride=kernel_size),
                Conv1dResidual(in_channels=hidden_size)
            ))
            current_length = int((current_length + padding) / kernel_size)

        self.encoder = nn.Sequential(*layers)

        if hidden_size * current_length != out_channels:
            self.lin_out = nn.Linear(hidden_size * current_length, out_channels)
        else:
            self.lin_out = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [b f s] -> [b f]
        bs = x.size(0)
        x = self.encoder(x)
        x = x.view(bs, -1)
        if self.lin_out is not None:
            x = self.lin_out(x)
        return x


class GatedGraphNetworkMLPModel(GatedGraphNetworkModel):

    def __init__(self, input_size,
                 input_window_size,
                 hidden_size,
                 output_size,
                 horizon,
                 n_nodes,
                 exog_size,
                 enc_layers,
                 gnn_layers,
                 full_graph,
                 positional_encoding=True,
                 activation='silu'):
        super(GatedGraphNetworkMLPModel, self).__init__(input_size,
                                                        input_window_size,
                                                        hidden_size,
                                                        output_size,
                                                        horizon,
                                                        n_nodes,
                                                        exog_size,
                                                        enc_layers,
                                                        gnn_layers,
                                                        full_graph,
                                                        activation=activation)
        if not positional_encoding:
            del self.emb
            self.register_parameter('emb', None)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batches steps nodes features]
        x = rearrange(x[:, -self.input_window_size:], 'b s n f -> b n (s f)')

        x = self.input_encoder(x)
        for layer in self.encoder_layers:
            x = layer(x) + x

        return x  # [batches nodes features]

    def forward(self, x, edge_index=None, u=None, node_index=None, **kwargs):
        """"""
        # x: [batches steps nodes features]
        x = utils.maybe_cat_exog(x, u)

        if self.full_graph or edge_index is None:
            num_nodes = x.size(-2)
            nodes = torch.arange(num_nodes, device=x.device)
            edge_index = torch.cartesian_prod(nodes, nodes).T

        x = self.encode(x)

        # add encoding
        if self.emb is not None:
            x = x + self.emb(token_index=node_index)

        for layer in self.gcn_layers:
            x = layer(x, edge_index)

        x = self.decoder(x) + x

        return self.readout(x)

    @staticmethod
    def add_model_specific_args(parser: ArgParser):
        parser.opt_list('--hidden-size', type=int, default=64, tunable=True,
                        options=[16, 32, 64, 128, 256])
        parser.opt_list('--enc-layers', type=int, default=2, tunable=True,
                        options=[1, 2, 3])
        parser.opt_list('--gnn-layers', type=int, default=2, tunable=True,
                        options=[1, 2, 3])
        parser.opt_list('--full-graph', type=str_to_bool, nargs='?', const=True,
                        default=False)
        parser.opt_list('--activation', type=str, default='silu', tunable=False,
                        options=['relu', 'elu', 'silu'])
        parser.add_argument('--positional-encoding', type=str_to_bool,
                            nargs='?', const=True, default=True)
        return parser


class GatedGraphNetworkConvModel(GatedGraphNetworkMLPModel):
    def __init__(self, input_size,
                 input_window_size,
                 hidden_size,
                 output_size,
                 horizon,
                 n_nodes,
                 exog_size,
                 enc_layers,
                 gnn_layers,
                 full_graph,
                 activation='silu'):
        super(GatedGraphNetworkConvModel, self).__init__(input_size,
                                                         input_window_size,
                                                         hidden_size,
                                                         output_size,
                                                         horizon,
                                                         n_nodes,
                                                         exog_size,
                                                         enc_layers,
                                                         gnn_layers,
                                                         full_graph,
                                                         activation=activation)
        del self.encoder_layers, self.input_encoder
        self.encoder = CNNResidual(input_size + exog_size, hidden_size,
                                   input_window_size=input_window_size,
                                   hidden_size=hidden_size)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batches steps nodes features]
        b = x.size(0)
        x = rearrange(x[:, -self.input_window_size:], 'b s n f -> (b n) f s')
        x = self.encoder(x)
        x = rearrange(x, '(b n) f -> b n f', b=b)
        return x  # [batches nodes features]
