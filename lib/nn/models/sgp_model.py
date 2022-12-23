import torch
from einops.layers.torch import Rearrange
from torch import nn
from tsl.nn.base import StaticGraphEmbedding
from tsl.nn.blocks.decoders import LinearReadout
from tsl.nn.blocks.encoders import MLP, ResidualMLP
from tsl.nn.functional import expand_then_cat
from tsl.nn.utils import get_layer_activation
from tsl.utils.parser_utils import ArgParser, str_to_bool

from lib.sgp_preprocessing import sgp_spatial_embedding


class SGPModel(nn.Module):
    def __init__(self,
                 input_size,
                 order,
                 n_nodes,
                 hidden_size,
                 mlp_size,
                 output_size,
                 n_layers,
                 horizon,
                 positional_encoding,
                 emb_size=32,
                 exog_size=None,
                 resnet=False,
                 fully_connected=False,
                 dropout=0.,
                 activation='silu'):
        super(SGPModel, self).__init__()

        if fully_connected:
            out_channels = hidden_size
            self.input_encoder = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                get_layer_activation(activation)(),
                nn.Dropout(dropout)
            )
        else:
            out_channels = hidden_size - hidden_size % order
            self.input_encoder = nn.Sequential(
                # [b n f] -> [b 1 n f]
                Rearrange('b n f -> b f n '),
                nn.Conv1d(in_channels=input_size,
                          out_channels=out_channels,
                          kernel_size=1,
                          groups=order),
                Rearrange('b f n -> b n f'),
                get_layer_activation(activation)(),
                nn.Dropout(dropout)
            )

        if resnet:
            self.mlp = ResidualMLP(
                input_size=out_channels,
                hidden_size=mlp_size,
                exog_size=exog_size,
                n_layers=n_layers,
                activation=activation,
                dropout=dropout,
                parametrized_skip=True
            )
        else:
            self.mlp = MLP(
                input_size=out_channels,
                n_layers=n_layers,
                hidden_size=mlp_size,
                exog_size=exog_size,
                activation=activation,
                dropout=dropout
            )

        if positional_encoding:
            self.node_emb = StaticGraphEmbedding(
                n_tokens=n_nodes,
                emb_size=emb_size
            )
            self.lin_emb = nn.Linear(emb_size, out_channels)

        else:
            self.register_parameter('node_emb', None)
            self.register_parameter('lin_emb', None)

        self.readout = LinearReadout(
            input_size=mlp_size,
            output_size=output_size,
            horizon=horizon,
        )

    def forward(self, x, u=None, node_index=None, **kwargs):
        """"""
        # x: [batches steps nodes features]
        x = x[:, -1] if x.ndim == 4 else x
        x = self.input_encoder(x)
        if self.node_emb is not None:
            x = x + self.lin_emb(self.node_emb(token_index=node_index))
        if u is not None:
            u = u[:, -1] if u.ndim == 4 else u
            x = expand_then_cat([x, u], dim=-1)
        x = self.mlp(x)

        return self.readout(x)

    @staticmethod
    def add_model_specific_args(parser: ArgParser):
        parser.opt_list('--hidden-size', type=int, default=32, tunable=True,
                        options=[16, 32, 64, 128, 256])
        parser.opt_list('--mlp-size', type=int, default=32, tunable=True,
                        options=[16, 32, 64, 128, 256])
        parser.opt_list('--emb-size', type=int, default=32, tunable=True,
                        options=[16, 32, 64])
        parser.opt_list('--n-layers', type=int, default=1, tunable=True,
                        options=[1, 2, 3])
        parser.opt_list('--dropout', type=float, default=0., tunable=True,
                        options=[0., 0.2, 0.3])
        parser.opt_list('--fully-connected', type=str_to_bool, nargs='?',
                        const=True, default=False)
        parser.opt_list('--positional-encoding', type=str_to_bool, nargs='?',
                        const=True, default=False)
        parser.opt_list('--resnet', type=str_to_bool, nargs='?', const=True,
                        default=False)
        return parser


class OnlineSGPModel(SGPModel):

    def __init__(self, input_size, output_size,
                 n_nodes,
                 horizon,
                 hidden_size=128,
                 mlp_size=64,
                 n_layers=1,
                 positional_encoding=True,
                 exog_size=None,
                 resnet=False,
                 fully_connected=False,
                 dropout=0.,
                 activation='silu',
                 # sgp spatial embedding params
                 receptive_field=3,
                 reservoir_layers=1,
                 bidirectional=True,
                 undirected=False,
                 add_self_loops=False):
        self.receptive_field = receptive_field
        self.bidirectional = bidirectional
        self.undirected = undirected
        self.add_self_loops = add_self_loops
        order = 1 + (2 if bidirectional else 1) * receptive_field
        # order *= reservoir_layers
        self.order = order
        super(OnlineSGPModel, self).__init__(
            input_size=input_size * order,
            order=order * reservoir_layers,
            n_nodes=n_nodes,
            hidden_size=hidden_size,
            mlp_size=mlp_size,
            output_size=output_size,
            n_layers=n_layers,
            horizon=horizon,
            positional_encoding=positional_encoding,
            exog_size=exog_size,
            resnet=resnet,
            fully_connected=fully_connected,
            dropout=dropout,
            activation=activation)

    def forward(self, x, u=None, edge_index=None, edge_weight=None, **kwargs):
        """"""
        # x: [batches steps nodes features]
        x = x[:, -1]
        x = sgp_spatial_embedding(x, num_nodes=x.size(1),
                                  edge_index=edge_index,
                                  edge_weight=edge_weight,
                                  k=self.receptive_field,
                                  bidirectional=self.bidirectional,
                                  undirected=self.undirected,
                                  add_self_loops=self.add_self_loops)
        x = torch.cat(x, -1)
        return super(OnlineSGPModel, self).forward(x=x, u=u, **kwargs)

    @staticmethod
    def add_model_specific_args(parser: ArgParser):
        parser = SGPModel.add_model_specific_args(parser)
        parser.opt_list('--receptive-field', type=int, default=1, tunable=True,
                        options=[1, 2, 3])
        parser.opt_list('--bidirectional', type=str_to_bool, nargs='?',
                        const=True, default=False)
        parser.opt_list('--undirected', type=str_to_bool, nargs='?', const=True,
                        default=False)
        parser.opt_list('--add-self-loops', type=str_to_bool, nargs='?',
                        const=True, default=False)
        return parser
