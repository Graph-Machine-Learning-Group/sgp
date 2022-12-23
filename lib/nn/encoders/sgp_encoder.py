from einops import rearrange
from torch import nn
from tsl.utils.parser_utils import ArgParser, str_to_bool

from lib.nn.encoders.sgp_spatial_encoder import SGPSpatialEncoder
from lib.nn.reservoir import Reservoir


class SGPEncoder(nn.Module):
    def __init__(self,
                 input_size,
                 reservoir_size,
                 reservoir_layers,
                 leaking_rate,
                 spectral_radius,
                 density,
                 input_scaling,
                 receptive_field,
                 bidirectional,
                 alpha_decay,
                 global_attr,
                 add_self_loops=False,
                 undirected=False,
                 reservoir_activation='tanh'
                 ):
        super(SGPEncoder, self).__init__()
        self.reservoir = Reservoir(input_size=input_size,
                                   hidden_size=reservoir_size,
                                   input_scaling=input_scaling,
                                   num_layers=reservoir_layers,
                                   leaking_rate=leaking_rate,
                                   spectral_radius=spectral_radius,
                                   density=density,
                                   activation=reservoir_activation,
                                   alpha_decay=alpha_decay)

        self.sgp_encoder = SGPSpatialEncoder(
            receptive_field=receptive_field,
            bidirectional=bidirectional,
            undirected=undirected,
            add_self_loops=add_self_loops,
            global_attr=global_attr
        )

    def forward(self, x, edge_index, edge_weight):
        # x : [t n f]
        x = rearrange(x, 't n f -> 1 t n f')
        x = self.reservoir(x)
        x = x[0]
        x = self.sgp_encoder(x, edge_index, edge_weight)
        return x

    @staticmethod
    def add_model_specific_args(parser: ArgParser):
        parser.opt_list('--reservoir-size', type=int, default=32, tunable=True,
                        options=[16, 32, 64, 128, 256])
        parser.opt_list('--reservoir-layers', type=int, default=1, tunable=True,
                        options=[1, 2, 3])
        parser.opt_list('--receptive-field', type=int, default=1, tunable=True,
                        options=[1, 2, 3])
        parser.opt_list('--spectral-radius', type=float, default=0.9,
                        tunable=True, options=[0.7, 0.8, 0.9])
        parser.opt_list('--leaking-rate', type=float, default=0.9, tunable=True,
                        options=[0.7, 0.8, 0.9])
        parser.opt_list('--density', type=float, default=0.7, tunable=True,
                        options=[0.7, 0.8, 0.9])
        parser.opt_list('--input-scaling', type=float, default=1., tunable=True,
                        options=[1., 1.5, 2.])
        parser.opt_list('--bidirectional', type=str_to_bool, nargs='?',
                        const=True, default=False)
        parser.opt_list('--undirected', type=str_to_bool, nargs='?', const=True,
                        default=False)
        parser.opt_list('--add-self-loops', type=str_to_bool, nargs='?',
                        const=True, default=False)
        parser.opt_list('--alpha-decay', type=str_to_bool, nargs='?',
                        const=True, default=False)
        parser.opt_list('--global-attr', type=str_to_bool, nargs='?',
                        const=True, default=False)
        parser.add_argument('--reservoir-activation', type=str, default='tanh')
        return parser
