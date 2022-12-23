from einops import rearrange
from torch import nn, Tensor
from tsl.utils.parser_utils import ArgParser, str_to_bool

from lib.nn.reservoir import Reservoir


class SGPTemporalEncoder(nn.Module):
    def __init__(self, input_size,
                 reservoir_size=32,
                 reservoir_layers=1,
                 leaking_rate=0.9,
                 spectral_radius=0.9,
                 density=0.7,
                 input_scaling=1.,
                 alpha_decay=False,
                 reservoir_activation='tanh'):
        super(SGPTemporalEncoder, self).__init__()
        self.reservoir = Reservoir(input_size=input_size,
                                   hidden_size=reservoir_size,
                                   input_scaling=input_scaling,
                                   num_layers=reservoir_layers,
                                   leaking_rate=leaking_rate,
                                   spectral_radius=spectral_radius,
                                   density=density,
                                   activation=reservoir_activation,
                                   alpha_decay=alpha_decay)

    def forward(self, x: Tensor, *args, **kwargs):
        # x : [t n f]
        x = rearrange(x, 't n f -> 1 t n f')
        x = self.reservoir(x)
        x = x[0]
        return x

    @staticmethod
    def add_model_specific_args(parser: ArgParser):
        parser.opt_list('--reservoir-size', type=int, default=32, tunable=True,
                        options=[16, 32, 64, 128, 256])
        parser.opt_list('--reservoir-layers', type=int, default=1, tunable=True,
                        options=[1, 2, 3])
        parser.opt_list('--spectral-radius', type=float, default=0.9,
                        tunable=True, options=[0.7, 0.8, 0.9])
        parser.opt_list('--leaking-rate', type=float, default=0.9, tunable=True,
                        options=[0.7, 0.8, 0.9])
        parser.opt_list('--density', type=float, default=0.7, tunable=True,
                        options=[0.7, 0.8, 0.9])
        parser.opt_list('--input-scaling', type=float, default=1., tunable=True,
                        options=[1., 1.5, 2.])
        parser.opt_list('--alpha-decay', type=str_to_bool, nargs='?',
                        const=True, default=False)
        parser.add_argument('--reservoir-activation', type=str, default='tanh')
        # for sgp spatial preprocessing
        parser.opt_list('--receptive-field', type=int, default=1, tunable=True,
                        options=[1, 2, 3])
        parser.opt_list('--bidirectional', type=str_to_bool, nargs='?',
                        const=True, default=False)
        parser.opt_list('--undirected', type=str_to_bool, nargs='?', const=True,
                        default=False)
        parser.opt_list('--add-self-loops', type=str_to_bool, nargs='?',
                        const=True, default=False)
        parser.opt_list('--global-attr', type=str_to_bool, nargs='?',
                        const=True, default=False)
        return parser
