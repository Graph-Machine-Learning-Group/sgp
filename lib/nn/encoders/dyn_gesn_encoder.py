from einops import rearrange
from torch import nn
from torch_geometric.utils import add_self_loops
from torch_sparse import SparseTensor
from tsl.ops.connectivity import normalize
from tsl.utils.parser_utils import ArgParser, str_to_bool

from lib.nn.reservoir.graph_reservoir import GraphESN


class GESNEncoder(nn.Module):
    def __init__(self,
                 input_size,
                 reservoir_size,
                 reservoir_layers,
                 leaking_rate,
                 spectral_radius,
                 density,
                 input_scaling,
                 alpha_decay,
                 reservoir_activation='tanh'
                 ):
        super(GESNEncoder, self).__init__()
        self.reservoir = GraphESN(input_size=input_size,
                                  hidden_size=reservoir_size,
                                  input_scaling=input_scaling,
                                  num_layers=reservoir_layers,
                                  leaking_rate=leaking_rate,
                                  spectral_radius=spectral_radius,
                                  density=density,
                                  activation=reservoir_activation,
                                  alpha_decay=alpha_decay)

    def forward(self, x, edge_index, edge_weight):
        # x : [t n f]
        x = rearrange(x, 't n f -> 1 t n f')
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight)
        if not isinstance(edge_index, SparseTensor):
            _, edge_weight = normalize(edge_index, edge_weight, dim=1)
            col, row = edge_index
            edge_index = SparseTensor(row=row, col=col, value=edge_weight,
                                      sparse_sizes=(x.size(-2), x.size(-2)))
        x, _ = self.reservoir(x, edge_index)
        return x[0]

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
        parser.add_argument('--reservoir-activation', type=str, default='tanh')
        parser.opt_list('--alpha-decay', type=str_to_bool, nargs='?',
                        const=True, default=False)
        return parser
