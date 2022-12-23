import torch
from torch import nn
from tsl.utils.parser_utils import ArgParser, str_to_bool

from lib.sgp_preprocessing import sgp_spatial_embedding


class SGPSpatialEncoder(nn.Module):
    def __init__(self,
                 receptive_field,
                 bidirectional,
                 undirected,
                 global_attr,
                 add_self_loops=False):
        super(SGPSpatialEncoder, self).__init__()
        self.receptive_field = receptive_field
        self.bidirectional = bidirectional
        self.undirected = undirected
        self.add_self_loops = add_self_loops
        self.global_attr = global_attr

    def forward(self, x, edge_index, edge_weight):
        num_nodes = x.size(-2)
        out = sgp_spatial_embedding(x,
                                    num_nodes=num_nodes,
                                    edge_index=edge_index,
                                    edge_weight=edge_weight,
                                    k=self.receptive_field,
                                    bidirectional=self.bidirectional,
                                    undirected=self.undirected,
                                    add_self_loops=self.add_self_loops)
        if self.global_attr:
            g = torch.ones_like(x) * x.mean(-2, keepdim=True)
            out.append(g)
        return torch.cat(out, -1)

    @staticmethod
    def add_model_specific_args(parser: ArgParser):
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
