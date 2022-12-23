import torch
from einops import rearrange
from torch import nn
from tsl.nn.base import StaticGraphEmbedding
from tsl.nn.blocks.decoders import MLPDecoder
from tsl.nn.blocks.encoders import MLP
from tsl.utils.parser_utils import ArgParser

from lib.sgp_preprocessing import sgp_spatial_embedding


class SGPOnlineModel(nn.Module):
    def __init__(self,
                 input_size,
                 n_nodes,
                 hidden_size,
                 output_size,
                 n_layers,
                 window,
                 horizon,
                 k=2,
                 bidirectional=True,
                 dropout=0.,
                 activation='relu'):
        super(SGPOnlineModel, self).__init__()

        n_features = input_size * window
        input_size = n_features * (1 + k * (int(bidirectional) + 1))
        self.k = k
        self.bidirectional = bidirectional

        self.mlp = MLP(
            input_size=input_size,
            n_layers=n_layers,
            hidden_size=hidden_size,
            activation=activation,
            dropout=dropout
        )

        self.node_emb = StaticGraphEmbedding(
            n_tokens=n_nodes,
            emb_size=hidden_size
        )

        self.readout = MLPDecoder(
            input_size=hidden_size,
            hidden_size=hidden_size,
            output_size=output_size,
            horizon=horizon,
        )

    def forward(self, x, edge_index, edge_weight, **kwargs):
        """"""
        # x: [batches steps nodes features]
        x = rearrange(x, 'b t n f -> b n (t f)')

        x = sgp_spatial_embedding(x, num_nodes=x.size(1),
                                  edge_index=edge_index,
                                  edge_weight=edge_weight,
                                  k=self.k, bidirectional=self.bidirectional)
        x = torch.cat(x, -1)

        x = self.mlp(x) + self.node_emb()

        return self.readout(x)

    @staticmethod
    def add_model_specific_args(parser: ArgParser):
        parser.opt_list('--hidden-size', type=int, default=32, tunable=True,
                        options=[16, 32, 64, 128, 256])
        parser.opt_list('--n-layers', type=int, default=1, tunable=True,
                        options=[1, 2, 3])
        parser.opt_list('--dropout', type=float, default=0., tunable=True,
                        options=[0., 0.2, 0.3])
        return parser
