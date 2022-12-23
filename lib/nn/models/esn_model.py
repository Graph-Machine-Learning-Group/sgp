from torch import nn
from tsl.nn.blocks.decoders import LinearReadout
from tsl.nn.utils.utils import maybe_cat_exog
from tsl.utils.parser_utils import ArgParser

from lib.nn.reservoir import Reservoir


class ESNModel(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size,
                 exog_size,
                 rec_layers,
                 horizon,
                 activation='tanh',
                 spectral_radius=0.9,
                 leaking_rate=0.9,
                 density=0.7):
        super(ESNModel, self).__init__()

        self.reservoir = Reservoir(input_size=input_size + exog_size,
                                   hidden_size=hidden_size,
                                   num_layers=rec_layers,
                                   leaking_rate=leaking_rate,
                                   spectral_radius=spectral_radius,
                                   density=density,
                                   activation=activation)

        self.readout = LinearReadout(
            input_size=hidden_size * rec_layers,
            output_size=output_size,
            horizon=horizon,
        )

    def forward(self, x, u=None, **kwargs):
        """"""
        # x: [batches steps nodes features]
        # u: [batches steps (nodes) features]
        x = maybe_cat_exog(x, u)

        x = self.reservoir(x, return_last_state=True)

        return self.readout(x)

    @staticmethod
    def add_model_specific_args(parser: ArgParser):
        parser.opt_list('--hidden-size', type=int, default=32, tunable=True,
                        options=[16, 32, 64, 128, 256])
        parser.opt_list('--rec-layers', type=int, default=1, tunable=True,
                        options=[1, 2, 3])
        parser.opt_list('--spectral-radius', type=float, default=0.9,
                        tunable=True, options=[0.7, 0.8, 0.9])
        parser.opt_list('--leaking-rate', type=float, default=0.9, tunable=True,
                        options=[0.7, 0.8, 0.9])
        parser.opt_list('--density', type=float, default=0.7, tunable=True,
                        options=[0.7, 0.8, 0.9])
        return parser
