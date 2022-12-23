import torch
from einops import repeat
from torch.nn import functional as F
from tsl.nn.models.stgn import GraphWaveNetModel as GWN


class GraphWaveNetModel(GWN):

    def get_learned_adj(self, node_index=None):
        logits = F.relu(self.source_embeddings(token_index=node_index) @
                        self.target_embeddings(token_index=node_index).T)
        adj = torch.softmax(logits, dim=1)
        return adj

    def forward(self, x, edge_index, edge_weight=None, u=None, node_index=None,
                **kwargs):
        """"""
        # x: [batches, steps, nodes, channels]

        if u is not None:
            if u.dim() == 3:
                u = repeat(u, 'b s c -> b s n c', n=x.size(-2))
            x = torch.cat([x, u], -1)

        if self.receptive_field > x.size(1):
            # pad temporal dimension
            x = F.pad(x, (0, 0, 0, 0, self.receptive_field - x.size(1), 0))

        if len(self.dense_sconvs):
            adj_z = self.get_learned_adj(node_index)

        x = self.input_encoder(x)

        out = torch.zeros(1, x.size(1), 1, 1, device=x.device)
        for i, (tconv, sconv, skip_conn, norm) in enumerate(
                zip(self.tconvs, self.sconvs, self.skip_connections,
                    self.norms)):
            res = x
            # temporal conv
            x = tconv(x)
            # residual connection -> out
            out = skip_conn(x) + out[:, -x.size(1):]
            # spatial conv
            xs = sconv(x, edge_index, edge_weight)
            if len(self.dense_sconvs):
                x = xs + self.dense_sconvs[i](x, adj_z)
            else:
                x = xs
            x = self.dropout(x)
            # residual connection -> next layer
            x = x + res[:, -x.size(1):]
            x = norm(x)

        return self.readout(out)
