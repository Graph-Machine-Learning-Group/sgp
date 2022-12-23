from typing import Union, List, Optional

import numpy as np
import torch
from torch import Tensor
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import dropout_adj, to_undirected
from torch_sparse import SparseTensor
from tsl.data import SpatioTemporalDataset
from tsl.utils.python_utils import ensure_list

from .nn.reservoir import Reservoir


def preprocess_dataset(dataset: SpatioTemporalDataset,
                       preprocess_exogenous,
                       reservoir_kwargs,
                       sgp_kwargs, ):
    # if preprocess_exogenous is True, preprocess all exogenous
    if isinstance(preprocess_exogenous, bool):
        preprocess_exogenous = dataset.exogenous.keys() \
            if preprocess_exogenous else []
    preprocess_exogenous = ensure_list(preprocess_exogenous)

    data, _ = dataset.get_tensors(['data'] + preprocess_exogenous,
                                  preprocess=True, cat_dim=-1)

    res = reservoir_preprocessing_(data, **reservoir_kwargs)
    res = sgp_spatial_embedding(res,
                                num_nodes=data.size(1),
                                edge_index=dataset.edge_index,
                                edge_weight=dataset.edge_weight,
                                **sgp_kwargs)

    dataset.add_exogenous('processed_x', torch.cat(res, -1),
                          add_to_input_map=False)
    dataset.set_input_map({'x': ['processed_x']})


def reservoir_preprocessing_(data,
                             hidden_size: int,
                             preprocess_exogenous: Union[bool, List] = False,
                             num_layers=1,
                             leaking_rate=0.9,
                             spectral_radius=0.9,
                             density=0.9,
                             activation='tanh',
                             bias=True,
                             cuda=False):
    reservoir = Reservoir(input_size=data.size(-1),
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          leaking_rate=leaking_rate,
                          spectral_radius=spectral_radius,
                          density=density,
                          activation=activation,
                          bias=bias)

    device = data.device
    if cuda and torch.cuda.is_available():
        data = data.cuda()
        reservoir = reservoir.cuda()

    return reservoir(data[None])[0].to(device)


def preprocess_adj(edge_index: Adj, edge_weight: OptTensor = None,
                   num_nodes: Optional[int] = None,
                   gcn_norm: bool = False,
                   set_diag: bool = True,
                   remove_diag: bool = False) -> SparseTensor:
    # convert numpy to torch
    if isinstance(edge_index, np.ndarray):
        edge_index = torch.from_numpy(edge_index)
        if edge_weight is not None:
            edge_weight = torch.from_numpy(edge_weight)

    if isinstance(edge_index, Tensor):
        # transpose
        col, row = edge_index
        adj = SparseTensor(row=row, col=col, value=edge_weight,
                           sparse_sizes=(num_nodes, num_nodes))
    elif isinstance(edge_index, SparseTensor):
        adj = edge_index
    else:
        raise RuntimeError("Edge index must be (edge_index, edge_weight) tuple "
                           "or SparseTensor.")

    if set_diag:
        adj = adj.set_diag()
    elif remove_diag:
        adj = adj.remove_diag()

    if gcn_norm:
        deg = adj.sum(dim=1).to(torch.float)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        adj = deg_inv_sqrt.view(-1, 1) * adj * deg_inv_sqrt.view(1, -1)
    else:
        deg = adj.sum(dim=1).to(torch.float)
        deg_inv = deg.pow(-1.0)
        deg_inv[deg_inv == float('inf')] = 0
        adj = deg_inv.view(-1, 1) * adj

    return adj


def sgp_spatial_support(edge_index: Adj,
                        edge_weight=None,
                        num_nodes=None,
                        k=2,
                        undirected=False,
                        add_self_loops=False,
                        remove_self_loops=False,
                        bidirectional=False,
                        global_attr=False) -> List[SparseTensor]:
    if not isinstance(edge_index, SparseTensor):
        # transpose
        col, row = edge_index
        adj = SparseTensor(row=row, col=col, value=edge_weight,
                           sparse_sizes=(num_nodes, num_nodes))
    else:
        adj = edge_index
    if undirected:
        adj = adj + adj.t()

    if add_self_loops:
        adj = adj.set_diag()
    elif remove_self_loops:
        adj = adj.remove_diag()

    if undirected:
        deg = adj.sum(dim=1).to(torch.float)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        adj_0 = deg_inv_sqrt.view(-1, 1) * adj * deg_inv_sqrt.view(1, -1)
    else:
        deg = adj.sum(dim=1).to(torch.float)
        deg_inv = deg.pow(-1.0)
        deg_inv[deg_inv == float('inf')] = 0
        adj_0 = deg_inv.view(-1, 1) * adj

    support = [adj_0]
    for _ in range(k - 1):
        support.append(adj_0 @ adj_0)

    if bidirectional:
        support += sgp_spatial_support(edge_index=adj,
                                       k=k,
                                       undirected=False,
                                       add_self_loops=False,
                                       remove_self_loops=False,
                                       bidirectional=False,
                                       global_attr=False)
    if global_attr:
        N = num_nodes if num_nodes is not None else adj.size(0)
        mean_adj = torch.full((N, N), fill_value=1 / N)
        support.append(mean_adj)

    return support


def sgp_spatial_embedding(x,
                          num_nodes,
                          edge_index,
                          edge_weight=None,
                          k=2,
                          undirected=False,
                          add_self_loops=False,
                          remove_self_loops=False,
                          bidirectional=False,
                          one_hot_encoding=False,
                          dropout_rate=0.):
    # x [batch, node, features]

    # subsample operator
    edge_index, edge_weight = dropout_adj(edge_index, edge_weight,
                                          p=dropout_rate,
                                          num_nodes=num_nodes)

    # to undirected
    if undirected:
        assert bidirectional is False
        edge_index, edge_weight = to_undirected(edge_index, edge_weight,
                                                num_nodes)

    # get adj
    adj = preprocess_adj(edge_index, edge_weight,
                         num_nodes=num_nodes,
                         gcn_norm=undirected,
                         set_diag=add_self_loops,
                         remove_diag=remove_self_loops)

    if one_hot_encoding:
        ids = torch.eye(num_nodes, dtype=x.dtype, device=x.device)
        ids = ids.unsqueeze(0).expand(x.size(0), -1, -1)
        x = torch.cat([x, ids], dim=-1)

    # preprocessing of features
    res = [x]
    for _ in range(k):
        x = adj @ x
        res.append(x)

    if bidirectional:
        res_bwd = sgp_spatial_embedding(res[0],
                                        num_nodes,
                                        edge_index[[1, 0]],
                                        edge_weight,
                                        k=k,
                                        undirected=False,
                                        add_self_loops=add_self_loops,
                                        remove_self_loops=remove_self_loops,
                                        bidirectional=False,
                                        one_hot_encoding=False,
                                        dropout_rate=0)
        res += res_bwd[1:]
    return res
