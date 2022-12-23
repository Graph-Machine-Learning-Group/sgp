from typing import List, Optional

import torch
from numpy.random import choice
from torch.utils import data
from torch_geometric.utils import k_hop_subgraph
from torch_sparse import SparseTensor
from tsl.data import Batch
from tsl.data import static_graph_collate, Data, SpatioTemporalDataset
from tsl.ops.connectivity import weighted_degree


def subgraph_collate(data_list: List[Data], node_index,
                     roots=None, node_map=None, cls=None):
    elem = data_list[0]
    # get node axis for each key with node dimension
    node_dims = {k: pattern.split(' ').index('n')
                 for k, pattern in elem.pattern.items()
                 if 'n' in pattern}
    # node-wise scalers (with non-dummy node dimension) must be collated
    # together, stacking over the batch dimension
    node_wise = [k for k, trans in elem.transform.items()
                 if k in node_dims and trans.bias.size(node_dims[k]) > 1]
    if roots is None:
        roots = node_index
    # subsample every item in batch
    for sample in data_list:
        # slice every tensor with node dimension
        for k, dim in node_dims.items():
            # subsample
            if k in elem.target or k == 'mask':
                sample[k] = sample[k].index_select(dim, roots)
            else:
                sample[k] = sample[k].index_select(dim, node_index)
            # remove dropped nodes from scaler's parameters
            if k in node_wise:
                trans = sample.transform[k]
                trans.bias = trans.bias.index_select(dim, node_index)
                trans.scale = trans.scale.index_select(dim, node_index)
    # collate tensors in batch
    batch = static_graph_collate(data_list, cls)
    # collate every node-wise scaler (otherwise only first one is kept)
    for k in node_wise:
        scaler = batch.transform[k]
        scaler.bias = torch.stack([d.transform[k].bias for d in data_list])
        scaler.scale = torch.stack([d.transform[k].scale for d in data_list])
    batch.input.node_index = node_index  # index of nodes in subgraph
    if node_map is not None:
        batch.input.target_nodes = node_map  # index of roots in subgraph
    return batch


class SubsetLoader(data.DataLoader):

    def __init__(self, dataset: SpatioTemporalDataset,
                 max_nodes: Optional[int] = None,
                 batch_size: Optional[int] = 1,
                 shuffle: bool = False,
                 num_workers: int = 0,
                 **kwargs):
        if 'collate_fn' in kwargs:
            del kwargs['collate_fn']
        self.max_nodes = max_nodes
        super().__init__(dataset,
                         shuffle=shuffle,
                         batch_size=batch_size,
                         num_workers=num_workers,
                         collate_fn=self.collate,
                         **kwargs)

    def collate(self, data_list: List[Data]):
        batch_nodes = data_list[0].num_nodes
        # initalize buffers
        node_index, node_wise = [], []
        if self.max_nodes is not None and batch_nodes > self.max_nodes:
            # get node axis for each key with node dimension
            node_dims = {k: pattern.split(' ').index('n')
                         for k, pattern in data_list[0].pattern.items()
                         if 'n' in pattern}
            # node-wise scalers (with non-dummy node dimension) must be collated
            # together, stacking over the batch dimension
            node_wise = [k for k, trans in data_list[0].transform.items()
                         if
                         k in node_dims and trans.bias.size(node_dims[k]) > 1]
            # subsample every item in batch
            for sample in data_list:
                # sample different max_nodes nodes at random for each sample
                node_idx = torch.randperm(batch_nodes)[:self.max_nodes]
                node_index.append(node_idx)
                # slice every tensor with node dimension
                for k, dim in node_dims.items():
                    # subsample
                    sample[k] = sample[k].index_select(dim, node_idx)
                    # remove dropped nodes from scaler's parameters
                    if k in node_wise:
                        trans = sample.transform[k]
                        trans.bias = trans.bias.index_select(dim, node_idx)
                        trans.scale = trans.scale.index_select(dim, node_idx)
        # collate tensors in batch
        batch = static_graph_collate(data_list, Batch)
        # collate every node-wise scaler (otherwise only first one is kept)
        for k in node_wise:
            scaler = batch.transform[k]
            scaler.bias = torch.stack([d.transform[k].bias for d in data_list])
            scaler.scale = torch.stack(
                [d.transform[k].scale for d in data_list])

        # subset sampler can only be used over set of nodes (without edges)
        if 'edge_index' in batch:
            del batch['edge_index']
        if 'edge_weight' in batch:
            del batch['edge_weight']

        # add node_index to batch input
        if len(node_index):
            batch.input['node_index'] = torch.stack(node_index)
        batch.__dict__['batch_size'] = len(data_list)

        return batch


class SubgraphLoader(data.DataLoader):

    def __init__(self, dataset: SpatioTemporalDataset,
                 k: int, num_nodes: Optional[int] = None,
                 max_edges: Optional[int] = None,
                 cut_edges_uniformly: bool = False,
                 batch_size: Optional[int] = 1,
                 shuffle: bool = False,
                 num_workers: int = 0,
                 **kwargs):
        if 'collate_fn' in kwargs:
            del kwargs['collate_fn']
        self.k = k
        self.num_nodes = num_nodes
        self.max_edges = max_edges
        self.cut_edges_uniformly = cut_edges_uniformly
        super().__init__(dataset,
                         shuffle=shuffle,
                         batch_size=batch_size,
                         num_workers=num_workers,
                         collate_fn=self.collate,
                         **kwargs)

    def collate(self, data_list: List[Data]):
        elem = data_list[0]
        adj, edge_weight = elem.edge_index, elem.edge_weight

        if isinstance(adj, SparseTensor):
            row, col, edge_weight = adj.t().coo()
            edge_index = torch.stack([row, col], dim=0)
        else:
            edge_index = adj

        N = elem.num_nodes
        # cap nodes
        if self.num_nodes is not None and N > self.num_nodes:
            roots = torch.randperm(N)[:self.num_nodes]
            subgraph = k_hop_subgraph(roots, self.k, edge_index,
                                      relabel_nodes=True,
                                      num_nodes=N, flow='target_to_source')
            node_idx, edge_index, node_map, edge_mask = subgraph
            if edge_weight is not None:
                edge_weight = edge_weight[edge_mask]
            N = len(node_idx)
            batch = subgraph_collate(data_list, node_idx, roots=roots,
                                     node_map=node_map, cls=Batch)
        else:
            batch = static_graph_collate(data_list, Batch)

        # cap edges limit
        if self.max_edges is not None and self.max_edges < edge_index.size(1):
            col = edge_index[1]
            if self.cut_edges_uniformly:
                keep_edges = torch.randperm(len(col))[:self.max_edges]
            else:
                in_degree = weighted_degree(col, num_nodes=N)
                deg = (1 / in_degree)[col].cpu().numpy()
                p = deg / deg.sum()
                keep_edges = choice(len(col), self.max_edges, replace=False,
                                    p=p)
                keep_edges = torch.tensor(keep_edges, dtype=torch.long)

            edge_index = edge_index[:, keep_edges]
            if edge_weight is not None:
                edge_weight = edge_weight[keep_edges]
            if isinstance(adj, SparseTensor):
                edge_index = SparseTensor.from_edge_index(edge_index,
                                                          edge_weight,
                                                          sparse_sizes=(
                                                              N, N)).t()
            else:
                batch.edge_weight = edge_weight
            batch.edge_index = edge_index

        batch.__dict__['batch_size'] = len(data_list)

        return batch
