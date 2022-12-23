from typing import List, Optional

import torch
from torch.utils import data
from tsl.data import static_graph_collate, Batch, Data, SpatioTemporalDataset

from lib.sgp_preprocessing import sgp_spatial_support


class SGPLoader(data.DataLoader):

    def __init__(self, dataset: SpatioTemporalDataset, keys: List = None,
                 k: int = 2,
                 undirected: bool = False,
                 add_self_loops: bool = False,
                 remove_self_loops: bool = False,
                 bidirectional: bool = False,
                 global_attr: bool = False,
                 batch_size: Optional[int] = 1,
                 shuffle: bool = False,
                 num_workers: int = 0,
                 **kwargs):
        if 'collate_fn' in kwargs:
            del kwargs['collate_fn']
        self.keys = keys
        self.k = k
        self.undirected = undirected
        self.add_self_loops = add_self_loops
        self.remove_self_loops = remove_self_loops
        self.bidirectional = bidirectional
        self.global_attr = global_attr
        super().__init__(dataset,
                         shuffle=shuffle,
                         batch_size=batch_size,
                         num_workers=num_workers,
                         collate_fn=self.collate,
                         **kwargs)

    def collate(self, data_list: List[Data]):
        elem = data_list[0]
        edge_index, edge_weight = elem.edge_index, elem.edge_weight
        num_nodes = elem.num_nodes
        support = sgp_spatial_support(edge_index, edge_weight,
                                      num_nodes=num_nodes,
                                      k=self.k,
                                      undirected=self.undirected,
                                      add_self_loops=self.add_self_loops,
                                      remove_self_loops=self.remove_self_loops,
                                      bidirectional=self.bidirectional,
                                      global_attr=self.global_attr)
        keys = self.keys
        if keys is None:
            keys = [k for k, pattern in elem.pattern.items()
                    if 'n' in pattern and k in elem.input]
        # subsample every item in batch
        for sample in data_list:
            for key in keys:
                x = sample[key]
                sample[key] = torch.cat([x] + [adj @ x for adj in support],
                                        dim=-1)
        # collate tensors in batch
        batch = static_graph_collate(data_list, Batch)

        # subset sampler can only be used over set of nodes (without edges)
        if 'edge_index' in batch:
            del batch['edge_index']
        if 'edge_weight' in batch:
            del batch['edge_weight']
        batch.__dict__['batch_size'] = len(data_list)

        return batch
