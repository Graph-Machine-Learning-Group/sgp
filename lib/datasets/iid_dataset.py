import numpy as np
import torch
from tsl.data import SpatioTemporalDataset, Data
from tsl.data.preprocessing import ScalerModule
from tsl.data.utils import WINDOW, HORIZON, outer_pattern
from tsl.typing import TensArray

from lib.sgp_preprocessing import sgp_spatial_support

_WINDOWING_KEYS = ['data', 'window', 'delay', 'horizon', 'stride']


class IIDDataset(SpatioTemporalDataset):

    def __len__(self):
        return len(self._indices)

    def __setattr__(self, key, value):
        super(SpatioTemporalDataset, self).__setattr__(key, value)
        if key in _WINDOWING_KEYS and all([hasattr(self, attr)
                                           for attr in _WINDOWING_KEYS]):
            last = (self.n_steps - self.sample_span + 1) * self.n_nodes
            self._indices = torch.arange(0, last, self.stride)

    def __getitem__(self, item):
        if self.is_random_iid:
            return self.sample(self.batch_size)
        return self.get(item)

    def sgp_preprocessing(self, k, undirected=False, bidirectional=True,
                           add_self_loops=False, remove_self_loops=True,
                           global_attr=False):
        self.sgp_support = sgp_spatial_support(self.edge_index,
                                               self.edge_weight,
                                               num_nodes=self.n_nodes,
                                               k=k, undirected=undirected,
                                               add_self_loops=add_self_loops,
                                               remove_self_loops=remove_self_loops,
                                               bidirectional=bidirectional,
                                               global_attr=global_attr)
        for v in self.input_map.values():
            pattern = outer_pattern([self.patterns[key] for key in v.keys])
            if 'n' in pattern:
                v.n_channels = v.n_channels * (len(self.sgp_support) + 1)

    @property
    def sgp_preprocess(self):
        return hasattr(self, 'sgp_support')

    @property
    def is_random_iid(self):
        return hasattr(self, 'batch_size')

    def make_random_iid(self, batch_size):
        self.batch_size = batch_size

    def sample(self, N):
        step_index = torch.randint(0, self.n_steps - self.horizon, (N,))
        node_index = torch.randint(0, self.n_nodes, (N,))
        sample = Data()
        for key, value in self.input_map.by_synch_mode(WINDOW).items():
            assert len(value.keys) == 1
            k = value.keys[0]
            tens, trans, pattern = getattr(self, k), self.scalers.get(k), \
                                   self.patterns[k]
            if 'n' in pattern:
                tens = tens[(step_index, None, None, node_index)]  # [N 1 1 f]
            elif 't' in pattern:
                tens = tens[(step_index, None)]
            if trans is not None:
                sample.transform[key] = ScalerModule(**{k: p[None]
                                                        for k, p in
                                                        trans.params().items()})
                if value.preprocess:
                    tens = trans.transform(tens)
            sample.input[key] = tens
            sample.pattern[key] = pattern
        hor_index = torch.stack(
            [step_index + i for i in
             range(self.delay + 1, self.horizon + 1, self.horizon_lag)], 1)
        for key, value in self.target_map.items():
            assert len(value.keys) == 1
            k = value.keys[0]
            tens, trans, pattern = getattr(self, k), self.scalers.get(k), \
                                   self.patterns[k]
            if 'n' in pattern:
                tens = tens[(hor_index, node_index[:, None], None)]  # [N h 1 f]
            elif 't' in pattern:
                tens = tens[(hor_index, None)]
            if trans is not None:
                sample.transform[key] = ScalerModule(**{k: p[None]
                                                        for k, p in
                                                        trans.params().items()})
                if value.preprocess:
                    tens = trans.transform(tens)
            sample.target[key] = tens
            sample.pattern[key] = pattern
        sample.input.node_index = node_index[:, None]
        return sample

    def _populate_input_frame(self, step_index, node_index, synch_mode,
                              out):  # noqa
        for key, value in self.input_map.by_synch_mode(synch_mode).items():
            tgt_node = None if self.sgp_preprocess else node_index
            tens, trans, pattern = self.get_tensors(value.keys,
                                                    cat_dim=value.cat_dim,
                                                    preprocess=value.preprocess,
                                                    step_index=step_index,
                                                    node_index=tgt_node,
                                                    return_pattern=True)
            if 'n' in pattern and self.sgp_preprocess:
                tens = torch.cat([tens.index_select(1, node_index)] +
                                 [adj.index_select(0, node_index) @ tens
                                  for adj in self.sgp_support], dim=-1)
            out.input[key] = tens
            if trans is not None:
                out.transform[key] = trans
            out.pattern[key] = pattern

    def _populate_target_frame(self, step_index, node_index, out):  # noqa
        for key, value in self.targets.items():
            tens, trans, pattern = self.get_tensors(value.keys,
                                                    cat_dim=value.cat_dim,
                                                    preprocess=value.preprocess,
                                                    step_index=step_index,
                                                    node_index=node_index,
                                                    return_pattern=True)
            out.target[key] = tens
            if trans is not None:
                out.transform[key] = trans
            out.pattern[key] = pattern

    def get(self, item):
        if item < 0:
            item = len(self) + item
        step_idx, node_idx = item // self.n_nodes, item % self.n_nodes
        node_idx = torch.tensor([node_idx])
        sample = Data()
        # get input synchronized with window
        wdw_idxs = torch.arange(step_idx, step_idx + self.window,
                                self.window_lag)
        self._populate_input_frame(wdw_idxs, node_idx, WINDOW, sample)
        # get input synchronized with horizon
        hrz_idxs = torch.arange(step_idx + self.horizon_offset,
                                step_idx + self.horizon_offset + self.horizon,
                                self.horizon_lag)
        self._populate_input_frame(hrz_idxs, node_idx, HORIZON, sample)

        # get static attributes
        for key, value in self.get_static_attributes().items():
            sample.input[key] = value
            pattern = self.patterns.get(key)
            if pattern is not None:
                sample.pattern[key] = pattern

        # get mask (if any)
        if self.mask is not None:
            sample.mask = self.mask \
                .index_select(0, hrz_idxs) \
                .index_select(1, node_idx)
            sample.pattern['mask'] = 's n c'

        # get target
        self._populate_target_frame(hrz_idxs, node_idx, sample)

        sample.input['node_index'] = node_idx

        return sample

    # Indexing ################################################################

    def set_indices(self, indices: TensArray):
        indices = torch.as_tensor(indices, dtype=torch.long)
        max_index = (self.n_steps * self.n_nodes) - self.sample_span
        assert all((indices >= 0) & (indices <= max_index)), \
            f"indices must be in the range [0, {max_index}] for {self.name}."
        self._indices = indices

    def expand_indices(self, indices=None, unique=False, merge=False):
        if indices is None:
            samples = (self.n_steps - self.sample_span + 1) // self.stride
            indices = np.arange(samples)
        else:
            indices = np.unique(torch.div(self.indices[indices], self.n_nodes,
                                          rounding_mode='trunc'))
        return super(IIDDataset, self).expand_indices(indices, unique, merge)
