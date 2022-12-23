from tsl.data import SpatioTemporalDataModule
from tsl.data.loader import StaticGraphLoader

from ..dataloader import SGPLoader, IIDLoader
from ..datasets import IIDDataset


class SGPDataModule(SpatioTemporalDataModule):
    def __init__(self, dm: SpatioTemporalDataModule,
                 iid_sampling: bool = True,
                 max_nodes_training: int = None,
                 sgp_preprocessing: bool = False,
                 receptive_field: int = 1,
                 undirected: bool = False,
                 bidirectional: bool = True,
                 add_self_loops: bool = False,
                 global_attr: bool = False,
                 batch_inference: int = None):
        super(SGPDataModule, self).__init__(dm.torch_dataset,
                                            scalers=dm.scalers,
                                            splitter=dm.splitter,
                                            mask_scaling=dm.mask_scaling,
                                            batch_size=dm.batch_size,
                                            workers=dm.workers,
                                            pin_memory=dm.pin_memory)
        if max_nodes_training is not None:
            assert not iid_sampling
        self.max_nodes_training = max_nodes_training
        self.iid_sampling = iid_sampling
        self.sgp_preprocessing = sgp_preprocessing
        self.batch_inference = batch_inference or dm.batch_size
        # sgp_preprocessing
        self.receptive_field = receptive_field
        self.undirected = undirected
        self.bidirectional = bidirectional
        self.add_self_loops = add_self_loops
        self.global_attr = global_attr

    def setup(self, stage=None):
        super(SGPDataModule, self).setup(stage)
        if self.iid_sampling:
            train_index = self.train_slice.tolist()
            trainset = self.torch_dataset.reduce(step_index=train_index)
            trainset = IIDDataset(trainset.data,
                                  trainset.index,
                                  trainset.mask,
                                  (trainset.edge_index, trainset.edge_weight),
                                  trainset._exogenous,
                                  trainset._attributes,
                                  trainset.input_map,
                                  trainset.target_map,
                                  scalers=trainset.scalers,
                                  window=trainset.window,
                                  horizon=trainset.horizon,
                                  delay=trainset.delay,
                                  stride=trainset.stride,
                                  window_lag=trainset.window_lag,
                                  horizon_lag=trainset.horizon_lag,
                                  precision=trainset.precision)
            if self.sgp_preprocessing:
                trainset.sgp_preprocessing(k=self.receptive_field,
                                           undirected=self.undirected,
                                           bidirectional=self.bidirectional,
                                           add_self_loops=self.add_self_loops,
                                           global_attr=self.global_attr)
            self.trainset = trainset

    def train_dataloader(self, shuffle=True, batch_size=None):
        if self.trainset is None:
            return None
        if not self.iid_sampling and self.sgp_preprocessing:
            return SGPLoader(self.trainset,
                             k=self.receptive_field,
                             add_self_loops=self.add_self_loops,
                             undirected=self.undirected,
                             bidirectional=self.bidirectional,
                             global_attr=self.global_attr,
                             batch_size=batch_size or self.batch_size,
                             shuffle=shuffle,
                             pin_memory=self.pin_memory,
                             num_workers=self.workers,
                             drop_last=True)
        elif self.iid_sampling:
            return IIDLoader(self.trainset, batch_size or self.batch_size)
        return StaticGraphLoader(self.trainset,
                                 batch_size=batch_size or self.batch_size,
                                 shuffle=shuffle,
                                 pin_memory=self.pin_memory,
                                 num_workers=self.workers,
                                 drop_last=True)

    def val_dataloader(self, shuffle=False, batch_size=None):
        if self.valset is None:
            return None
        if self.sgp_preprocessing:
            return SGPLoader(self.valset,
                             k=self.receptive_field,
                             add_self_loops=self.add_self_loops,
                             undirected=self.undirected,
                             bidirectional=self.bidirectional,
                             global_attr=self.global_attr,
                             batch_size=batch_size or self.batch_inference,
                             shuffle=shuffle,
                             num_workers=self.workers)
        return StaticGraphLoader(self.valset,
                                 batch_size=batch_size or self.batch_inference,
                                 shuffle=shuffle,
                                 num_workers=self.workers)

    def test_dataloader(self, shuffle=False, batch_size=None):
        if self.testset is None:
            return None
        if self.sgp_preprocessing:
            return SGPLoader(self.testset,
                             k=self.receptive_field,
                             add_self_loops=self.add_self_loops,
                             undirected=self.undirected,
                             bidirectional=self.bidirectional,
                             global_attr=self.global_attr,
                             batch_size=batch_size or self.batch_inference,
                             shuffle=shuffle,
                             num_workers=self.workers)
        return StaticGraphLoader(self.testset,
                                 batch_size=batch_size or self.batch_inference,
                                 shuffle=shuffle,
                                 num_workers=self.workers)
