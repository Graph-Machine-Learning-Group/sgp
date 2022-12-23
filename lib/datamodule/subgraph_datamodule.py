from typing import Optional, Mapping

from tsl.data import SpatioTemporalDataModule, SpatioTemporalDataset, Splitter
from tsl.data.loader import StaticGraphLoader

from ..dataloader import SubgraphLoader, SubsetLoader


class SubgraphDataModule(SpatioTemporalDataModule):
    def __init__(self, dataset: SpatioTemporalDataset,
                 max_nodes_training: int = None,
                 receptive_field: int = 1,
                 max_edges: Optional[int] = None,
                 cut_edges_uniformly: bool = False,
                 val_stride: int = None,
                 scalers: Optional[Mapping] = None,
                 splitter: Optional[Splitter] = None,
                 batch_size: int = 32,
                 batch_inference: int = 32,
                 workers: int = 0,
                 pin_memory: bool = False):
        super(SubgraphDataModule, self).__init__(dataset,
                                                 scalers=scalers,
                                                 splitter=splitter,
                                                 mask_scaling=True,
                                                 batch_size=batch_size,
                                                 workers=workers,
                                                 pin_memory=pin_memory)
        if max_nodes_training is not None:
            if receptive_field > 0:
                self._trainloader_class = SubgraphLoader
                self.train_loader_kwargs = dict(
                    num_nodes=max_nodes_training,
                    k=receptive_field,
                    max_edges=max_edges,
                    cut_edges_uniformly=cut_edges_uniformly)
            else:
                self._trainloader_class = SubsetLoader
                self.train_loader_kwargs = dict(max_nodes=max_nodes_training)
        else:
            self._trainloader_class = StaticGraphLoader
            self.train_loader_kwargs = dict()
        self.batch_inference = batch_inference
        self.val_stride = val_stride

    def setup(self, stage=None):
        super(SubgraphDataModule, self).setup(stage)
        if self.val_stride is not None:
            self.valset.indices = self.valset.indices[::self.val_stride]

    def train_dataloader(self, shuffle=True, batch_size=None):
        if self.trainset is None:
            return None
        return self._trainloader_class(self.trainset,
                                       batch_size=batch_size or self.batch_size,
                                       shuffle=shuffle,
                                       num_workers=self.workers,
                                       pin_memory=self.pin_memory,
                                       drop_last=True,
                                       **self.train_loader_kwargs)

    def val_dataloader(self, shuffle=False, batch_size=None):
        if self.valset is None:
            return None
        return StaticGraphLoader(self.valset,
                                 batch_size=batch_size or self.batch_inference,
                                 shuffle=shuffle,
                                 num_workers=self.workers)

    def test_dataloader(self, shuffle=False, batch_size=None):
        if self.testset is None:
            return None
        return StaticGraphLoader(self.testset,
                                 batch_size=batch_size or self.batch_inference,
                                 shuffle=shuffle,
                                 num_workers=self.workers)

    @staticmethod
    def add_argparse_args(parser, **kwargs):
        parser.add_argument('--receptive-field', type=int, default=1)
        parser.add_argument('--max-nodes-training', type=int, default=None)
        parser.add_argument('--max-edges', type=int, default=None)
        parser.add_argument('--cut-edges-uniformly', type=bool, default=False)
        parser.add_argument('--batch-size', type=int, default=32)
        parser.add_argument('--batch-inference', type=int, default=32)
        parser.add_argument('--mask-scaling', type=bool, default=True)
        parser.add_argument('--workers', type=int, default=0)
        return parser
