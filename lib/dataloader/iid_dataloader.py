from typing import List, Optional

import torch
from torch.utils import data
from torch.utils.data import Sampler
from tsl.data import Data

from lib.datasets import IIDDataset


class IIDSampler(Sampler):

    def __init__(self, num_batches):
        super(IIDSampler, self).__init__(torch.arange(num_batches))
        self.num_batches = num_batches

    def __iter__(self):
        for i in range(self.num_batches):
            yield i

    def __len__(self):
        return self.num_batches


class IIDLoader(data.DataLoader):

    def __init__(self, dataset: IIDDataset,
                 batch_size: Optional[int] = 1024,
                 num_batches: int = 1000,
                 num_workers: int = 0,
                 **kwargs):
        if 'collate_fn' in kwargs:
            del kwargs['collate_fn']
        self._batch_size = batch_size
        dataset.make_random_iid(batch_size)
        super().__init__(dataset,
                         batch_size=1,
                         sampler=IIDSampler(num_batches),
                         num_workers=num_workers,
                         collate_fn=self.collate,
                         **kwargs)

    def collate(self, data_list: List[Data]):
        batch = data_list[0]
        batch.__dict__['batch_size'] = self._batch_size
        return batch
