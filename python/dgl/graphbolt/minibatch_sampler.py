"""Minibatch Sampler"""

from typing import Optional

import torch
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe

from .itemset import *


__all__ = ["MinibatchSampler"]


@functional_datapipe("dgl_minibatch")
class DGLMinibatcherIterDataPipe(IterDataPipe):
    """Custom DataPipe."""

    def __init__(self, item_set: ItemSet or DictItemSet):
        super().__init__()
        self.item_set = item_set

    def __iter__(self):
        for item in self.item_set:
            yield item


class MinibatchSampler:
    """Minibatch Sampler.

    This sampler generates mini-batches from input `ItemSet` according to
    specified configuration.
    """

    def __init__(
        self,
        item_set: ItemSet or DictItemSet,
        batch_size: int,
        shuffle: Optional[bool] = False,
        drop_last: Optional[bool] = False,
    ):
        self.item_set = item_set
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self):
        data_pipe = DGLMinibatcherIterDataPipe(self.item_set)
        if self.shuffle:
            data_pipe = data_pipe.shuffle()
        data_pipe = data_pipe.batch(
            batch_size=self.batch_size,
            drop_last=self.drop_last,
            wrapper_class=torch.Tensor,
        )
        return iter(data_pipe)
