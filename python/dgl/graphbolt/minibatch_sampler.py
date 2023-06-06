"""Minibatch Sampler"""

from functools import partial
from typing import Optional

import torch
from torch.utils.data import default_collate
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe, IterableWrapper
from .itemset import *
from ..batch import batch as dgl_batch
from ..heterograph import DGLGraph

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


def colloate(batch):
    data = next(iter(batch))
    if isinstance(data, DGLGraph):
        return dgl_batch(batch)
    elif isinstance(data, dict):
        def _wrapper(batch):
            if isinstance(batch, list):
                return tuple(batch)
            return batch
        collected_batch = {}
        for b in batch:
            b_key = list(b.keys())[0]
            b_value = list(b.values())[0]
            if b_key not in collected_batch:
                collected_batch[b_key] = []
            collected_batch[b_key].append(b_value)
        return {
            key: _wrapper(default_collate(value))
            for key, value in collected_batch.items()
        }
    return default_collate(batch)

@functional_datapipe("graphbolt_batch")
class MinibatchSampler(IterDataPipe):
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
        super().__init__()
        self.item_set = item_set
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self):
        data_pipe = IterableWrapper(self.item_set)
        if self.shuffle:
            data_pipe = data_pipe.shuffle()
        data_pipe = data_pipe.batch(
            batch_size=self.batch_size,
            drop_last=self.drop_last,
            # wrapper_class=torch.Tensor,
            # wrapper_class=DGLGraph,
        )
        data_pipe = data_pipe.collate(collate_fn=colloate)
        return iter(data_pipe)
