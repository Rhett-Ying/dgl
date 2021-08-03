

import torch.multiprocessing as mp
#mp.set_start_method('spawn')


import torch

import dgl



#import dgl.multiprocessing as mp
#from dgl import multiprocessing as mp

import logging
import time


def slice():
    idx = torch.randperm(100)
    feat = torch.rand(200, 1000)
    #logger.info("slice_begin")
    s_feat = feat[idx]
    #logger.info("slice_end")

def run():
    g = dgl.data.RedditDataset()[0]
    #g = dgl.data.CoraGraphDataset()[0]

    #logger = mp.log_to_stderr()
    #logger.setLevel(logging.INFO)

    ctx = mp.get_context('spawn')

    #p_slice = mp.Process(target=slice, args=())
    p_slice = ctx.Process(target=slice, args=())
    p_slice.start()
    p_slice.join()

    #logger.info("All is done...")

if __name__ == '__main__':
    run()