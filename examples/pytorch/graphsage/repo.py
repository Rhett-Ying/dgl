
import dgl
import dgl.multiprocessing as mp
import torch
import logging
import time
import nvtx


def process_dl(dl_output, dep_done):
    for _ in range(20):
        idx = torch.randperm(10000).share_memory_()
        feat = torch.rand(10000, 600).share_memory_()
        dl_output.put((idx, feat))
    dl_output.put((None, None))
    dep_done.wait()
    print("--- process_dl is done")

def run():
    #g = dgl.data.RedditDataset()[0]
    dl_output = mp.Queue()
    dep_done = mp.Event()
    p_dl = mp.Process(target=process_dl, args=(dl_output, dep_done))
    p_dl.start()

    while True:
        h_tf_get = nvtx.start_range(message="tf_get")
        idx, feat = dl_output.get()
        nvtx.end_range(h_tf_get)
        if idx is None:
            break
    dep_done.set()
    p_dl.join()

    print("All is done...")

if __name__ == '__main__':
    run()