
import dgl
#import dgl.multiprocessing as mp
import threading
import queue
import torch
import logging
import time
import nvtx

def thread_dl(dl_output):
    for _ in range(10):
        idx = torch.randperm(10000)
        dl_output.put(idx)
    dl_output.put(None)
    print("--- thread_dl is done...") 

def thread_tf(dl_output, feat, tf_output):
    tf = dgl.dataloading.AsyncTransferer('cuda')
    while True:
        idx = dl_output.get()
        if idx is None:
            break
        s_feat = feat[idx]
        fut = tf.async_copy(s_feat, 'cuda')
        tf_output.put(fut)
    tf_output.put(None)
    print("--- thread_tf is done")

def run():

    dl_output = queue.Queue()
    tf_output = queue.Queue()

    feat = torch.rand(100000, 500)

    t_dl = threading.Thread(target=thread_dl, args=(dl_output,))
    t_dl.start()
    t_tf = threading.Thread(target=thread_tf, args=(dl_output, feat, tf_output,))
    t_tf.start()

    while True:
        h_tf_get = nvtx.start_range(message="tf_get")
        feat = tf_output.get()
        nvtx.end_range(h_tf_get)
        if feat is None:
            break
    t_dl.join()
    t_tf.join()
    print("All is done...")

if __name__ == '__main__':
    run()