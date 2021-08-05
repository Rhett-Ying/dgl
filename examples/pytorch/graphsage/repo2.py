
import dgl
import dgl.multiprocessing as mp
import torch
import logging
import time
import nvtx

@nvtx.annotate("proc_tf")
def process_tf(dl_output, dep_done, tf_output):
    while True:
        h_tf_get = nvtx.start_range(message="tf_get")
        idx, feat = dl_output.get()
        nvtx.end_range(h_tf_get)
        if idx is None:
            break
        h_tf_slice = nvtx.start_range(message="tf_slice")
        s_feat = feat[idx]
        nvtx.end_range(h_tf_slice)
        h_tf_shm = nvtx.start_range(message="tf_shm")
        s_feat.share_memory_()
        nvtx.end_range(h_tf_shm)
        h_tf_put = nvtx.start_range(message="tf_put")
        tf_output.put(s_feat)
        nvtx.end_range(h_tf_put)
    dep_done.set()

def run():
    g = dgl.data.RedditDataset()[0]
    g_feat = g.ndata['feat']
    dl_output = mp.Queue()
    dep_done = mp.Event()
    tf_output = mp.Queue()

    for _ in range(10):
        idx = torch.randperm(11000).share_memory_()
        feat = g_feat[idx].share_memory_()
        dl_output.put((idx,feat))
    dl_output.put((None,None))

    p_tf = mp.Process(target=process_tf, args=(dl_output, dep_done, tf_output))
    p_tf.start()
    p_tf.join()
    print("tf_output.size(): {}".format(tf_output.qsize()))
    print("All is done...")

if __name__ == '__main__':
    run()