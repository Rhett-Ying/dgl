import torch
import torch.multiprocessing as mp
from torch.multiprocessing import Process, Queue, SimpleQueue, Value
import logging
from queue import Empty
import time
import dgl
import gc

start_method = 'spawn' #'forkserver'
#mp.set_start_method(start_method)

dl_dep_done = mp.Event()

g_slice_num = 1000


def process_dl(logger, dl_output, dl_done, dep_done):
    logger.info("process_dl get started...")
    limit = 5
    for i in range(10):
        while True:
            if dl_output.qsize() > limit:
                #print("--- process_dl sleeps...")
                time.sleep(0.1)
                continue
            else:
                break
        idx = torch.randperm(g_slice_num)
        dl_output.put((idx.unique(), i))
        #dl_output.put(i)
    dl_done.value = True
    """
    while not dep_done.value:
        time.sleep(0.1)
    """
    dl_dep_done.wait()
    print("process_dl is done...")


def process_tf(logger, dl_output, dl_done, tf_output, tf_done, feat, dep_done):
    #logger.info(graph.num_nodes())
    logger.info("process_tf get started...")
    limit = 5
    snap = 0.1
    while not dl_done.value or not dl_output.empty():
        if tf_output.qsize() > limit:
            time.sleep(snap)
            continue
        try:
            idx, step = dl_output.get(timeout=snap)
            print(step)
        except Empty:
            continue
        logger.info("slice begin")
        feat = torch.rand(3000, 100)
        idx = torch.randperm(g_slice_num)
        #s_feat = torch.Tensor()
        #for _, x in enumerate(idx):
        #    s_feat = torch.cat((s_feat, feat[x]))
        #s_feat = feat[idx[0]]
        s_feat = feat[idx]
        #s_feat = torch.index_select(feat, 0, idx)
        logger.info("slice end")
        tf_output.put(s_feat)
    tf_done.value = True
    logger.info("process_tf is done...")
    dl_dep_done.set()
    while not dep_done.value:
        time.sleep(0.1)


def run():
    logger = mp.log_to_stderr()
    logger.setLevel(logging.INFO)

    dl_output = Queue()
    dl_done = Value('b', False)
    tf_output = Queue()
    tf_done = Value('b', False)
    run_done = Value('b', False)

    graph = dgl.data.RedditDataset()[0]

    # below del/collect not work
    #del graph
    #gc.collect()

    #time.sleep(5)
    #graph =dgl.data.CoraGraphDataset()[0]
    #feat = graph.ndata['feat']
    #shape = feat.shape
    #feat = torch.Tensor(feat).share_memory_()
    #feat.share_memory_()
    #logger.info(graph.num_nodes())

    #feat = torch.rand(shape).share_memory_()

    #feat1 = torch.rand(500000, 10000)
    #feat2 = torch.rand(500000, 10000)
    feat = torch.rand(1000, 1000).share_memory_()
    #idx = torch.randperm(g_slice_num)
    #logger.info(feat[idx])
    #logger.info("---------------- done slice in run()")


    Process(target=process_dl, args=(logger, dl_output, dl_done,
            tf_done, ), daemon=True).start()
    Process(target=process_tf, args=(logger, dl_output, dl_done, tf_output,
            tf_done, feat, run_done), daemon=True).start()
    while not tf_done.value or not tf_output.empty():
        try:
            item = tf_output.get(timeout=0.1)
            print(item)
            #del item
        except Empty:
            continue
        #print("item: ".format(item))
    run_done.value = True


if __name__ == '__main__':
    #graph = dgl.data.RedditDataset()[0]
    run()
