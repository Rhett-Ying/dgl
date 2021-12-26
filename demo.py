"""
DGL_HOME=/home/ubuntu/dgl_0 DGL_LIBRARY_PATH=$DGL_HOME/build PYTHONPATH=$DGL_HOME/python:$PYTHONPATH PYTHONPATH=.:$DGL_HOME/python:$PYTHONPATH DGLBACKEND=pytorch python3 demo.py
"""
import dgl
import torch
import time
import psutil
import gc

num_nodes = 90*1024*1024
num_edges = int(2.2*1024*1024*1024)

start = time.time()
src_ids = torch.randint(0, num_nodes, size=(num_edges,),dtype=torch.int64)
dst_ids = torch.randint(0, num_nodes, size=(num_edges,),dtype=torch.int64)
src = torch.cat((src_ids,dst_ids))
dst = torch.cat((dst_ids,src_ids))
print("------ Prepare edges done in {:.3f}s".format(time.time()-start))
start = time.time()
g=dgl.graph((src, dst), num_nodes=num_nodes)
g=dgl.to_simple(g)
print("----- original g: {}".format(g))
g=dgl.to_bidirected(g)
print("----- bidirected g: {}".format(g))
print("------ Prepare graph done in {:.3f}s".format(time.time()-start))
outdir = '/home/ubuntu/dgl_0/outdir'
print("------- psutil.vm_0: {}".format(psutil.virtual_memory()))
del src_ids, dst_ids, src, dst
gc.collect()
print("------- psutil.vm_1: {}".format(psutil.virtual_memory()))
start = time.time()
dgl.distributed.partition_graph(g, 'order', 2, outdir, part_method='metis', balance_ntypes=None, balance_edges=None)
print("------ partition graph done in {:.3f}s".format(time.time()-start))
print("------- psutil.vm_2: {}".format(psutil.virtual_memory()))