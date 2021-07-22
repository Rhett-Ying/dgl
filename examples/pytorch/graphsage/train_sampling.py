import dgl
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dgl.nn.pytorch as dglnn
import time
import argparse
import tqdm
import statistics

from model import SAGE
from load_graph import load_reddit, inductive_split, load_ogb

def compute_acc(pred, labels):
    """
    Compute the accuracy of prediction given the labels.
    """
    labels = labels.long()
    return (th.argmax(pred, dim=1) == labels).float().sum() / len(pred)

def evaluate(model, g, nfeat, labels, val_nid, device):
    """
    Evaluate the model on the validation set specified by ``val_nid``.
    g : The entire graph.
    inputs : The features of all the nodes.
    labels : The labels of all the nodes.
    val_nid : the node Ids for validation.
    device : The GPU device to evaluate on.
    """
    model.eval()
    with th.no_grad():
        pred = model.inference(g, nfeat, device, args.batch_size, args.num_workers)
    model.train()
    return compute_acc(pred[val_nid], labels[val_nid].to(pred.device))

def load_subtensor(nfeat, labels, seeds, input_nodes, device):
    """
    Extracts features and labels for a subset of nodes
    """
    batch_inputs = nfeat[input_nodes].to(device)
    batch_labels = labels[seeds].to(device)
    return batch_inputs, batch_labels


class AsyncNodeDataLoader:
    def __init__(self, feat, label):
        self.feat = feat
        self.label = label
        self.tf = dgl.dataloading.AsyncTransferer('cuda')
    def _select(in_tensor, idx):
        shape = list(in_tensor.shape)
        shape[0]=len(idx)
        out_tensor = th.empty(*shape, dtype=in_tensor.dtype, pin_memory=True)
        th.index_select(in_tensor, 0,idx, out=out_tensor)
        return out_tensor
    def load(self, blocks):
        res = []
        feat = self._select(self.feat, blocks[0].srcdata[dgl.NID])
        res.append(self.tf.async_copy(feat, 'cuda'))
        label = self._select(self.label, blocks[-1].dstdata[dgl.NID])
        res.append(self.tf.async_copy(label, 'cuda'))
        return res
    def post_process(self, data):
        res = [elem.wait() for elem in data]
        return res

import nvtx


#### Entry point
@nvtx.annotate("run", color="blue")
def run(args, device, data):
    # Unpack data
    n_classes, train_g, val_g, test_g, train_nfeat, train_labels, \
    val_nfeat, val_labels, test_nfeat, test_labels = data
    in_feats = train_nfeat.shape[1]
    train_nid = th.nonzero(train_g.ndata['train_mask'], as_tuple=True)[0]
    val_nid = th.nonzero(val_g.ndata['val_mask'], as_tuple=True)[0]
    test_nid = th.nonzero(~(test_g.ndata['train_mask'] | test_g.ndata['val_mask']), as_tuple=True)[0]

    dataloader_device = th.device('cpu')
    if args.sample_gpu:
        #train_nid = train_nid.to(device)
        # copy only the csc to the GPU
        train_g = train_g.formats(['csc'])
        #train_g = train_g.to(device)
        dataloader_device = device

    # Create PyTorch DataLoader for constructing blocks
    sampler = dgl.dataloading.MultiLayerNeighborSampler(
        [int(fanout) for fanout in args.fan_out.split(',')])
    
    """
    collator = dgl.dataloading.NodeCollator(train_g, train_nid, sampler)
    dataloader = th.utils.data.DataLoader(collator.dataset, collate_fn=collator.collate,
                                             batch_size=args.batch_size,
                                             shuffle=True,
                                             drop_last=False,
                                             num_workers=args.num_workers)
    """
    dataloader = dgl.dataloading.NodeDataLoader(
        train_g,
        train_nid,
        sampler,
        device=dataloader_device,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
        async_tensors = [train_nfeat, train_labels] if not args.prefetch_feat else None)
        #async_tensors = [train_nfeat, train_labels])

    # Define model and optimizer
    model = SAGE(in_feats, args.num_hidden, n_classes, args.num_layers, F.relu, args.dropout)
    model = model.to(device)
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    avg = 0
    iter_tput = []
    rtt = []
    for epoch in range(args.num_epochs):

        """
        rtt_0 = time.perf_counter()
        for _,_,_ in dataloader:
            pass
        rtt.append(time.perf_counter() - rtt_0)
        print("~~~~~~~~~~ pure iter of dataloader: {}".format(time.perf_counter() - rtt_0))
        """


        tic = time.time()

        # Loop over the dataloader to sample the computation dependency graph as a list of
        # blocks.
        tic_step = time.time()
        t01 = [0.0]
        #for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
        h_for = nvtx.start_range(message="epoch")
        for step, (input_nodes, seeds, blocks, afeat, alabel) in enumerate(dataloader):
            with nvtx.annotate("blocks_to"):
                blocks = [block.int().to(device) for block in blocks]

            h_begin = nvtx.start_range(message="others")
            if args.prefetch_feat:
                # Load the input features as well as output labels
                t0 = time.perf_counter()
                batch_inputs, batch_labels = load_subtensor(train_nfeat, train_labels,
                                                            seeds, input_nodes, device)
                t1 = time.perf_counter()
                t01.append(t1-t0)
            else:
                batch_inputs = afeat
                batch_labels = alabel
                #batch_inputs = blocks[0].srcdata['feat']
                #batch_labels = blocks[-1].dstdata['label']

            

            # Compute loss and prediction
            batch_pred = model(blocks, batch_inputs)
            loss = loss_fcn(batch_pred, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_tput.append(len(seeds) / (time.time() - tic_step))
            if step % args.log_every == 0:
                acc = compute_acc(batch_pred, batch_labels)
                gpu_mem_alloc = th.cuda.max_memory_allocated() / 1000000 if th.cuda.is_available() else 0
                print('Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f} | Speed (samples/sec) {:.4f} | GPU {:.1f} MB'.format(
                    epoch, step, loss.item(), acc.item(), np.mean(iter_tput[3:]), gpu_mem_alloc))
            tic_step = time.time()
            nvtx.end_range(h_begin)
        nvtx.end_range(h_for)

        print("---------- median seconds of load_subtensor: {}, total: {}, len:{}".format(statistics.median(t01),
                    sum(t01), len(t01)))

        toc = time.time()
        print('Epoch Time(s): {:.4f}'.format(toc - tic))
        if epoch >= 5:
            avg += toc - tic
        if epoch % args.eval_every == 0 and epoch != 0:
            eval_acc = evaluate(model, val_g, val_nfeat, val_labels, val_nid, device)
            print('Eval Acc {:.4f}'.format(eval_acc))
            test_acc = evaluate(model, test_g, test_nfeat, test_labels, test_nid, device)
            print('Test Acc: {:.4f}'.format(test_acc))

    #print("~~~~~~~ pure iter of dataloader, median:{}, total:{}".format(statistics.median(rtt), sum(rtt)))
    print('Avg epoch time: {}'.format(avg / (epoch - 4)))

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--gpu', type=int, default=0,
                           help="GPU device ID. Use -1 for CPU training")
    argparser.add_argument('--dataset', type=str, default='reddit')
    argparser.add_argument('--num-epochs', type=int, default=20)
    argparser.add_argument('--num-hidden', type=int, default=16)
    argparser.add_argument('--num-layers', type=int, default=2)
    argparser.add_argument('--fan-out', type=str, default='10,25')
    argparser.add_argument('--batch-size', type=int, default=1000)
    argparser.add_argument('--log-every', type=int, default=20)
    argparser.add_argument('--eval-every', type=int, default=5)
    argparser.add_argument('--lr', type=float, default=0.003)
    argparser.add_argument('--dropout', type=float, default=0.5)
    argparser.add_argument('--num-workers', type=int, default=4,
                           help="Number of sampling processes. Use 0 for no extra process.")
    argparser.add_argument('--sample-gpu', action='store_true',
                           help="Perform the sampling process on the GPU. Must have 0 workers.")
    argparser.add_argument('--prefetch-feat', action='store_true',
                           help="fetch feat from separate tensor.")
    argparser.add_argument('--inductive', action='store_true',
                           help="Inductive learning setting")
    argparser.add_argument('--data-cpu', action='store_true',
                           help="By default the script puts all node features and labels "
                                "on GPU when using it to save time for data copy. This may "
                                "be undesired if they cannot fit in GPU memory at once. "
                                "This flag disables that.")
    args = argparser.parse_args()

    if args.gpu >= 0:
        device = th.device('cuda:%d' % args.gpu)
    else:
        device = th.device('cpu')

    if args.dataset == 'reddit':
        g, n_classes = load_reddit()
    elif args.dataset == 'ogbn-products':
        g, n_classes = load_ogb('ogbn-products')
    else:
        raise Exception('unknown dataset')

    if args.inductive:
        train_g, val_g, test_g = inductive_split(g)
        train_nfeat = train_g.ndata.pop('features')
        val_nfeat = val_g.ndata.pop('features')
        test_nfeat = test_g.ndata.pop('features')
        train_labels = train_g.ndata.pop('labels')
        val_labels = val_g.ndata.pop('labels')
        test_labels = test_g.ndata.pop('labels')
    else:
        train_g = val_g = test_g = g
        train_nfeat = val_nfeat = test_nfeat = g.ndata.pop('features')
        if 'feat' in g.ndata:
            g.ndata.pop('feat')
        if 'label' in g.ndata:
            g.ndata.pop('label')
        train_labels = val_labels = test_labels = g.ndata.pop('labels')

    if not args.data_cpu:
        train_nfeat = train_nfeat.to(device)
        train_labels = train_labels.to(device)

    # Pack data
    data = n_classes, train_g, val_g, test_g, train_nfeat, train_labels, \
           val_nfeat, val_labels, test_nfeat, test_labels

    run(args, device, data)
