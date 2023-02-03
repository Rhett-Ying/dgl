import os
import time
from ogb.lsc import MAG240MDataset
import torch
import numpy as np
import dgl

g_tic = time.time()

base_dir = "/home/ubuntu"
base_dir = "/"
input_dir = os.path.join(base_dir, "datasets/mag240m_kddcup2021")

graph_path = os.path.join(input_dir, "graph.dgl")
if not os.path.isfile(graph_path):
    dataset = MAG240MDataset(root=os.path.join(base_dir, "datasets/"))

    print(dataset.num_papers) # number of paper nodes
    print(dataset.num_authors) # number of author nodes
    print(dataset.num_institutions) # number of institution nodes
    print(dataset.num_paper_features) # dimensionality of paper features
    print(dataset.num_classes) # number of subject area classes

    # get edges
    edges_writes = dataset.edge_index('author', 'writes', 'paper')
    edges_cites = dataset.edge_index('paper', 'paper')
    edges_aff = dataset.edge_index('author', 'institution')
    edges_dict = {
        ('author', 'writes', 'paper') : (torch.from_numpy(edges_writes[0]), torch.from_numpy(edges_writes[1])),
        ('paper', 'cites', 'paper') : (torch.from_numpy(edges_cites[0]), torch.from_numpy(edges_cites[1])),
        ('author', 'affiliated_with', 'institution') : (torch.from_numpy(edges_aff[0]), torch.from_numpy(edges_aff[1])),
    }
    g = dgl.heterograph(edges_dict, num_nodes_dict = {
        "paper": dataset.num_papers,
        "author": dataset.num_authors,
        "institution": dataset.num_institutions,
    })
    print(g)

    dgl.save_graphs(graph_path, [g])
    print(f"graph is saved to {graph_path}...")
else:
    glist, _ = dgl.load_graphs(graph_path)
    g = glist[0]
    print(g)


# train/val/test split
train_mask_path = os.path.join(input_dir, "processed/paper/train_mask.npy")
if not os.path.isfile(train_mask_path):
    train_idx = torch.from_numpy(dataset.get_idx_split('train'))
    paper_train_mask = np.zeros(g.num_nodes("paper"), dtype=np.bool_)
    paper_train_mask[train_idx] = True
    with open(train_mask_path, "wb") as f:
        np.save(f, paper_train_mask)
    del train_idx, paper_train_mask
    print("train idx mask are saved...")

val_mask_path = os.path.join(input_dir, "processed/paper/val_mask.npy")
if not os.path.isfile(val_mask_path):
    valid_idx = torch.from_numpy(dataset.get_idx_split('valid'))
    paper_val_mask = np.zeros(g.num_nodes("paper"), dtype=np.bool_)
    paper_val_mask[valid_idx] = True
    with open(val_mask_path, "wb") as f:
        np.save(f, paper_val_mask)
    del valid_idx, paper_val_mask
    print("val idx mask are saved...")

test_mask_path = os.path.join(input_dir, "processed/paper/test_mask.npy")
if not os.path.isfile(test_mask_path):
    testdev_idx = torch.from_numpy(dataset.get_idx_split('test-dev'))
    paper_test_mask = np.zeros(g.num_nodes("paper"), dtype=np.bool_)
    paper_test_mask[testdev_idx] = True
    with open(test_mask_path, "wb") as f:
        np.save(f, paper_test_mask)
    del testdev_idx, paper_test_mask
    print("test idx mask are saved...")

print("start to chunk graph...")
output_dir = os.path.join(base_dir, "datasets/mag240m_kddcup2021/chunked_data")
num_chunks = 4
from tools.chunk_graph import chunk_graph
chunk_graph(
    g,
    "ogb-mag240m",
    {
        "paper": {
            "feat": os.path.join(input_dir, "processed/paper/node_feat.npy"),
            "label": os.path.join(input_dir, "processed/paper/node_label.npy"),
            "year": os.path.join(input_dir, "processed/paper/node_year.npy"),
            "train_mask": train_mask_path,
            "val_mask": val_mask_path,
            "test_mask": test_mask_path,
        }
    },
    {},
    num_chunks,
    output_dir,
)

print(f"All is done... Elapsed time: {time.time() - g_tic}")
