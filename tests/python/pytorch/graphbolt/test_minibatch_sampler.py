import dgl
import pytest
import torch
from torch.testing import assert_close
from dgl.graphbolt import *


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("shuffle", [True, False])
@pytest.mark.parametrize("drop_last", [True, False])
def test_MinibatchSampler_node_edge_ids(batch_size, shuffle, drop_last):
    # Node or edge IDs.
    num_ids = 11
    item_set = ItemSet(torch.arange(0, num_ids))
    minibatch_sampler = MinibatchSampler(
        item_set, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last
    )
    items = []
    for i, item in enumerate(minibatch_sampler):
        is_last = (i + 1) * batch_size >= num_ids
        if not is_last or num_ids % batch_size == 0:
            assert len(item) == batch_size
        else:
            if not drop_last:
                assert len(item) == num_ids % batch_size
            else:
                assert False
        items.append(item)
    items = torch.cat(items)
    assert torch.all(items[:-1] <= items[1:]) is not shuffle


"""
def test_ItemSet_graphs():
    # Graphs.
    graphs = [dgl.rand_graph(10, 20) for _ in range(5)]
    item_set = ItemSet(graphs)
    for i, item in enumerate(item_set):
        assert graphs[i] == item


def test_ItemSet_node_pairs():
    # Node pairs.
    node_pairs = (torch.arange(0, 5), torch.arange(5, 10))
    item_set = ItemSet(node_pairs)
    for i, (src, dst) in enumerate(item_set):
        assert node_pairs[0][i] == src
        assert node_pairs[1][i] == dst


def test_ItemSet_node_pairs_labels():
    # Node pairs and labels
    node_pairs = (torch.arange(0, 5), torch.arange(5, 10))
    labels = torch.randint(0, 3, (5,))
    item_set = ItemSet((node_pairs[0], node_pairs[1], labels))
    for i, (src, dst, label) in enumerate(item_set):
        assert node_pairs[0][i] == src
        assert node_pairs[1][i] == dst
        assert labels[i] == label


def test_ItemSet_head_tail_neg_tails():
    # Head, tail and negative tails.
    heads = torch.arange(0, 5)
    tails = torch.arange(5, 10)
    neg_tails = torch.arange(10, 20).reshape(5, 2)
    item_set = ItemSet((heads, tails, neg_tails))
    for i, (head, tail, negs) in enumerate(item_set):
        assert heads[i] == head
        assert tails[i] == tail
        assert_close(neg_tails[i], negs)


def test_DictItemSet_node_edge_ids():
    # Node or edge IDs
    ids = {
        ("user", "like", "item"): ItemSet(torch.arange(0, 5)),
        ("user", "follow", "user"): ItemSet(torch.arange(0, 5)),
    }
    chained_ids = []
    for key, value in ids.items():
        chained_ids += [(key, v) for v in value]
    item_set = DictItemSet(ids)
    for i, item in enumerate(item_set):
        assert len(item) == 1
        assert isinstance(item, dict)
        assert chained_ids[i][0] in item
        assert item[chained_ids[i][0]] == chained_ids[i][1]


def test_DictItemSet_node_pairs():
    # Node pairs.
    node_pairs = (torch.arange(0, 5), torch.arange(5, 10))
    node_pairs_dict = {
        ("user", "like", "item"): ItemSet(node_pairs),
        ("user", "follow", "user"): ItemSet(node_pairs),
    }
    expected_data = []
    for key, value in node_pairs_dict.items():
        expected_data += [(key, v) for v in value]
    item_set = DictItemSet(node_pairs_dict)
    for i, item in enumerate(item_set):
        assert len(item) == 1
        assert isinstance(item, dict)
        assert expected_data[i][0] in item
        assert item[expected_data[i][0]] == expected_data[i][1]


def test_DictItemSet_node_pairs_labels():
    # Node pairs and labels
    node_pairs = (torch.arange(0, 5), torch.arange(5, 10))
    labels = torch.randint(0, 3, (5,))
    node_pairs_dict = {
        ("user", "like", "item"): ItemSet(
            (node_pairs[0], node_pairs[1], labels)
        ),
        ("user", "follow", "user"): ItemSet(
            (node_pairs[0], node_pairs[1], labels)
        ),
    }
    expected_data = []
    for key, value in node_pairs_dict.items():
        expected_data += [(key, v) for v in value]
    item_set = DictItemSet(node_pairs_dict)
    for i, item in enumerate(item_set):
        assert len(item) == 1
        assert isinstance(item, dict)
        assert expected_data[i][0] in item
        assert item[expected_data[i][0]] == expected_data[i][1]


def test_DictItemSet_head_tail_neg_tails():
    # Head, tail and negative tails.
    heads = torch.arange(0, 5)
    tails = torch.arange(5, 10)
    neg_tails = torch.arange(10, 20).reshape(5, 2)
    item_set = ItemSet((heads, tails, neg_tails))
    data_dict = {
        ("user", "like", "item"): ItemSet((heads, tails, neg_tails)),
        ("user", "follow", "user"): ItemSet((heads, tails, neg_tails)),
    }
    expected_data = []
    for key, value in data_dict.items():
        expected_data += [(key, v) for v in value]
    item_set = DictItemSet(data_dict)
    for i, item in enumerate(item_set):
        assert len(item) == 1
        assert isinstance(item, dict)
        assert expected_data[i][0] in item
        assert_close(item[expected_data[i][0]], expected_data[i][1])
"""
