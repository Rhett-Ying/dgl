import argparse
import os

import dgl.graphbolt as gb
import numpy as np
import pandas as pd

from dgl import AddReverse, Compose, ToSimple
from ogb.nodeproppred import DglNodePropPredDataset


def prepare_data(args):
    dataset = DglNodePropPredDataset(
        name=args.dataset, root="/home/ubuntu/workspace/datasets"
    )
    # Get train/valid/test index.
    split_idx = dataset.get_idx_split()

    # - graph: dgl graph object.
    # - label: torch tensor of shape (num_nodes, num_tasks).
    g, labels = dataset[0]

    # Flatten the labels for "paper" type nodes. This step reduces the
    # dimensionality of the labels. We need to flatten the labels because
    # the model requires a 1-dimensional label tensor.
    labels = labels["paper"].flatten()

    # Apply transformation to the graph.
    # - "ToSimple()" removes multi-edge between two nodes.
    # - "AddReverse()" adds reverse edges to the graph.
    transform = Compose([ToSimple(), AddReverse()])
    g = transform(g)

    print(f"Loaded graph: {g}")

    return g, labels, dataset.num_classes, split_idx


def main(args):
    # Prepare the data.
    g, labels, num_classes, split_idx = prepare_data(args)

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory {output_dir}")

    # Save edges into CSV files.
    edges_dir = os.path.join(output_dir, "edges")
    if not os.path.exists(edges_dir):
        os.makedirs(edges_dir)
        print(f"Created directory {edges_dir}")
    for etype in g.canonical_etypes:
        src, dst = g.edges(etype=etype)
        etype_str = gb.etype_tuple_to_str(etype)
        df = pd.DataFrame({"src": src.numpy(), "dst": dst.numpy()})
        csv_path = os.path.join(edges_dir, f"{etype_str}.csv")
        df.to_csv(csv_path, index=False, header=False)
        print(f"Saved {etype_str} edges to {csv_path}")

    # Save node features into NPY files.
    feat_dir = os.path.join(output_dir, "features")
    if not os.path.exists(feat_dir):
        os.makedirs(feat_dir)
        print(f"Created directory {feat_dir}")
    # Only 'paper' type nodes have features.
    for feat_name in g.nodes["paper"].data.keys():
        feat = g.nodes["paper"].data[feat_name]
        npy_path = os.path.join(feat_dir, f"paper-{feat_name}.npy")
        np.save(npy_path, feat.numpy())
        print(f"Saved 'paper' node features to {npy_path}")

    # No edge features.

    # Save train/valid/test index/labels into NPY files.
    tvt_dir = os.path.join(output_dir, "tvt")
    if not os.path.exists(tvt_dir):
        os.makedirs(tvt_dir)
        print(f"Created directory {tvt_dir}")
    tvt_type = "paper"  # Only 'paper' type nodes have idx/labels.
    for tvt_name in ["train", "valid", "test"]:
        tvt_idx = split_idx[tvt_name][tvt_type]
        npy_path = os.path.join(
            tvt_dir, f"{tvt_type}-{tvt_name}-seed-nodes.npy"
        )
        np.save(npy_path, tvt_idx)
        print(f"Saved {tvt_name} index of {tvt_type} to {npy_path}")

        label = labels[tvt_idx]
        npy_path = os.path.join(tvt_dir, f"{tvt_type}-{tvt_name}-labels.npy")
        np.save(npy_path, label.numpy())

    # YAML content.
    # Relative path is required instead of absolute path.
    edges_dir = os.path.relpath(edges_dir, args.output_dir)
    feat_dir = os.path.relpath(feat_dir, args.output_dir)
    tvt_dir = os.path.relpath(tvt_dir, args.output_dir)
    yaml_content = f"""
      dataset_name: {args.dataset}
      graph:
        nodes:
          - type: author
            num: {g.num_nodes('author')}
          - type: field_of_study
            num: {g.num_nodes('field_of_study')}
          - type: institution
            num: {g.num_nodes('institution')}
          - type: paper
            num: {g.num_nodes('paper')}
        edges:
          - type: 'author:affiliated_with:institution'
            format: csv
            path: {os.path.join(edges_dir, 'author:affiliated_with:institution.csv')}
          - type: 'author:writes:paper'
            format: csv
            path: {os.path.join(edges_dir, 'author:writes:paper.csv')}
          - type: 'paper:cites:paper'
            format: csv
            path: {os.path.join(edges_dir, 'paper:cites:paper.csv')}
          - type: 'paper:has_topic:field_of_study'
            format: csv
            path: {os.path.join(edges_dir, 'paper:has_topic:field_of_study.csv')}
      feature_data:
        - domain: node
          type: paper
          name: feat
          format: numpy
          in_memory: true
          path: {os.path.join(feat_dir, 'paper-feat.npy')}
        - domain: node
          type: paper
          name: year
          format: numpy
          in_memory: true
          path: {os.path.join(feat_dir, 'paper-year.npy')}
      tasks:
        - name: node_classification
          num_classes: {num_classes}
          train_set:
            - type: paper
              data:
                - name: seed_nodes
                  format: numpy
                  in_memory: true
                  path: {os.path.join(tvt_dir, 'paper-train-seed-nodes.npy')}
                - name: labels
                  format: numpy
                  in_memory: true
                  path: {os.path.join(tvt_dir, 'paper-train-labels.npy')}
          validation_set:
            - type: paper
              data:
                - name: seed_nodes
                  format: numpy
                  in_memory: true
                  path: {os.path.join(tvt_dir, 'paper-valid-seed-nodes.npy')}
                - name: labels
                  format: numpy
                  in_memory: true
                  path: {os.path.join(tvt_dir, 'paper-valid-labels.npy')}
          test_set:
            - type: paper
              data:
                - name: seed_nodes
                  format: numpy
                  in_memory: true
                  path: {os.path.join(tvt_dir, 'paper-test-seed-nodes.npy')}
                - name: labels
                  format: numpy
                  in_memory: true
                  path: {os.path.join(tvt_dir, 'paper-test-labels.npy')}
    """
    yaml_file = os.path.join(args.output_dir, "metadata.yaml")
    with open(yaml_file, "w") as f:
        f.write(yaml_content)
        print(f"Saved metadata to {yaml_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GraphBolt-RGCN-OGBN-MAG-Preprocess"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/ubuntu/workspace/gb_examples/datasets",
    )
    parser.add_argument("--dataset", type=str, default="ogbn-mag")
    parser.add_argument("--force", action="store_true", default=False)

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created directory {args.output_dir}")
    args.output_dir = os.path.join(args.output_dir, args.dataset)

    # Generate metata.yaml and data files.
    yaml_file = os.path.join(args.output_dir, "metadata.yaml")
    if args.force or (not os.path.exists(yaml_file)):
        main(args)
    else:
        print(f"Found existing metadata file {yaml_file}. Skipping...")

    # Preprocess the generated data.
    new_yaml_file = gb.preprocess_ondisk_dataset(args.output_dir)
    print(f"Preprocessed metadata file: {new_yaml_file}")
