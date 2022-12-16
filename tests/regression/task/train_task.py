import logging
import os
import time
import argparse

from task import Task

class TrainTask(Task):
    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Dist Train Task",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        parser.add_argument(
            "--data_store",
            type=str,
            required=True,
            help="data store path like S3 URI which stores datasets"
        )
        parser.add_argument(
            "--data_name",
            type=str,
            required=True,
            help="target dataset name"
        )
        args, _ = parser.parse_known_args()
        self.data_store = args.data_store
        self.data_name = args.data_name

    def _prepare_data(self):
        workspace = os.environ.get('WORKSPACE', '/workspace')

        # download partitioned graphs
        os.system(
            f"python3 /dgl/tests/regression/data_store.py"
            f" --data_store {self.data_store} "
            f" --data_name {self.data_name}"
            f" --output_dir {workspace}"
        )
        self.data_path = os.path.join(workspace, self.data_name)

    def _do_run(self):
        logging.info("Running distributed train task...")
        workspace = os.environ.get('WORKSPACE', '/workspace')
        os.system(f"ls -lh {self.data_path}")
        part_config = os.path.join(self.data_path, 'ogb-product.json')
        ip_config = os.environ['IP_CONFIG']
        tic = time.time()
        os.system(
            f"python3 /dgl/tools/launch.py"
            f" --workspace {workspace}"
            f" --num_trainers 1 --num_samplers 0 --num_servers 1"
            f" --part_config {part_config}"
            f" --ip_config {ip_config}"
            f" 'python3 /dgl/tests/regression/task/scripts/train_dist.py"
            f" --graph_name ogb-product --ip_config {ip_config}"
            f" --num_epochs 5 --batch_size 1000"
            f" '"
        )
        self.tic_toc = time.time() - tic

    def _print_metrics(self):
        logging.info(
            f"-------------- metrics ---------------- \n"
            f"######## Task: Distributed Train (GraphSAGE) \n"
            f"######## Dataset: {self.data_name} \n"
            f"######## Time(seconds): {self.tic_toc:.3f} \n"
        )
