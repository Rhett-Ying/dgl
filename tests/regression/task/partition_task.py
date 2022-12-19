import logging
import os
import time
import re
import argparse

from task import Task

class PartitionTask(Task):
    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Graph Partition Task",
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
        self.num_parts = 4

    def _prepare_data(self):
        workspace = os.environ.get('WORKSPACE', '/workspace')

        # download raw data
        bin_path = os.path.join(
            os.environ['DGL_ROOT_DIR'],
            'tests/regression/data_store.py'
        )
        os.system(
            f"python3 {bin_path}"
            f" --data_store {self.data_store} "
            f" --data_name {self.data_name}"
            f" --output_dir {workspace}"
        )
        self.data_path = os.path.join(workspace, self.data_name)

    def _do_run(self):
        logging.info("Running partition task...")

        # Step1: graph partition
        in_dir = os.path.join(self.data_path, "chunked-data")
        output_dir = os.path.join(self.data_path, "parted_data")
        bin_path = os.path.join(
            os.environ['DGL_ROOT_DIR'],
            'tools/partition_algo/random_partition.py'
        )
        os.system(
            f"python3 {bin_path}"
            f" --in_dir {in_dir} --out_dir {output_dir}"
            f" --num_partitions {self.num_parts}"
        )
        ## copy partition results to all nodes
        ip_config = os.environ["IP_CONFIG"]
        ssh_port = os.environ["SSH_PORT"]
        with open(ip_config, 'r') as f:
            for i, line in enumerate(f):
                if i == 0:
                    # skip current node
                    continue
                ip = line.rstrip()
                os.system(
                    f"scp -r -o StrictHostKeyChecking=no -P {ssh_port} "
                    f" {output_dir} {ip}:{output_dir} "
                )
                logging.info(f"Finished to copy partition results to {ip}...")
        with open(ip_config, 'r') as f:
            for line in f:
                ip = line.rstrip()
                logging.info(f"-------IP: {ip}")
                os.system(
                    f"ssh -o StrictHostKeyChecking=no -p {ssh_port} {ip} 'ls -lh {self.data_path}/*'"
                )

        tic = time.time()
        # Step2: data dispatch
        partition_dir = os.path.join(self.data_path, 'parted_data')
        out_dir = os.path.join(self.data_path, 'partitioned')
        in_dir = os.path.join(self.data_path, "chunked-data")

        bin_path = os.path.join(
            os.environ['DGL_ROOT_DIR'],
            'tools/dispatch_data.py'
        )
        cmd = f"python3 {bin_path}"
        cmd += f" --in-dir {in_dir}"
        cmd += f" --partitions-dir {partition_dir}"
        cmd += f" --out-dir {out_dir}"
        cmd += f" --ip-config {ip_config}"
        cmd += f" --ssh-port {os.environ['SSH_PORT']}"
        cmd += " --process-group-timeout 60"

        os.system(cmd)
        os.system(f"ls -lh {out_dir}")
        self.tic_toc = time.time() - tic

        logging.info(
            f"Graph partition for {self.data_name} with "
            f"{self.num_parts} parts is done..."
        )

    def _print_metrics(self):
        logging.info(
            f"-------------- metrics ---------------- \n"
            f"######## Task: Graph Partition [Dispatch data] \n"
            f"######## Dataset: {self.data_name} \n"
            f"######## Num_parts: {self.num_parts} \n"
            f"######## Time(seconds): {self.tic_toc:.3f} \n"
            f"######## Peak memory(GB): {get_peak_mem():.3f} \n"
        )


def get_peak_mem():
    """Get the peak memory size.

    Returns
    -------
    float
        The peak memory size in GB.
    """
    if not os.path.exists("/proc/self/status"):
        return 0.0
    for line in open("/proc/self/status", "r"):
        if "VmPeak" in line:
            mem = re.findall(r"\d+", line)[0]
            return int(mem) / 1024 / 1024
    return 0.0

