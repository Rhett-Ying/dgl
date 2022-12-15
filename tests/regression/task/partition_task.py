import logging
import os

from task import Task

class PartitionTask(Task):
    def __init__(self, data_store, data_name, num_parts):
        self.data_store = data_store
        self.data_name = data_name
        self.num_parts = num_parts

    def _prepare_data(self):
        workspace = os.environ.get('WORKSPACE', '/workspace')

        # download raw data
        os.system(
            f"python3 /dgl/tests/regression/data_store.py"
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
        os.system(
            "python3 /dgl/tools/partition_algo/random_partition.py"
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
                    f"rsync -avrz -e 'ssh -o StrictHostKeyChecking=no -p {ssh_port}' "
                    f" {output_dir} {ip}:{output_dir} "
                    f" && ls -lh {self.data_path}/*"
                )
                logging.info(f"Finished to copy partition results to {ip}...")
        with open(ip_config, 'r') as f:
            for line in f:
                ip = line.rstrip()
                logging.info(f"IP: {ip}")
                os.system(
                    f"ssh -o StrictHostKeyChecking=no -p {ssh_port} {ip} 'ls -lh {self.data_path}/*'"
                )

        # Step2: data dispatch
        partition_dir = os.path.join(self.data_path, 'parted_data')
        out_dir = os.path.join(self.data_path, 'partitioned')
        in_dir = os.path.join(self.data_path, "chunked-data")

        cmd = "python3 /dgl/tools/dispatch_data.py"
        cmd += f" --in-dir {in_dir}"
        cmd += f" --partitions-dir {partition_dir}"
        cmd += f" --out-dir {out_dir}"
        cmd += f" --ip-config {ip_config}"
        cmd += f" --ssh-port {os.environ['SSH_PORT']}"
        cmd += " --process-group-timeout 60"

        os.system(cmd)

        os.system(f"ls -lh {out_dir}")
