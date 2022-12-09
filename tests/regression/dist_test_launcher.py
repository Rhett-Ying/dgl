import os
import argparse

def prepare_dgl():
    os.system(
        "pip3 install --pre dgl -f https://data.dgl.ai/wheels-test/repo.html"
    )

def prepare_env():
    os.system('service ssh restart')

    workspace = os.environ.get('WORKSPACE', '/workspace')
    if not os.path.isdir(workspace):
        os.makedirs(workspace)
        os.environ['WORKSPACE'] = workspace
    os.environ['IP_CONFIG'] = os.path.join(workspace, 'ip_config.txt')
    os.environ['SSH_PORT'] = '2233'

def fetch_raw_data():
    workspace = os.environ.get('WORKSPACE', '/workspace')
    data_path = os.path.join(workspace, 'test_dataset')
    os.system(
        f'aws s3 sync s3://dgl-data-store/test_dataset {data_path}'
    )
    os.system(
        f'ls -lh {data_path}'
    )
    return data_path

def graph_partition(root_dir):
    num_parts = 4

    # Step1: graph partition
    in_dir = os.path.join(root_dir, "chunked-data")
    output_dir = os.path.join(root_dir, "parted_data")
    os.system(
        "python3 /dgl/tools/partition_algo/random_partition.py "
        "--in_dir {} --out_dir {} --num_partitions {}".format(
            in_dir, output_dir, num_parts
        )
    )

    # Step2: data dispatch
    partition_dir = os.path.join(root_dir, 'parted_data')
    out_dir = os.path.join(root_dir, 'partitioned')
    ip_config = os.environ['IP_CONFIG']

    cmd = "python3 /dgl/tools/dispatch_data.py"
    cmd += f" --in-dir {in_dir}"
    cmd += f" --partitions-dir {partition_dir}"
    cmd += f" --out-dir {out_dir}"
    cmd += f" --ip-config {ip_config}"
    cmd += f" --ssh-port {os.environ['SSH_PORT']}"
    cmd += " --process-group-timeout 60"

    os.system(cmd)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Distributed test launcher",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    _, _ = parser.parse_known_args()

    # DGL preparation
    #prepare_dgl() #public network is not accessible but works manually connect EC2.

    # export envs
    prepare_env()

    # generate ip_config.txt
    os.system(
        "bash /dgl/tests/regression/generate_ip_config.sh"
    )

    # fetch raw data
    data_path = fetch_raw_data()

    # graph partition
    graph_partition(data_path)
