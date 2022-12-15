import argparse
import os
import logging


def func_wrapper(func):
    def wrap_func(*args, **kwargs):
        logging.info(f'{func.__name__} begin')
        result = func(*args, **kwargs)
        logging.info(f'{func.__name__} end')
        return result
    return wrap_func


@func_wrapper
def graph_partition(root_dir, num_parts):
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

    os.system(
        f"ls -lh {out_dir}"
    )


@func_wrapper
def dist_part_pipeline(dataset, num_parts):
    data_path = os.environ["DATA_PATH"]
    graph_partition(data_path, num_parts)

    logging.info(
        f"Finished distributed partition pipeline test for dataset[{dataset}]"
        f" with {num_parts} partitions."
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Distributed graph partition test suite",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="target dataset name to partition"
    )
    parser.add_argument(
        "--num_parts",
        type=int,
        required=True,
        help="target number of partitions"
    )
    args, _ = parser.parse_known_args()

    # distributed partition pipeline
    dist_part_pipeline(args.dataset, args.num_parts)

    logging.info(
        f"Graph partition test[Dataset:{args.dataset}, "
        f"num_parts:{args.num_parts}] is done..."
    )
