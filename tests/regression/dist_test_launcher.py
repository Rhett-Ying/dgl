import os
import argparse
import logging
import sys

def func_wrapper(func):
    def wrap_func(*args, **kwargs):
        logging.info(f'{func.__name__} begin')
        result = func(*args, **kwargs)
        logging.info(f'{func.__name__} end')
        return result
    return wrap_func


@func_wrapper
def graph_partition():
    os.system(
        "python3 /dgl/tests/regression/graph_partition/main.py"
    )


@func_wrapper
def dist_train():
    os.system(
        "python3 /dgl/tests/regression/dist_train/main.py"
    )


@func_wrapper
def report_gen():
    os.system(
        "python3 /dgl/tests/regression/report_generator.py"
    )


@func_wrapper
def prepare_env():
    # restart ssh service to enable port 2233
    os.system(
        "service ssh restart"
    )

    # check and defines required envs
    workspace = os.environ.get("WORKSPACE", "/workspace")
    if not os.path.isdir(workspace):
        os.mkdir(workspace)
    os.environ["WORKSPACE"] = workspace
    os.environ["IP_CONFIG"] = os.path.join(workspace, "ip_config.txt")
    os.environ["SSH_PORT"] = "2233"

    # generate ip_config.txt
    os.system("bash /dgl/tests/regression/dist_env_setup.sh")


if __name__ == '__main__':
    fmt = "%(asctime)s %(levelname)s %(message)s"
    logging.basicConfig(format=fmt, level=logging.INFO)

    logging.info("-------------------------- DistTestLauncher -------------")

    parser = argparse.ArgumentParser(
        description="Distributed test launcher",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    _, _ = parser.parse_known_args()

    # prepare distributed compute environment
    prepare_env()

    # non-main nodes exits now
    if os.environ["AWS_BATCH_JOB_MAIN_NODE_INDEX"] != \
        os.environ["AWS_BATCH_JOB_NODE_INDEX"]:
        sys.exit(0)

    # graph partition
    graph_partition()

    # distributed train
    dist_train()

    # report generation
    report_gen()

    logging.info("Dist test launcher is done...")
