import os
import argparse
import logging
import sys
import time
import importlib
#from task import TrainTask, PartitionTask

def func_wrapper(func):
    def wrap_func(*args, **kwargs):
        logging.info(f'{func.__name__} begin')
        result = func(*args, **kwargs)
        logging.info(f'{func.__name__} end')
        return result
    return wrap_func


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
    if os.environ["AWS_BATCH_JOB_MAIN_NODE_INDEX"] == \
        os.environ["AWS_BATCH_JOB_NODE_INDEX"]:
        os.environ["NODE_TYPE"] = "MAIN_NODE"
    else:
        os.environ["NODE_TYPE"] = "CHILD_NODE"

    # generate ip_config.txt
    os.system("bash /dgl/tests/regression/dist_env_setup.sh")


@func_wrapper
def create_task(task_type):
    task_mod = importlib.import_module('task')
    return getattr(task_mod, task_type)()


if __name__ == '__main__':
    fmt = "%(asctime)s %(levelname)s %(message)s"
    logging.basicConfig(format=fmt, level=logging.INFO)

    logging.info("-------------------------- DistTestLauncher -------------")

    parser = argparse.ArgumentParser(
        description="Distributed test launcher",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="task type: partition or train"
    )
    args, _ = parser.parse_known_args()

    # prepare distributed compute environment
    prepare_env()

    # run partition or train test
    task = create_task(args.task)
    task.run()

    # report generation
    report_gen()

    logging.info("Dist test launcher is done...")
