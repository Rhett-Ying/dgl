import os
import argparse
import logging
import sys
import time
import importlib.util
#from task import TrainTask, PartitionTask

def func_wrapper(func):
    def wrap_func(*args, **kwargs):
        logging.info(f'{func.__name__} begin')
        result = func(*args, **kwargs)
        logging.info(f'{func.__name__} end')
        return result
    return wrap_func


@func_wrapper
def prepare_env():
    # restart ssh service to enable port 2233
    os.system(
        "service ssh restart"
    )

    # install latest DGL nightly build
    if 'DIST_TEST_SKIP_INSTALL' not in os.environ:
        os.system(
            'pip3 install --pre dgl -f https://data.dgl.ai/wheels-test/repo.html'
        )
        os.system(
            "python3 -c 'import dgl;print(dgl.__version__)'"
        )
        logging.info('Latest DGL nightly build(CPU) is installed...')

    os.environ['DGL_ROOT_DIR'] = '/dgl'
    if 'DIST_TEST_SKIP_FETCH' not in os.environ:
        os.system(
            "git clone https://github.com/Rhett-Ying/dgl.git --branch dist_aws_batch /dgl_"
        )
        os.environ['DGL_ROOT_DIR'] = '/dgl_'
        logging.info('Latest DGL branch for test is fetched...')


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
    bin_path = os.path.join(
        os.environ['DGL_ROOT_DIR'],
        'tests/regression/dist_env_setup.sh'
    )
    os.system(f"bash {bin_path}")


@func_wrapper
def create_task(task_type):
    #[TODO]
    '''
    mod_path = os.path.join(
        os.environ['DGL_ROOT_DIR'],
        'tests/regression'
    )
    mod_name = 'task'
    spec = importlib.util.spec_from_file_location(mod_name, mod_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return getattr(module, task_type)()
    '''
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


    logging.info("Dist test launcher is done...")
