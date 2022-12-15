import os
import logging
import time
import re


class Task:
    def __init__(self):
        pass

    def run(self):
        self._prepare_data()
        self._continue_on_main_nodes_only()
        self._do_run()
        self._print_metrics()

    def _prepare_data(self):
        raise RuntimeError("Not implemented...")

    def _continue_on_main_nodes_only(self):
        if os.environ["AWS_BATCH_JOB_MAIN_NODE_INDEX"] != \
            os.environ["AWS_BATCH_JOB_NODE_INDEX"]:
            logging.info("Child node goes to sleep now...")
            time.sleep(60*60*24)
        logging.info("Main node continues...")

    def _do_run(self):
        raise RuntimeError("Not implemented...")

    def _print_metrics(self):
        raise RuntimeError("Not implemented...")


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
