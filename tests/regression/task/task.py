import os
import logging
import time


class Task:
    def __init__(self):
        pass

    def run(self):
        self._prepare_data()
        self._continue_on_main_nodes_only()
        self._do_run()

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
