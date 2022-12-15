import logging
import os

from task import Task

class PartitionTask(Task):
    def __init__(self, data_store, data_name):
        self.data_store = data_store
        self.data_name = data_name

    def _prepare_data(self):
        workspace = os.environ.get('WORKSPACE', '/workspace')
        os.system(
            f"python3 /dgl/tests/regression/data_store.py"
            f" --data_store {self.data_store} "
            f" --data_name {self.data_name}"
            f" --output_dir {workspace}"
        )
        self.data_path = os.path.join(workspace, self.data_name)

    def _do_run(self):
        logging.info("Running partition task...")
        pass