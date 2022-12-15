import logging

from task import Task

class TrainTask(Task):
    def __init__(self, data_store, data_name):
        pass

    def _prepare_data(self):
        raise RuntimeError("Not implemented...")

    def _do_run(self):
        raise RuntimeError("Not implemented...")
