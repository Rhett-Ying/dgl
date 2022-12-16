import logging

from task import Task

class TrainTask(Task):
    def __init__(self, args):
        pass

    def _prepare_data(self):
        raise RuntimeError("Not implemented...")

    def _do_run(self):
        raise RuntimeError("Not implemented...")

    def _print_metrics(self):
        raise RuntimeError("Not implemented...")
