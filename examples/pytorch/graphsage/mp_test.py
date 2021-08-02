import multiprocessing
from multiprocessing import Process, Queue, Value
import logging
from queue import Empty
import time

def process_dl(dl_output, dl_done):
    logger = multiprocessing.log_to_stderr()
    logger.setLevel(logging.INFO)
    logger.info("thread_tf get started...")
    limit = 5
    for i in range(13):
        while True:
            if dl_output.qsize() > limit:
                #print("--- process_dl sleeps...")
                time.sleep(0.1)
                continue
            else:
                break
        dl_output.put(i)
    dl_done.value = True

def run():
    dl_output = Queue()
    dl_done = Value('b', False)
    Process(target=process_dl, args=(dl_output, dl_done, ), daemon=True).start()
    while not dl_done.value or not dl_output.empty():
        try:
            item = dl_output.get(timeout=0.1)
            print(item)
        except Empty:
            continue
        #print("item: ".format(item))
    
if __name__ == '__main__':
    run()