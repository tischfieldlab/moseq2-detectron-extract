from queue import Empty
from threading import Thread
from typing import List

import tqdm
from torch.multiprocessing import Queue


class ProcessProgress(Thread):
    def __init__(self) -> None:
        super().__init__(name='progress', daemon=True)
        self.workers: List[dict] = []
        self.done = False
        self.disabled = False

    def shutdown(self):
        self.done = True

    def add(self, **kwargs):
        q = Queue()
        prog = { 'q': q }
        if not self.disabled:
            prog['tqdm'] = tqdm.tqdm(**kwargs)
        self.workers.append(prog)
        return q

    def run(self) -> None:
        if self.disabled:
            return
        while not self.done:
            for worker in self.workers:
                try:
                    data = worker['q'].get_nowait()
                    if 'total' in data:
                        worker['tqdm'].reset(total=data['total'])

                    if 'update' in data:
                        worker['tqdm'].update(data['update'])

                    if 'message' in data:
                        tqdm.tqdm.write(data['message'])
                    
                    if 'flush' in data:
                        worker['tqdm'].refresh()
                except Empty:
                    pass
        for worker in self.workers:
            worker['tqdm'].close()
