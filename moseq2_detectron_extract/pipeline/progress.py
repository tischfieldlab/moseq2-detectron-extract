import logging
from queue import Empty
from threading import Thread
from typing import List, Union

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

    def add(self, name='', show=True, **kwargs):
        q = Queue()
        prog = {
            'name': name,
            'show': show,
            'q': q
        }
        if not self.disabled and show:
            prog['tqdm'] = tqdm.tqdm(**kwargs)
        self.workers.append(prog)
        return q

    def get_tqdm(self, name: str) -> Union[tqdm.tqdm, None]:
        for w in self.workers:
            if w['name'] == name:
                return w['tqdm']
        return None

    def run(self) -> None:
        if self.disabled:
            return
        while not self.done:
            for worker in self.workers:
                try:
                    data = worker['q'].get_nowait()

                    if worker['show'] and 'total' in data:
                        worker['tqdm'].reset(total=data['total'])

                    if worker['show'] and 'update' in data:
                        worker['tqdm'].update(data['update'])

                    if 'message' in data:
                        #tqdm.tqdm.write(data['message'])
                        logging.info(data['message'])

                    if worker['show'] and 'flush' in data:
                        worker['tqdm'].refresh()

                except Empty:
                    pass
        for worker in self.workers:
            if 'tqdm' in worker:
                worker['tqdm'].close()
