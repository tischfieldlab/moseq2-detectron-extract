import logging
import sys
import threading
from queue import Empty
from time import time
from typing import List, Union

import tqdm
from torch.multiprocessing import SimpleQueue


class ProcessProgress(threading.Thread):
    ''' Class to track progress in multiprocessing workers
    '''
    def __init__(self) -> None:
        super().__init__(name='progress', daemon=True)
        self.workers: List[dict] = []
        self.done = False

    def shutdown(self):
        ''' Signal that this thread should shutdown
        '''
        self.done = True

    def add(self, name: str, show: bool=True, **kwargs) -> SimpleQueue:
        ''' Add a progress tracker

        Parameters:
        name (str): Name of this progress tracker, used as tqdm desc attribute
        show (bool): If True, display progress using tqdm progress bars
        **kwargs: additional arguments passed to tqdm constructor

        Returns:
        Queue - used to send messages for this progress
        '''
        queue = SimpleQueue()
        if self.get_tqdm(name) is not None:
            raise KeyError(f'Progress with name "{name}" has already been added! Names should be unique!')
        prog = {
            'name': name,
            'show': show,
            'q': queue
        }
        if show:
            prog['tqdm'] = tqdm.tqdm(**kwargs)
        self.workers.append(prog)
        return queue

    def get_tqdm(self, name: str) -> Union[tqdm.tqdm, None]:
        ''' Get the tqdm progress instance for a particular progress queue. If not shown, returns None

        Parameters:
        name (str): name of the progress queue to find

        Returns:
        tqdm.tqdm | None - if name is found and is shown, returns tqdm instance, otherwise None
        '''
        for worker in self.workers:
            if worker['name'] == name:
                return worker['tqdm']
        return None

    def run(self) -> None:
        ''' Main loop for this thread
        '''
        self.exception = None
        while not self.done:
            for worker in self.workers:
                try:
                    data = worker['q'].get_nowait()

                    if worker['show'] and 'total' in data:
                        worker['tqdm'].reset(total=data['total'])

                    if worker['show'] and 'update' in data:
                        worker['tqdm'].update(data['update'])

                    if 'message' in data:
                        if data.get('raise', False):
                            self.exception = RuntimeError(f'Worker "{worker["name"]}" raised an exception:\n{data["message"]}')
                            self.shutdown()
                        else:
                            logging.log(data.get('level', logging.INFO), data['message'])

                    if worker['show'] and 'flush' in data:
                        worker['tqdm'].refresh()

                except Empty:
                    pass
        for worker in self.workers:
            if 'tqdm' in worker:
                worker['tqdm'].close()

    def join(self, timeout=None):
        super().join(time)
        if self.exception is not None:
            raise self.exception
