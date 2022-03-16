import logging
from multiprocessing import Event
import threading
from queue import Empty
import time
from typing import List, TypedDict, Union

import tqdm
from torch.multiprocessing import Queue


# pylint: disable=attribute-defined-outside-init

class WorkerErrorInfo:
    ''' Data class holding some error information
    '''
    def __init__(self, name, message) -> None:
        self.name = name
        self.message = message

class WorkerError(RuntimeError):
    ''' Exception thrown for workers
    '''
    def __init__(self, *args: object, error_info) -> None:
        super().__init__(*args)
        self.error_info = error_info

class WorkerStats(TypedDict):
    ''' Holds stats about a worker
    '''
    total: int
    completed: int
    elapsed: float

class ProcessProgress(threading.Thread):
    ''' Class to track progress in multiprocessing workers
    '''
    __QUEUE_TYPE = Queue

    def __init__(self, problem_queue: Queue) -> None:
        super().__init__(name='progress', daemon=True)
        self.workers: List[dict] = []
        self.shutdown_event = Event()
        self.problem_queue = problem_queue

    def shutdown(self):
        ''' Signal that this thread should shutdown
        '''
        self.shutdown_event.set()

    def add(self, name: str, show: bool=True, **kwargs) -> Queue:
        ''' Add a progress tracker

        Parameters:
        name (str): Name of this progress tracker, used as tqdm desc attribute
        show (bool): If True, display progress using tqdm progress bars
        **kwargs: additional arguments passed to tqdm constructor

        Returns:
        Queue - used to send messages for this progress
        '''
        queue = self.__QUEUE_TYPE()
        if self.get_tqdm(name) is not None:
            raise KeyError(f'Progress with name "{name}" has already been added! Names should be unique!')
        prog = {
            'name': name,
            'show': show,
            'q': queue,
            'tqdm': tqdm.tqdm(disable=not show, **kwargs),
            'start': time.time(),
            'total': 0,
            'completed': 0
        }
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

    def get_stats(self, name: str) -> WorkerStats:
        ''' Get statistics for a worker, including:
            - the total number of items
            - number of completed items
            - time elapsed

        Parameters:
        name (str): name of the progress queue to find

        Returns:
        WorkerStats - stats about the worker
        '''
        for worker in self.workers:
            if worker['name'] == name:
                return {
                    'total': worker['total'],
                    'completed': worker['completed'],
                    'elapsed': time.time() - worker['start']
                }
        return None

    def run(self) -> None:
        ''' Main loop for this thread
        '''
        while not self.shutdown_event.is_set():
            for worker in self.workers:
                try:
                    data = worker['q'].get_nowait()

                    if 'total' in data:
                        worker['tqdm'].reset(total=data['total'])
                        worker['total'] = data['total']

                    if 'update' in data:
                        worker['tqdm'].update(data['update'])
                        worker['completed'] += data['update']


                    if 'message' in data:
                        if data.get('raise', False):
                            self.problem_queue.put(WorkerErrorInfo(worker["name"], data["message"]))
                            #self.shutdown()
                        else:
                            logging.log(data.get('level', logging.INFO), data['message'])

                    if 'flush' in data:
                        worker['tqdm'].refresh()

                except Empty:
                    pass
        for worker in self.workers:
            if 'tqdm' in worker:
                worker['tqdm'].close()
