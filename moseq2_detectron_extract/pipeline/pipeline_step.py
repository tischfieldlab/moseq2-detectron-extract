import logging
import traceback
from multiprocessing.synchronize import Event as EventClass
from queue import Empty
from threading import Thread
from typing import Optional, List, Union, cast

import torch
from torch.multiprocessing import Event, Process, Queue


class PipelineStep():
    '''  Represents a single step in a Pipeline
        Takes a single input queue to work on and adds results to one or more output queues
    '''

    def __init__(self, config: dict, name: str, **kwargs) -> None:
        super().__init__()
        self.step_name = name
        self.is_producer = False
        self.shutdown_event: Union[EventClass, None] = None
        self.progress: Union[Queue, None] = None
        self.in_queue: Union[Queue, None] = None
        self.out_queue: List[Queue] = []
        self.is_complete: EventClass = Event()

        self.config = config

    def attach_progress(self, progress_queue: Queue):
        ''' Should be called only before pipeline start
            Attach the progress queue to this step
        '''
        self.progress = progress_queue

    def reset_progress(self, total: int):
        ''' Reset progress on this step

            Parameters:
            total (int): new total value for progress
        '''
        assert self.progress is not None
        self.progress.put({'total': total})

    def update_progress(self, incremental_progress: int=1):
        ''' Update the progress on this step

            Parameters:
            n (int): incremental progress made
        '''
        assert self.progress is not None
        self.progress.put({'update': incremental_progress})

    def write_message(self, message: str, level=logging.INFO, raise_exc=False):
        ''' Write a message to the progress message pump

            Parameters:
            message (str): message to write
        '''
        assert self.progress is not None
        self.progress.put({
            'message': message,
            'level': level,
            'raise': raise_exc
        })

    def flush_progress(self):
        ''' Flush any progress made on the reciever side
        '''
        assert self.progress is not None
        self.progress.put({'flush': True})

    def set_outputs(self, data):
        ''' Send data to the output queues

            Parameters:
            data (Any): data to send to output consumers
        '''
        for queue in self.out_queue:
            queue.put(data)

    def is_output_empty(self) -> bool:
        ''' Tell if the outputs have been consumed
        '''
        return all(q.empty() for q in self.out_queue)

    def signal_shutdown(self):
        ''' Signal that this step should stop processing data and shutdown
        '''
        if self.shutdown_event is not None:
            self.shutdown_event.set()

    def shutdown(self):
        ''' Perform cleanup actions
        '''
        for out_queue in self.out_queue:
            out_queue.close()
        if self.progress is not None:
            self.progress.close()

    @property
    def total_items(self) -> int:
        ''' Get the total number of items this step is expected to process
        '''
        return self.config['nframes']

    def run(self) -> None:
        ''' Implementation of run() '''
        try:
            # Run some setup
            torch.multiprocessing.set_sharing_strategy('file_system')
            self.reset_progress(self.total_items)
            if not self.is_producer:
                assert self.in_queue is not None
            assert self.shutdown_event is not None

            # Allow the step to initalize itself
            self.initialize()

            # Loop while we have not recieved a shutdown signal
            while not self.shutdown_event.is_set():

                # Possible the input queue is None (pure producer step)
                # if there is a input queue, try and get the next input data
                # if the value is None (sentinal value)
                #  - propagate the None to the next step
                #  - break out of the processing loop
                # if we get queue.Empty, continue and allow a re-check of the shutdown signal
                if self.is_producer:
                    data = None
                else:
                    try:
                        assert self.in_queue is not None
                        data = self.in_queue.get(block=True, timeout=0.1)
                        if data is None:
                            self.set_outputs(None)
                            self.is_complete.set()
                            break
                    except Empty:
                        continue

                out: Union[dict, None] = None
                out = self.process(cast(dict, data)) # pylint: disable=assignment-from-no-return
                self.set_outputs(out)
                self.flush_progress()

                # If this step is a producer step, and we produced None
                # that signals the producer is done producing, and we can
                # stop running this step.
                if self.is_producer and out is None:
                    self.is_complete.set()
                    break
        except Exception: # pylint: disable=broad-except
            msg = traceback.format_exc()
            self.write_message(msg, level=logging.CRITICAL, raise_exc=True)
            self.finalize()
            self.signal_shutdown()
        finally:
            self.flush_progress()
            self.is_complete.set()
            #self.shutdown()

    def initialize(self):
        ''' Implement to execute actions on first startup of this step
        '''
        pass # pylint: disable=unnecessary-pass

    def process(self, data: dict) -> Union[dict, None]:
        ''' Implement to process data on each new batch of data
        '''
        pass # pylint: disable=unnecessary-pass

    def finalize(self):
        ''' Implement to execute actions on final shutdown of this step
        '''
        pass # pylint: disable=unnecessary-pass


class ThreadPipelineStep(PipelineStep, Thread):
    ''' Pipeline step which runs in a thread '''
    pass # pylint: disable=unnecessary-pass

class ProcessPipelineStep(PipelineStep, Process):
    ''' Pipeline step which runs in a process '''
    pass # pylint: disable=unnecessary-pass

class ProducerPipelineStep(ThreadPipelineStep):
    ''' Pipeline step who only produces
    '''
    def __init__(self, config: dict, name: str, **kwargs) -> None:
        super().__init__(config, name, **kwargs)
        self.is_producer = True
        self.daemon = True
