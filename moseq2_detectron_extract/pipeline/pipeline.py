from threading import Timer
import time
from typing import Callable, List, Union
import torch

from torch.multiprocessing import Queue, Event

from moseq2_detectron_extract.pipeline.pipeline_step import PipelineStep, ProcessPipelineStep, ThreadPipelineStep
from moseq2_detectron_extract.pipeline.progress import ProcessProgress, WorkerError


class Pipeline:
    ''' a multiprocessing pipeline composed of PipelineSteps
    '''
    __QUEUE_TYPE = Queue
    STOP_WAIT_SECS = 3

    def __init__(self) -> None:
        torch.multiprocessing.set_sharing_strategy('file_system')
        self.problem_queue = Queue()
        self.progress = ProcessProgress(self.problem_queue)
        self.steps: List[Union[ThreadPipelineStep, ProcessPipelineStep]] = []
        self.shutdown_event = Event()
        self.timers = []


    def link(self, producer: PipelineStep, *consumers: PipelineStep):
        ''' Link the outputs of produce to the inputs of consumer
        '''
        for consumer in consumers:
            _queue = self.__QUEUE_TYPE()
            producer.out_queue.append(_queue)
            consumer.in_queue = _queue


    def add_step(self, name: str, klass: type, show_progress=True, **kwargs) -> PipelineStep:
        ''' Add a step to this pipeline

        Parameters:
        name (str): Name of the step to add
        klass (type): The type of step to create
        in_queue (SimpleQueue): input data queue for the step
        show_progress (bool): If true, show the step's progress with a progress bar
        num_listners (int): Number of output queues to create and attach, return value will include this number of SimpleQueue's
        **kwargs: additional arguments passed to `klass` constructor

        Returns:
        The pipeline step
        '''
        # create the pipeline step
        step: PipelineStep = klass(name=name, **kwargs)
        step.shutdown_event = self.shutdown_event

        # create progress, which also serves as a message pump, and attach to step
        pbar = self.progress.add(name, desc=name, show=show_progress)
        step.attach_progress(pbar)

        # add and return the step
        self.steps.append(step)
        return step


    def add_timed_callback(self, interval: float, function: Callable[['Pipeline'], None]):
        ''' Call a function every `interval` seconds, with this Pipeline as the only parameter

        Parameters:
        interval (float): number of seconds between every call of `function`
        function (Callable[[Pipeline], None]): Function to be called
        '''
        self.timers.append(RepeatTimer(interval, function, args=(self,)))


    def is_running(self) -> bool:
        ''' Check if the pipline appears to be healthy
        '''
        in_shutdown_mode = self.shutdown_event.is_set()
        any_consumers_not_complete = any([not step.is_complete.is_set() for step in self.steps])
        if not in_shutdown_mode and any_consumers_not_complete:
            return True
        else:
            return False


    def start(self):
        ''' Start this pipeline, including all workers
        '''
        self.progress.start()
        for timer in self.timers:
            timer.start()

        for step in self.steps:
            step.start()


    def shutdown(self):
        ''' Shutdown this pipeline, including all workers
        '''
        # set shutdown event for all workers
        self.shutdown_event.set()

        end_time = time.time() + self.STOP_WAIT_SECS
        num_terminated = 0
        num_failed = 0

        # Wait up to STOP_WAIT_SECS for all processes to complete
        for step in self.steps:
            join_secs = max(0.0, min(end_time - time.time(), self.STOP_WAIT_SECS))
            step.join(join_secs)

        # Clear the steps list and _terminate_ any steps that have not yet exited
        while self.steps:
            step = self.steps.pop()
            if step.is_alive() and hasattr(step, 'terminate'):
                step.terminate()
                num_terminated += 1
            else:
                if hasattr(step, 'exitcode'):
                    exitcode = step.exitcode
                    if exitcode:
                        num_failed += 1

        # shutdown any timers
        for timer in self.timers:
            timer.cancel()

        # Shutdown the progress thread
        # - send shutdown signal
        # - collect any possible exceptions collected by the progress
        # - join the progress thread
        # - if any exceptions were collected, raise an exception (on the main thread)
        self.progress.shutdown()
        self.progress.join()
        exceptions = []
        while not self.problem_queue.empty():
            exceptions.append(self.problem_queue.get())

        if len(exceptions) > 0:
            raise WorkerError(error_info=exceptions)

        return num_failed, num_terminated



class RepeatTimer(Timer):
    ''' Extend threading.Timer to support repeating function call every `interval` seconds
    '''
    def run(self):
        while not self.finished.wait(self.interval):
            self.function(*self.args, **self.kwargs)
