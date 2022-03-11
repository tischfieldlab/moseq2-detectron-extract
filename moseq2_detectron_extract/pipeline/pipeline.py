from typing import List
import torch

from torch.multiprocessing import SimpleQueue

from moseq2_detectron_extract.pipeline.pipeline_step import PipelineStep
from moseq2_detectron_extract.pipeline.progress import ProcessProgress


class Pipeline:
    ''' a multiprocessing pipeline composed of PipelineSteps
    '''

    def __init__(self) -> None:
        torch.multiprocessing.set_sharing_strategy('file_system')
        self.progress = ProcessProgress()
        self.steps: List[PipelineStep] = []
        self.input: SimpleQueue = SimpleQueue()

    def add_step(self, name: str, klass: type, in_queue: SimpleQueue, show_progress=True, num_listners: int=1, **kwargs) -> List[SimpleQueue]:
        ''' Add a step to this pipeline

        Parameters:
        name (str): Name of the step to add
        klass (type): The type of step to create
        in_queue (SimpleQueue): input data queue for the step
        show_progress (bool): If true, show the step's progress with a progress bar
        num_listners (int): Number of output queues to create and attach, return value will include this number of SimpleQueue's
        **kwargs: additional arguments passed to `klass` constructor

        Returns:
        List[SimpleQueue]: with number of elements equal to `num_listners`
        '''
        # create output queue based on number of listners
        out_queue: List[SimpleQueue] = [SimpleQueue() for i in range(num_listners)]

        # create progress, which also serves as a message pump
        pbar = self.progress.add(desc=name, show=show_progress)

        # create the pipeline step, attaching queues and progress
        step = klass(in_queue=in_queue, out_queue=out_queue, progress=pbar, **kwargs)

        # add the step and return the output queue[s]
        self.steps.append(step)
        return out_queue

    def start(self):
        ''' Start this pipeline, including all workers
        '''
        for step in self.steps:
            step.start()
        self.progress.start()

    def shutdown(self):
        ''' Shutdown this pipeline, including all workers
        '''
        for step in self.steps:
            step.join()
        self.progress.shutdown()
        self.progress.join()
