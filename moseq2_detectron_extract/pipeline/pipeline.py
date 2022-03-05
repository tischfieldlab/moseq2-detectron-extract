from typing import List, Optional, Union

from moseq2_detectron_extract.pipeline.pipeline_step import PipelineStep
from moseq2_detectron_extract.pipeline.progress import ProcessProgress
from torch.multiprocessing import Queue, SimpleQueue


class Pipeline:

    def __init__(self) -> None:
        self.progress = ProcessProgress()
        self.steps: List[PipelineStep] = []
        self.input: SimpleQueue = SimpleQueue()

    def add_step(self, name: str, klass: type, in_queue: SimpleQueue, show_progress=True, num_listners: int=1, **kwargs) -> List[SimpleQueue]:
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
        for t in self.steps:
            t.start()
        self.progress.start()

    def shutdown(self):
        for t in self.steps:
            t.join()
        self.progress.shutdown()
        self.progress.join()
