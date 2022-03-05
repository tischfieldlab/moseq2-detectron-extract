from typing import List, Optional, Union

from moseq2_detectron_extract.pipeline.pipeline_step import PipelineStep
from moseq2_detectron_extract.pipeline.progress import ProcessProgress
from torch.multiprocessing import Queue, SimpleQueue


class Pipeline:

    def __init__(self) -> None:
        self.progress = ProcessProgress()
        self.steps: PipelineStep = []
        self.input: Queue = SimpleQueue()

    def add_step(self, name: str, klass: type, in_queue: Queue, show_progress=True, num_listners: int=1, **kwargs) -> Union[Queue, List[Queue], None]:
        # create output queue based on number of listners
        out_queue = None
        if num_listners == 1:
            out_queue: Queue = SimpleQueue()
        elif num_listners > 1:
            out_queue: List[Queue] = [SimpleQueue() for i in range(num_listners)]

        pbar = self.progress.add(desc=name, show=show_progress)
        step = klass(in_queue=in_queue, out_queue=out_queue, progress=pbar, **kwargs)
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
