import logging
import multiprocessing as mp
import traceback
from typing import List, Union

from torch.multiprocessing import Process, Queue, SimpleQueue


class PipelineStep(Process):

    def __init__(self, config: dict, in_queue: SimpleQueue, out_queue: List[SimpleQueue], progress: Queue, name: str=None, **kwargs) -> None:
        super().__init__(name=name, **kwargs)
        self.config = config
        self.progress = progress
        self.in_queue = in_queue
        self.out_queue = out_queue
        if not isinstance(out_queue, list):
            raise TypeError('expected List[Queue] (possibly empty) for parameter `out_queue`')
        self.reset_progress(config['nframes'])

    def reset_progress(self, total):
        self.progress.put({'total': total})

    def update_progress(self, n=1):
        self.progress.put({'update': n})

    def write_message(self, message):
        self.progress.put({'message': message})

    def flush_progress(self):
        self.progress.put({'flush': True})

    def set_outputs(self, data):
        for q in self.out_queue:
            q.put(data)

    def run(self) -> None:
        try:
            self.initialize()
            while True:
                data = self.in_queue.get()
                if data is None:
                    self.set_outputs(None)
                    break

                out = None
                try:
                    out = self.process(data)
                except Exception as e:
                    msg = traceback.format_exc()
                    self.write_message(msg)

                self.set_outputs(out)
                self.flush_progress()
            self.finalize()
        except Exception as e:
            msg = traceback.format_exc()
            self.write_message(msg)
            pass

    def initialize(self):
        pass

    def process(self, data) -> Union[dict, None]:
        pass

    def finalize(self):
        pass
