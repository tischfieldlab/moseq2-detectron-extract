import logging
import multiprocessing as mp
import traceback
from typing import List, Union

from torch.multiprocessing import Process, Queue


class PipelineStep(Process):

    def __init__(self, config: dict, in_queue: Queue, out_queue: Union[Queue, List[Queue], None], progress: Queue=None, name: str=None, **kwargs) -> None:
        super().__init__(name=name, **kwargs)
        self.config = config
        self.progress = progress
        self.in_queue = in_queue
        if isinstance(out_queue, (mp.queues.Queue, mp.queues.SimpleQueue)):
            self.out_queue = [out_queue]
        elif isinstance(out_queue, list):
            self.out_queue = out_queue
        elif out_queue is None:
            self.out_queue = []
        else:
            raise TypeError('expected Queue or List[Queue]')
        self.reset_progress(config['nframes'])

    def reset_progress(self, total):
        if self.progress is not None:
            self.progress.put({'total': total})

    def update_progress(self, n=1):
        if self.progress is not None:
            self.progress.put({'update': n})

    def write_message(self, message):
        if self.progress is not None:
            self.progress.put({'message': message})
        else:
            logging.info(message)

    def flush_progress(self):
        if self.progress is not None:
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
            logging.error(exc_info=True)
            pass

    def initialize(self):
        pass

    def process(self, data) -> Union[dict, None]:
        pass

    def finalize(self):
        pass
