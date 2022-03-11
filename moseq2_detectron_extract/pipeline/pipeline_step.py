import traceback
from typing import List, Union, cast
from torch import Tensor

from torch.multiprocessing import Process, Queue, SimpleQueue


class PipelineStep(Process):
    '''  Represents a single step in a Pipeline
        Takes a single input queue to work on and adds results to one or more output queues
    '''

    def __init__(self, config: dict, in_queue: SimpleQueue, out_queue: List[SimpleQueue], progress: Queue, name: str=None, **kwargs) -> None:
        super().__init__(name=name, **kwargs)
        self.config = config
        self.progress = progress
        self.in_queue = in_queue
        self.out_queue = out_queue
        if not isinstance(out_queue, list):
            raise TypeError('expected List[SimpleQueue] (possibly empty) for parameter `out_queue`')
        self.reset_progress(config['nframes'])

    def reset_progress(self, total: int):
        ''' Reset progress on this step

            Parameters:
            total (int): new total value for progress
        '''
        self.progress.put({'total': total})

    def update_progress(self, incremental_progress: int=1):
        ''' Update the progress on this step

            Parameters:
            n (int): incremental progress made
        '''
        self.progress.put({'update': incremental_progress})

    def write_message(self, message: str):
        ''' Write a message to the progress message pump

            Parameters:
            message (str): message to write
        '''
        self.progress.put({'message': message})

    def flush_progress(self):
        ''' Flush any progress made on the reciever side
        '''
        self.progress.put({'flush': True})

    def set_outputs(self, data):
        ''' Send data to the output queues

            Parameters:
            data (Any): data to send to output consumers
        '''
        for queue in self.out_queue:
            queue.put(data)

    def __get_inputs(self):
        ''' For internal use!
            Get the next data item from the queue
            if we have any tensors, clone those over for reuse
        '''
        data = self.in_queue.get()
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, Tensor):
                    data[key] = value.clone()
        if isinstance(data, Tensor):
            data = data.clone()
        return data

    def run(self) -> None:
        try:
            self.initialize()
            while True:
                data = self.__get_inputs()
                if data is None:
                    self.set_outputs(None)
                    break

                out: Union[dict, None] = None
                try:
                    out = self.process(cast(dict, data)) # pylint: disable=assignment-from-no-return
                except Exception: # pylint: disable=broad-except
                    msg = traceback.format_exc()
                    self.write_message(msg)

                self.set_outputs(out)
                self.flush_progress()
            self.finalize()
        except Exception: # pylint: disable=broad-except
            msg = traceback.format_exc()
            self.write_message(msg)

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
