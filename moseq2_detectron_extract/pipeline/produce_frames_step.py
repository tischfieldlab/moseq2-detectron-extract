import time
from functools import partial
from moseq2_detectron_extract.io.session import Session, Stream
from moseq2_detectron_extract.pipeline.pipeline_step import ProducerPipelineStep

from moseq2_detectron_extract.proc.proc import prep_raw_frames

# pylint: disable=attribute-defined-outside-init

class ProduceFramesStep(ProducerPipelineStep):
    ''' PipelineStep to produce raw frames for processing
    '''

    def __init__(self, session: Session, config: dict, name: str = None, **kwargs) -> None:
        super().__init__(config, name, **kwargs)
        self.session = session

    def initialize(self):
        self.prep_frames = partial( prep_raw_frames,
                                    bground_im=self.session.bground_im,
                                    roi=self.session.roi,
                                    vmin=self.config['min_height'],
                                    vmax=self.config['max_height'])
        self.iterator = self.session.iterate(self.config['chunk_size'], self.config['chunk_overlap'])
        self.iterator.attach_filter(stream=Stream.DEPTH, filterer=self.prep_frames)
        self.enumerator = enumerate(self.iterator)

    def process(self, data) -> None:
        try:
            i, (frame_idxs, raw_frames) = next(self.enumerator)
            offset = self.config['chunk_overlap'] if i > 0 else 0
            data = {
                'batch': i,
                'chunk': raw_frames,
                'frame_idxs': frame_idxs,
                'offset': offset
            }

            while not self.shutdown_event.is_set() and not self.is_output_empty():
                # we only want to produce a new batch once the cosumer has taken
                time.sleep(0.1)

            self.update_progress(raw_frames.shape[0])
            return data
        except StopIteration:
            return None
