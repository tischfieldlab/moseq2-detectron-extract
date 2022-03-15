from typing import List
from functools import partial
from pytest import Session
from moseq2_detectron_extract.pipeline.pipeline_step import PipelineStep
from torch.multiprocessing import SimpleQueue

from moseq2_detectron_extract.proc.proc import prep_raw_frames


class ProduceFramesStep(PipelineStep):
    ''' PipelineStep to produce raw frames for processing
    '''

    def __init__(self, session: Session, config, in_queue: SimpleQueue, out_queue: List[SimpleQueue], **kwargs) -> None:
        super().__init__(config, in_queue, out_queue, name="Read Depth Data", **kwargs)
        self.session = session
        self.is_producer = True

    def initialize(self):
        self.prep_frames = partial( prep_raw_frames,
                                    bground_im=self.session.bground_im,
                                    roi=self.session.roi,
                                    vmin=self.config['min_height'],
                                    vmax=self.config['max_height'])
        self.iterator = enumerate(self.session.iterate(self.config['chunk_size'], self.config['chunk_overlap']))

    def process(self, data) -> None:
        try:
            i, (frame_idxs, raw_frames) = next(self.iterator)
            offset = self.config['chunk_overlap'] if i > 0 else 0
            raw_frames = self.prep_frames(raw_frames)
            data = { 'batch': i, 'chunk': raw_frames, 'frame_idxs': frame_idxs, 'offset': offset }

            self.update_progress(raw_frames.shape[0])
            return data
        except StopIteration:
            return None

        #     while not pipeline.input.empty():
        #         time.sleep(0.1)
        #     pipeline.input.put(shm)
        #     reader_pbar.put({'update': raw_frames.shape[0]})
        #     producer_progress = pipeline.progress.get_tqdm('producer')
        #     if producer_progress is not None:
        #         n_frames = (producer_progress.format_dict["n"] or 0)
        #         t_frames = producer_progress.format_dict["total"]
        #         s_elapsed = producer_progress.format_dict["elapsed"]
        #         msg = f'Processed {n_frames} / {t_frames} frames ({n_frames/t_frames:.2%}) in {timedelta(seconds=s_elapsed)}'
        #         logging.info(msg, extra={'nostream': True})
        # pipeline.input.put(None) # signal we are done
        # return data
