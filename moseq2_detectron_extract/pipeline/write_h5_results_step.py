import os
from typing import List, Union

import h5py
from moseq2_detectron_extract.io.result import (create_extract_h5,
                                                write_extracted_chunk_to_h5)
from moseq2_detectron_extract.io.session import Session, Stream
from moseq2_detectron_extract.pipeline.pipeline_step import PipelineStep
from torch.multiprocessing import Queue


class ResultH5WriterStep(PipelineStep):
    
    def __init__(self, session: Session, roi, bground_im, first_frame, status_dict, config, in_queue: Queue, out_queue: Union[Queue, List[Queue], None], **kwargs) -> None:
        super().__init__(config, in_queue, out_queue, name="ResultH5", **kwargs)
        self.timestamps = session.load_timestamps(Stream.Depth)
        self.acquisition_metadata = session.load_metadata()
        self.nframes = session.nframes
        self.roi = roi
        self.bground_im = bground_im
        self.first_frame = first_frame
        self.config = config
        self.status_dict = status_dict

    def initialize(self):
        result_h5_dest = os.path.join(self.config['output_dir'], 'results_{:02d}.h5'.format(self.config['bg_roi_index']))
        self.h5 = h5py.File(result_h5_dest, mode='w')
        create_extract_h5(self.h5, acquisition_metadata=self.acquisition_metadata, config_data=self.config, status_dict=self.status_dict,
                        nframes=self.nframes, roi=self.roi, bground_im=self.bground_im, first_frame=self.first_frame, timestamps=self.timestamps)

    def finalize(self):
        self.h5.close()

    def process(self, data):
        write_extracted_chunk_to_h5(self.h5, data)
