import os

import h5py

from moseq2_detectron_extract.io.result import (create_extract_h5,
                                                write_extracted_chunk_to_h5)
from moseq2_detectron_extract.io.session import Stream
from moseq2_detectron_extract.pipeline.pipeline_step import PipelineStep

# pylint: disable=attribute-defined-outside-init

class ResultH5WriterStep(PipelineStep):
    ''' Pipeline step to write results to h5 file
    '''

    def initialize(self):
        self.session = self.config['session']
        self.timestamps = self.session.load_timestamps(Stream.DEPTH)
        self.acquisition_metadata = self.session.load_metadata()
        self.nframes = self.session.nframes
        self.roi = self.config['roi']
        self.bground_im = self.config['bground_im']
        self.first_frame = self.config['first_frame']
        self.status_dict = self.config['status_dict']
        result_h5_dest = os.path.join(self.config['output_dir'], f"results_{self.config['bg_roi_index']:02d}.h5")
        self.h5_file = h5py.File(result_h5_dest, mode='w')
        create_extract_h5(self.h5_file,
                         acquisition_metadata=self.acquisition_metadata,
                         config_data=self.config,
                         status_dict=self.status_dict,
                         nframes=self.nframes,
                         roi=self.roi,
                         bground_im=self.bground_im,
                         first_frame=self.first_frame,
                         timestamps=self.timestamps)

    def finalize(self):
        self.h5_file.close()

    def process(self, data):
        write_extracted_chunk_to_h5(self.h5_file, data)
        self.update_progress(data['chunk'].shape[0])
