import os

import h5py
import pandas as pd

from moseq2_detectron_extract.io.result import (create_extract_h5,
                                                write_extracted_chunk_to_h5)
from moseq2_detectron_extract.pipeline.pipeline_step import ProcessPipelineStep
from moseq2_detectron_extract.proc.util import slice_dict

# pylint: disable=attribute-defined-outside-init

class ResultWriterStep(ProcessPipelineStep):
    ''' Pipeline step to write results to h5 file and csv file (keypoints)
    '''

    def initialize(self):
        self.__init_csv()
        self.__init_h5()

    def finalize(self):
        self.__finalize_h5()

    def process(self, data):
        self.__process_csv(data)
        self.__process_h5(data)
        self.update_progress(data['chunk'].shape[0])


    ################################
    #    Writing h5 result file    #
    ################################
    def __init_h5(self):
        result_h5_dest = os.path.join(self.config['output_dir'], f"results_{self.config['bg_roi_index']:02d}.h5")
        self.h5_file = h5py.File(result_h5_dest, mode='w')
        create_extract_h5(self.h5_file,
                         config_data=self.config,
                         status_dict=self.config['status_dict'])

    def __process_h5(self, data):
        write_extracted_chunk_to_h5(self.h5_file, data)

    def __finalize_h5(self):
        self.h5_file.close()


    ###################################
    #    Writing keypoint csv file    #
    ###################################
    def __init_csv(self):
        self.keypoint_data_dest = os.path.join(self.config['output_dir'], f"keypoints_{self.config['bg_roi_index']:02d}.tsv")
        self.kp_out_data = None

    def __process_csv(self, data) -> None:
        nframes = data['chunk'].shape[0]

        kp_data = []
        for i in range(nframes):
            kp_data.append({
                'Frame_Idx': data['frame_idxs'][i],
                'Flip': data['features']['flips'][i],
                'Centroid_X': data['features']['features']['centroid'][i][0],
                'Centroid_Y': data['features']['features']['centroid'][i][1],
                'Angle': data['features']['features']['orientation'][i],
                **slice_dict(data['keypoints'], i)
            })

        if self.kp_out_data is None:
            self.kp_out_data = pd.DataFrame(kp_data)
        else:
            self.kp_out_data = pd.concat((self.kp_out_data, pd.DataFrame(kp_data)), ignore_index=True)

        self.kp_out_data.to_csv(self.keypoint_data_dest, sep='\t', index=False)
