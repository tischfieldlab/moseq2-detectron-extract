import os
from typing import List, Union

import pandas as pd
from moseq2_detectron_extract.pipeline.pipeline_step import PipelineStep
from moseq2_detectron_extract.proc.util import slice_dict
from torch.multiprocessing import Queue


class KeypointWriterStep(PipelineStep):

    def initialize(self):
        self.kp_out_data = []

    def finalize(self):
        keypoint_data_dest = os.path.join(self.config['output_dir'], 'keypoints_{:02d}.tsv'.format(self.config['bg_roi_index']))
        pd.DataFrame(self.kp_out_data).to_csv(keypoint_data_dest, sep='\t', index=False)

    def process(self, data) -> None:
        keypoint_names = self.config['keypoint_names']
        nframes = data['chunk'].shape[0]
        self.update_progress(nframes)
        for i in range(nframes):
            self.kp_out_data.append({
                'Frame_Idx': data['frame_idxs'][i],
                'Flip': data['features']['flips'][i],
                'Centroid_X': data['features']['features']['centroid'][i][0],
                'Centroid_Y': data['features']['features']['centroid'][i][1],
                'Angle': data['features']['features']['orientation'][i],
                **slice_dict(data['keypoints'], i)
            })
            self.update_progress()
