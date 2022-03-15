import os

import pandas as pd
from moseq2_detectron_extract.pipeline.pipeline_step import PipelineStep
from moseq2_detectron_extract.proc.util import slice_dict

# pylint: disable=attribute-defined-outside-init

class KeypointWriterStep(PipelineStep):
    ''' PipelineStep to write keypoint data to csv file
    '''

    def initialize(self):
        self.keypoint_data_dest = os.path.join(self.config['output_dir'], f"keypoints_{self.config['bg_roi_index']:02d}.tsv")
        self.kp_out_data = None


    def process(self, data) -> None:
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
            self.update_progress()

        if self.kp_out_data is None:
            self.kp_out_data = pd.DataFrame(kp_data)
        else:
            self.kp_out_data = pd.concat((self.kp_out_data, pd.DataFrame(kp_data)), ignore_index=True)

        self.kp_out_data.to_csv(self.kp_out_data, sep='\t', index=False)
