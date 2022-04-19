from functools import partial

import numpy as np

from moseq2_detectron_extract.pipeline.pipeline_step import ProcessPipelineStep
from moseq2_detectron_extract.proc.keypoints import keypoints_to_dict
from moseq2_detectron_extract.proc.proc import crop_and_rotate_frame, instances_to_features
from moseq2_detectron_extract.proc.scalars import compute_scalars

# pylint: disable=attribute-defined-outside-init

class ProcessFeaturesStep(ProcessPipelineStep):
    ''' Pipeline step to extract features
    '''

    def initialize(self):
        self.crop = self.config['crop_size']
        true_depth = self.config['true_depth']
        self.compute_keypoints = partial(keypoints_to_dict, true_depth=true_depth)
        self.compute_scalars = partial(compute_scalars,
                                       min_height=self.config['min_height'],
                                       max_height=self.config['max_height'],
                                       true_depth=true_depth)

    def process(self, data: dict):
        data = self.__compute_features(data)
        data = self.__crop_and_rotate(data)
        return data


    def __compute_features(self, data):
        features = instances_to_features(data['inference'], data['chunk'])
        scalars = self.compute_scalars(data['chunk'] * features['masks'], features['features'])

        data['keypoints'] = self.compute_keypoints(features['keypoints'],
                                                   features['cleaned_frames'],
                                                   features['features']['centroid'],
                                                   features['features']['orientation'])
        data['features'] = features
        data['scalars'] = scalars
        #self.update_progress(data['chunk'].shape[0])
        return data

    def __crop_and_rotate(self, data) -> None:
        raw_frames = data['chunk']
        nframes = raw_frames.shape[0]
        centroids = data['features']['features']['centroid']
        angles = data['features']['features']['orientation']
        masks = data['features']['masks']
        num_instances = data['features']['num_instances']
        frame_idxs = data['frame_idxs']

        cropped_frames = np.zeros((nframes, self.crop[0], self.crop[1]), dtype='uint8')
        cropped_masks = np.zeros((nframes, self.crop[0], self.crop[1]), dtype='uint8')

        #self.reset_progress(nframes)
        for i in range(nframes):
            if num_instances[i] <= 0:
                self.write_message(f'WARN: No instances found for frame {frame_idxs[i]}')
            cropped_frames[i] = crop_and_rotate_frame(raw_frames[i], centroids[i], angles[i], self.crop)
            cropped_masks[i] = crop_and_rotate_frame(masks[i], centroids[i], angles[i], self.crop)
            self.update_progress()

        data['depth_frames'] = cropped_frames
        data['mask_frames'] = cropped_masks
        return data
