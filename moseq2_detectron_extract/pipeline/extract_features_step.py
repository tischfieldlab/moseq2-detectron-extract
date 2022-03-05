from functools import partial

from moseq2_detectron_extract.pipeline.pipeline_step import PipelineStep
from moseq2_detectron_extract.proc.keypoints import keypoints_to_dict
from moseq2_detectron_extract.proc.proc import instances_to_features
from moseq2_detectron_extract.proc.scalars import compute_scalars

# pylint: disable=attribute-defined-outside-init

class ExtractFeaturesStep(PipelineStep):
    ''' Pipeline step to extract features
    '''

    def initialize(self):
        self.compute_scalars = partial(compute_scalars,
            min_height=self.config['min_height'], max_height=self.config['max_height'], true_depth=self.config['true_depth'])
        self.compute_keypoints = partial(keypoints_to_dict, true_depth=self.config['true_depth'])

    def process(self, data):
        features = instances_to_features(data['inference'], data['chunk'])
        scalars = self.compute_scalars(data['chunk'] * features['masks'], features['features'])

        data['keypoints'] = self.compute_keypoints(features['keypoints'],
                                                   features['cleaned_frames'],
                                                   features['features']['centroid'],
                                                   features['features']['orientation'])
        data['features'] = features
        data['scalars'] = scalars
        return data
