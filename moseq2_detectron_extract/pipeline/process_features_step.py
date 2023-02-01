from functools import partial
import os
from typing import List

import numpy as np
from norfair import Detection, Tracker
from moseq2_detectron_extract.model.instance_logger import InstanceLogger

from moseq2_detectron_extract.pipeline.pipeline_step import ProcessPipelineStep
from moseq2_detectron_extract.proc.keypoints import keypoints_to_dict
from moseq2_detectron_extract.proc.proc import (crop_and_rotate_frame,
                                                instances_to_features)
from moseq2_detectron_extract.proc.scalars import compute_scalars
from detectron2.structures import Instances
from scipy.ndimage import center_of_mass


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
        self.tracker = Tracker(distance_function='euclidean',
                               distance_threshold=50,
                               initialization_delay=0,
                               hit_counter_max=3)

        self.instance_log = InstanceLogger(os.path.join(self.config['output_dir'], "instance_log.tsv"))

    def process(self, data: dict):
        data = self.__select_instances(data)
        data = self.__compute_features(data)
        data = self.__crop_and_rotate(data)
        return data

    # def matrix_nms(scores, masks, method='gauss', sigma=0.5):
    #     # scores: mask scores in descending order (N)
    #     # masks: binary masks (NxHxW)
    #     # method: ’linear’ or ’gauss’
    #     # sigma: std in gaussian method
    #     # reshape for computation: Nx(HW)
    #     masks = masks.reshape(N, HxW)
    #     # pre−compute the IoU matrix: NxN
    #     intersection = mm(masks, masks.T)
    #     areas = masks.sum(dim=1).expand(N, N)
    #     union = areas + areas.T − intersection
    #     ious = (intersection / union).triu(diagonal=1)
    #     # max IoU for each: NxN
    #     ious_cmax = ious.max(0)
    #     ious_cmax = ious_cmax.expand(N, N).T
    #     # Matrix NMS, Eqn.(4): NxN
    #     if method == ’gauss’: # gaussian
    #         decay = exp(−(ious^2 − ious_cmax^2) / sigma)
    #     else: # linear
    #         decay = (1 − ious) / (1 − ious_cmax)
    #         # decay factor: N
    #         decay = decay.min(dim=0)
    #     return scores ∗ decay

    # def __soft_nms_mask_instances(self, instances: Instances, method: Literal['linear', 'gauss']='linear', sigma: float=0.5) -> Instances:
    #     ''' Implementation of soft non-maximum suppression using mask-basked IOU
    #         see: https://arxiv.org/pdf/2003.10152.pdf
    #     '''

    #     N = len(instances)
    #     if N <= 1:
    #         return instances

    #     scores = instances.scores
    #     # reshape for computation: Nx(HW)
    #     masks = instances.pred_masks.reshape(N, -1).int()

    #     # pre−compute the IoU matrix: NxN
    #     intersection = np.matmul(masks, masks.T)
    #     areas = masks.sum(dim=1).expand(N, N)
    #     union = areas + areas.T - intersection
    #     ious = (intersection / union).triu(diagonal=1)

    #     # max IoU for each: NxN
    #     ious_cmax = ious.max(0).values
    #     ious_cmax = ious_cmax.expand(N, N).T

    #     # Matrix NMS, Eqn.(4): NxN
    #     if method == 'gauss': # gaussian
    #         decay = np.exp(-(ious^2 - ious_cmax^2) / sigma)
    #     else: # linear
    #         decay = (1 - ious) / (1 - ious_cmax)
    #         # decay factor: N
    #         decay = decay.min(dim=0).values
    #     return scores * decay


    def __nms_mask_instances(self, instances: Instances, iou_threshold: float=0.5) -> Instances:
        ''' Implementation of non-maximum suppression using mask-based IOU, rather than box-based IOU

            I found that box-based IOU performs poorly with mouse images, since the tail can make the box very large
            and producing low IOU for two boxes on the same mouse. Computing the IOU on masks instead greatly increases
            the IOU for the same instance.
            see: https://pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
        '''
        # if there are no instances or only a single, return the instances
        if len(instances) <= 1:
            return instances

        # process the instances in order of score
        idxs = np.argsort(instances.scores.numpy())

        # initialize the list of picked indexes
        pick = []

        # keep looping while some indexes still remain in the indexes list
        while len(idxs) > 0:
            # grab the last index in the indexes list and add the
            # index value to the list of picked indexes
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            try:
                # compute the IoU matrix: NxN
                masks = instances.pred_masks.clone()[idxs].reshape(len(idxs), -1).int()
                intersection = np.matmul(masks, masks.T)
                areas = masks.sum(dim=1).expand(len(idxs), len(idxs))
                union = areas + areas.T - intersection
                ious = (intersection / union).triu(diagonal=1)

                # delete all indexes from the index list that have iou over threshold
                ious_over_thresh = np.where(ious > iou_threshold)[0]
                idxs_to_delete = np.unique(np.concatenate(([last], ious_over_thresh)))
                idxs = np.delete(idxs, idxs_to_delete)

            except Exception as ex:
                pass

        # return only the instances that were picked
        return instances[pick]


    def __instances_to_detections(self, instances: Instances) -> List[Detection]:
        '''Convert a set instances to a list of Detections'''
        detections = []
        for i in range(len(instances)):
            instance = instances[i]
            center = np.array(center_of_mass(instance.pred_masks.cpu().numpy()[0]))
            data = {
                'index': i,
                'instance': instance
            }
            detections.append(Detection(center, data=data))
        return detections


    def __select_instances(self, data):
        
        for frame_idx, frame_instances in zip(data['frame_idxs'], data['inference']):
            #self.__soft_nms_mask_instances(frame_instances['instances'])
            frame_instances['instances'] = self.__nms_mask_instances(frame_instances['instances'])
            self.instance_log.log_instances(frame_idx, frame_instances['instances'])

            # perform tracking
            tracked_objects = self.tracker.update(detections=self.__instances_to_detections(frame_instances['instances']))

            if len(tracked_objects) <= 1:
                continue


            # filter out detections without any "live" points - i.e. keep only objects observed in the current frame
            filtered_tracked_objects = filter(lambda to: to.live_points.any() , tracked_objects)
            # sort the tracked objects by age
            sorted_tracked_objects = sorted(filtered_tracked_objects, key=lambda item: item.age)

            # while we have tracked objects, pick up to `expected_instances` number of objects
            selected_instances = []
            while len(selected_instances) < self.config['expected_instances'] and len(sorted_tracked_objects) > 0:
                selected_instances.append(sorted_tracked_objects.pop().last_detection.data["instance"])

            frame_instances['instances'] = Instances.cat(selected_instances)
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

    def __crop_and_rotate(self, data):
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
