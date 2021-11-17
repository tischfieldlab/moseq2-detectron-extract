
import os
from functools import partial
from typing import List, Union

import numpy as np
from moseq2_detectron_extract.io.video import PreviewVideoWriter
from moseq2_detectron_extract.pipeline.pipeline_step import PipelineStep
from moseq2_detectron_extract.proc.keypoints import \
    load_keypoint_data_from_dict
from moseq2_detectron_extract.proc.proc import (colorize_video,
                                                scale_raw_frames, stack_videos)
from moseq2_detectron_extract.proc.roi import get_roi_contour
from moseq2_detectron_extract.viz import (draw_instances_fast, draw_keypoints,
                                          draw_mask, scale_depth_frames)
from torch.multiprocessing import Queue


class PreviewVideoWriterStep(PipelineStep):

    def __init__(self, roi, config, in_queue: Queue, out_queue: Union[Queue, List[Queue], None], **kwargs) -> None:
        super().__init__(config, in_queue, out_queue, name="ResultH5", **kwargs)
        self.roi = roi

    def initialize(self):
        preview_video_dest = os.path.join(self.config['output_dir'], 'results_{:02d}.mp4'.format(self.config['bg_roi_index']))
        self.video_pipe = PreviewVideoWriter(preview_video_dest, fps=self.config['fps'], vmin=self.config['min_height'], vmax=self.config['max_height'])

        self.iscale = partial(scale_raw_frames, vmin=self.config['min_height'], vmax=self.config['max_height'])

        self.scale = 2.0
        self.roi_contours = get_roi_contour(self.roi, crop=True)
        self.draw_instances = partial(draw_instances_fast, roi_contour=self.roi_contours, scale=self.scale, keypoint_names=self.config['keypoint_names'], keypoint_connection_rules=self.config['keypoint_connection_rules'])

        self.load_rot_kpts = partial(load_keypoint_data_from_dict, keypoints=self.config['keypoint_names'], coord_system='rotated', units='px', root='')
        self.draw_keypoints = partial(draw_keypoints, keypoint_names=self.config['keypoint_names'], keypoint_connection_rules=self.config['keypoint_connection_rules'])

    def finalize(self):
        self.video_pipe.close()

    def process(self, data):
        raw_frames = data['chunk']
        instances = data['inference']
        masks = data['mask_frames']
        clean_frames = data['depth_frames']
        rfs = raw_frames.shape
        keypoints = self.load_rot_kpts(data['keypoints'])

        field_video = np.zeros((rfs[0], int(rfs[1]*self.scale), int(rfs[2]*self.scale), 3), dtype='uint8')

        rckv_width = int(clean_frames.shape[2] * 1.5)
        rckv_height = int(clean_frames.shape[1] * 1.5)
        rot_crop_keypoints_video = np.zeros((clean_frames.shape[0], rckv_height, rckv_width, 3), dtype='uint8')
        rot_crop_keypoints_origin = (int(rckv_width // 2), int(rckv_height // 2))
        for i in range(rfs[0]):
            frame_instances = instances[i]["instances"].to('cpu')
            field_video[i,:,:,:] = self.draw_instances(self.iscale(raw_frames[i,:,:,None].copy()), frame_instances)

            rot_crop_keypoints_video[i,:,:,:] = draw_mask(rot_crop_keypoints_video[i,:,:,:], masks[i], alpha=0.7)
            rot_crop_keypoints_video[i,:,:,:] = self.draw_keypoints(rot_crop_keypoints_video[i,:,:,:], keypoints[i], origin=rot_crop_keypoints_origin, scale=1.5)
            self.update_progress()

        cleaned_depth = colorize_video(scale_depth_frames(clean_frames * masks, scale=1.5))
        proc_stack = stack_videos([cleaned_depth, rot_crop_keypoints_video], orientation='vertical')
        out_video_combined = stack_videos([proc_stack, field_video], orientation='horizontal')
        self.video_pipe.write_frames(data['frame_idxs'], out_video_combined)
