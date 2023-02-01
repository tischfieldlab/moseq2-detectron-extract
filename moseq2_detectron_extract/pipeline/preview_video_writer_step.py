
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Optional

import numpy as np
from detectron2.data import MetadataCatalog

from moseq2_detectron_extract.io.util import gen_batch_sequence
from moseq2_detectron_extract.io.video import PreviewVideoWriter
from moseq2_detectron_extract.pipeline.pipeline_step import ProcessPipelineStep
from moseq2_detectron_extract.proc.keypoints import \
    load_keypoint_data_from_dict
from moseq2_detectron_extract.proc.proc import stack_videos
from moseq2_detectron_extract.viz import (ArenaView, CleanedFramesView,
                                          RotatedKeypointsView)

# pylint: disable=attribute-defined-outside-init

class PreviewVideoWriterStep(ProcessPipelineStep):
    ''' PipelineStep which writes a preview video
    '''
    def __init__(self, config: dict, name: Optional[str] = None, **kwargs) -> None:
        super().__init__(config, name, **kwargs)

        # We need to grab the metadata for the dataset here, or the data will not be available in the subprocess!!
        self.dset_meta = MetadataCatalog.get(self.config['dataset_name'])
        self.workers = 4

    def initialize(self):
        preview_video_dest = os.path.join(self.config['output_dir'], f"results_{self.config['bg_roi_index']:02d}.mp4")

        self.clean_frames_view = CleanedFramesView(scale=1.5, dset_meta=self.dset_meta)
        self.rot_kpt_view = RotatedKeypointsView(scale=1.5, dset_meta=self.dset_meta)
        self.arena_view = ArenaView(self.config['roi'], scale=2.0, vmin=self.config['min_height'], vmax=self.config['max_height'], dset_meta=self.dset_meta)

        self.video_pipe = PreviewVideoWriter(preview_video_dest,
                                             fps=self.config['fps'],
                                             vmin=self.config['min_height'],
                                             vmax=self.config['max_height'])


        self.load_rot_kpts = partial(load_keypoint_data_from_dict,
                                     keypoints=self.dset_meta.keypoint_names,
                                     coord_system='rotated',
                                     units='px',
                                     root='')
        self.load_ref_kpts = partial(load_keypoint_data_from_dict,
                                     keypoints=self.dset_meta.keypoint_names,
                                     coord_system='reference',
                                     units='px',
                                     root='')

        self.executor = ThreadPoolExecutor(self.workers)

    def finalize(self):
        self.executor.shutdown()
        self.video_pipe.close()

    def process(self, data):
        raw_frames = data['chunk']
        instances = data['inference']
        masks = data['mask_frames']
        clean_frames = data['depth_frames']
        rot_keypoints = self.load_rot_kpts(data['keypoints'])

        ref_masks = []
        ref_keypoints = []
        ref_boxes = []
        for i in range(len(instances)):
            frame_instances = instances[i]["instances"].to('cpu')
            ref_masks.append(frame_instances.pred_masks.numpy())
            ref_keypoints.append(frame_instances.pred_keypoints.numpy())
            ref_boxes.append(frame_instances.pred_boxes.tensor.numpy())
        ref_masks = np.array(ref_masks)
        ref_keypoints = np.array(ref_keypoints)
        ref_boxes = np.array(ref_boxes)
        frame_idxs = np.array(data['frame_idxs'])

        # field_video = self.arena_view.generate_frames(raw_frames=raw_frames, boxes=ref_boxes, keypoints=ref_keypoints, masks=ref_masks)
        # rc_kpts_video = self.rot_kpt_view.generate_frames(masks=masks, keypoints=rot_keypoints)
        # cln_depth_video = self.clean_frames_view.generate_frames(clean_frames=clean_frames, masks=masks)

        # proc_stack = stack_videos([cln_depth_video, rc_kpts_video], orientation='vertical')
        # out_video_combined = stack_videos([proc_stack, field_video], orientation='horizontal')

        futures = []
        nframes = raw_frames.shape[0]
        batch_size = math.ceil(nframes / self.workers)
        batches = [list(batch_idxs) for batch_idxs in gen_batch_sequence(nframes, batch_size)]
        for batch_idxs in batches:
            futures.append(self.executor.submit(self._process_chunk,
                                                frame_idxs[batch_idxs],
                                                raw_frames[batch_idxs],
                                                ref_boxes[batch_idxs],
                                                ref_keypoints[batch_idxs],
                                                ref_masks[batch_idxs],
                                                masks[batch_idxs],
                                                rot_keypoints[batch_idxs],
                                                clean_frames[batch_idxs]))

        for future in futures:
            while not future.done():
                time.sleep(0.1)

            frame_idxs, out_video_combined = future.result()
            self.video_pipe.write_frames(frame_idxs, out_video_combined)
            self.update_progress(frame_idxs.shape[0])

    def _process_chunk(self, frame_idxs, raw_frames, ref_boxes, ref_keypoints, ref_masks, masks, rot_keypoints, clean_frames):

        field_video = self.arena_view.generate_frames(raw_frames=raw_frames, boxes=ref_boxes, keypoints=ref_keypoints, masks=ref_masks)
        rc_kpts_video = self.rot_kpt_view.generate_frames(masks=masks, keypoints=rot_keypoints)
        cln_depth_video = self.clean_frames_view.generate_frames(clean_frames=clean_frames, masks=masks)

        proc_stack = stack_videos([cln_depth_video, rc_kpts_video], orientation='vertical')
        out_video_combined = stack_videos([proc_stack, field_video], orientation='horizontal')

        return frame_idxs, out_video_combined
