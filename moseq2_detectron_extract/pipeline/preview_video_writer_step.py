
import os

import numpy as np
from moseq2_detectron_extract.io.video import PreviewVideoWriter
from moseq2_detectron_extract.pipeline.pipeline_step import PipelineStep
from moseq2_detectron_extract.proc.proc import colorize_video, overlay_video
from moseq2_detectron_extract.viz import draw_instances_fast, scale_depth_frames


class PreviewVideoWriterStep(PipelineStep):

    def initialize(self):
        preview_video_dest = os.path.join(self.config['output_dir'], 'results_{:02d}.mp4'.format(self.config['bg_roi_index']))
        self.video_pipe = PreviewVideoWriter(preview_video_dest, fps=self.config['fps'], vmin=self.config['min_height'], vmax=self.config['max_height'])

    def finalize(self):
        self.video_pipe.close()

    def process(self, data):
        raw_frames = data['chunk']
        instances = data['inference']
        rfs = raw_frames.shape
        scale = 2.0
        out_video = np.zeros((rfs[0], int(rfs[1]*scale), int(rfs[2]*scale), 3), dtype='uint8')

        #self.reset_progress(rfs[0])
        for i in range(rfs[0]):
            frame_instances = instances[i]["instances"].to('cpu')
            out_video[i,:,:,:] = draw_instances_fast(raw_frames[i,:,:,None].copy(), frame_instances, scale=scale, keypoint_names=self.config['keypoint_names'], keypoint_connection_rules=self.config['keypoint_connection_rules'])
            self.update_progress()

        out_video_combined = overlay_video(out_video, colorize_video(scale_depth_frames(data['depth_frames'] * data['mask_frames'], scale=1.5)))
        self.video_pipe.write_frames(data['frame_idxs'], out_video_combined)
