import numpy as np
from moseq2_detectron_extract.pipeline.pipeline_step import PipelineStep
from moseq2_detectron_extract.proc.proc import crop_and_rotate_frame


class FrameCropStep(PipelineStep):
    ''' PipelineStep to crop and rotate frames
    '''

    def process(self, data) -> None:
        crop_size = self.config['crop_size']
        raw_frames = data['chunk']
        nframes = raw_frames.shape[0]
        centroids = data['features']['features']['centroid']
        angles = data['features']['features']['orientation']
        masks = data['features']['masks']
        num_instances = data['features']['num_instances']
        frame_idxs = data['frame_idxs']

        cropped_frames = np.zeros((nframes, crop_size[0], crop_size[1]), dtype='uint8')
        cropped_masks = np.zeros((nframes, crop_size[0], crop_size[1]), dtype='uint8')

        #self.reset_progress(nframes)
        for i in range(nframes):
            if num_instances[i] <= 0:
                self.write_message(f'WARN: No instances found for frame {frame_idxs[i]}')
            cropped_frames[i] = crop_and_rotate_frame(raw_frames[i], centroids[i], angles[i], crop_size)
            cropped_masks[i] = crop_and_rotate_frame(masks[i], centroids[i], angles[i], crop_size)
            self.update_progress()

        data['depth_frames'] = cropped_frames
        data['mask_frames'] = cropped_masks
        return data
