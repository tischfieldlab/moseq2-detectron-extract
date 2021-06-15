from moseq2_detectron_extract.io.util import get_last_checkpoint
from moseq2_detectron_extract.io.annot import register_dataset_metadata
from detectron2.data.catalog import MetadataCatalog
from moseq2_detectron_extract.io.proc import apply_roi, overlay_video, colorize_video
import click
import tqdm
from moseq2_detectron_extract.model.model import Predictor
from moseq2_detectron_extract.model.config import get_base_config, add_dataset_cfg
from moseq2_detectron_extract.io.session import Session
from moseq2_detectron_extract.io.video import write_frames_preview
import os
import numpy as np
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data.detection_utils import convert_image_to_rgb
from moseq2_detectron_extract.io.annot import default_keypoint_names
import cProfile
from pstats import Stats
import atexit
import math
import cv2

orig_init = click.core.Option.__init__
def new_init(self, *args, **kwargs):
    orig_init(self, *args, **kwargs)
    self.show_default = True
# end new_init()
click.core.Option.__init__ = new_init

def enable_profiling():
    print("Enabling profiling...")
    pr = cProfile.Profile()
    pr.enable()

    def exit():
        pr.disable()
        print("Profiling completed")
        with open('profiling_stats.txt', 'w') as stream:
            stats = Stats(pr, stream=stream)
            stats.strip_dirs()
            stats.sort_stats('time')
            stats.dump_stats('.prof_stats')
            stats.print_stats()
    atexit.register(exit)


@click.group()
def cli():
    pass



@cli.command(name='infer', help='run inference')
@click.argument('model_dir', nargs=1, type=click.Path(exists=True))
@click.argument('input_file', nargs=1, type=click.Path(exists=True))
@click.option('--frame-trim', default=(0, 0), type=(int, int), help='Frames to trim from beginning and end of data')
@click.option('--chunk-size', default=25, type=int, help='Number of frames for each processing iteration')
@click.option('--chunk-overlap', default=0, type=int, help='Frames overlapped in each chunk. Useful for cable tracking')
@click.option('--bg-roi-dilate', default=(10, 10), type=(int, int), help='Size of the mask dilation (to include environment walls)')
@click.option('--bg-roi-shape', default='ellipse', type=str, help='Shape to use for the mask dilation (ellipse or rect)')
@click.option('--bg-roi-index', default=0, type=int, help='Index of which background mask(s) to use')
@click.option('--bg-roi-weights', default=(1, .1, 1), type=(float, float, float), help='Feature weighting (area, extent, dist) of the background mask')
@click.option('--bg-roi-depth-range', default=(650, 750), type=(float, float), help='Range to search for floor of arena (in mm)')
@click.option('--bg-roi-gradient-filter', default=False, type=bool, help='Exclude walls with gradient filtering')
@click.option('--bg-roi-gradient-threshold', default=3000, type=float, help='Gradient must be < this to include points')
@click.option('--bg-roi-gradient-kernel', default=7, type=int, help='Kernel size for Sobel gradient filtering')
@click.option('--bg-roi-fill-holes', default=True, type=bool, help='Fill holes in ROI')
@click.option('--use-plane-bground', is_flag=True, help='Use a plane fit for the background. Useful for mice that don\'t move much')
@click.option('--frame-dtype', default='uint8', type=click.Choice(['uint8', 'uint16']), help='Data type for processed frames')
@click.option('--output-dir', default=None, help='Output directory to save the results h5 file')
@click.option('--min-height', default=0, type=int, help='Min mouse height from floor (mm)')
@click.option('--max-height', default=100, type=int, help='Max mouse height from floor (mm)')
@click.option('--fps', default=30, type=int, help='Frame rate of camera')
@click.option('--crop-size', default=(80, 80), type=(int, int), help='size of crop region')
@click.option("--profile", is_flag=True)
def infer(model_dir, input_file, frame_trim, chunk_size, chunk_overlap, bg_roi_dilate, bg_roi_shape, bg_roi_index, bg_roi_weights, bg_roi_depth_range,
          bg_roi_gradient_filter, bg_roi_gradient_threshold, bg_roi_gradient_kernel, bg_roi_fill_holes, use_plane_bground, frame_dtype, output_dir,
          min_height, max_height, fps, crop_size, profile):

    if profile:
        enable_profiling()

    print('Loading model....')
    register_dataset_metadata("moseq_train", default_keypoint_names)
    cfg = get_base_config()
    with open(os.path.join(model_dir, 'config.yaml'), 'r') as cfg_file:
        cfg = cfg.load_cfg(cfg_file)
    cfg.MODEL.WEIGHTS = get_last_checkpoint(model_dir)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6  # set a custom testing threshold
    cfg.TEST.DETECTIONS_PER_IMAGE = 1
    predictor = Predictor(cfg)

    print('Processing: {}'.format(input_file))
    session = Session(input_file, frame_trim=frame_trim)

    # set up the output directory
    if output_dir is None:
        output_dir = os.path.join(session.dirname, 'proc')
    else:
        output_dir = os.path.join(session.dirname, output_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    images_dir = os.path.join(output_dir, 'images')
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)

    info_dir = os.path.join(images_dir, '.info')
    if not os.path.exists(info_dir):
        os.makedirs(info_dir)

    # Find image background and ROI
    bground_im, roi, true_depth = session.find_roi(bg_roi_dilate, bg_roi_shape, bg_roi_index, bg_roi_weights, bg_roi_depth_range,
            bg_roi_gradient_filter, bg_roi_gradient_threshold, bg_roi_gradient_kernel, bg_roi_fill_holes, use_plane_bground, cache_dir=info_dir)

    video_pipe = None
    vid_tqdm_opts = {
        'leave': False,
        'disable': True,
    }


    # Iterate Frames and write images
    last_frame = None
    for frame_idxs, raw_frames in tqdm.tqdm(session.iterate(chunk_size, chunk_overlap), desc='Processing batches'):
        raw_frames = bground_im - raw_frames
        raw_frames[raw_frames < min_height] = 0
        raw_frames[raw_frames > max_height] = max_height
        #raw_frames = raw_frames.astype(frame_dtype)
        raw_frames = apply_roi(raw_frames, roi)

        # Do the inference
        outputs = predictor(raw_frames[:,:,:,None])



        rfs = raw_frames.shape
        scale = 2.0
        out_video = np.zeros([rfs[0], int(rfs[1]*scale), int(rfs[2]*scale), 3], dtype='uint8')
        cropped_frames = np.zeros((rfs[0], crop_size[0], crop_size[1], 3), dtype='uint8')
        for i, (raw_frame, output) in enumerate(zip(raw_frames, outputs)):
            im = raw_frame[:,:,None]
            v = Visualizer(convert_image_to_rgb(im, "L"),
                   metadata=MetadataCatalog.get("moseq_train"),
                   scale=scale,
                   instance_mode=ColorMode.SEGMENTATION   # remove the colors of unsegmented pixels. This option is only available for segmentation models
            )
            instances = output["instances"].to("cpu")
            out = v.draw_instance_predictions(instances)
            out_video[i,:,:,:] = out.get_image()

            try:
                slope, intercept = np.polyfit(instances.pred_keypoints[0, :7, 0], instances.pred_keypoints[0, :7, 1], 1)
            except:
                slope = 0
            rot_mat = cv2.getRotationMatrix2D((crop_size[0] // 2, crop_size[1] // 2), -np.rad2deg(math.atan(slope)), 1)
            box = instances.pred_boxes.tensor.numpy()[0]
            cropped = cv2.warpAffine(raw_frame[int(box[1]):int(box[3]), int(box[0]):int(box[2])], rot_mat, (crop_size[0], crop_size[1]))

            cropped_frames[i, :, :, :] = colorize_video(cropped)

        out_video_combined = overlay_video(out_video, cropped_frames)

        #if last_frame is None:
        #    last_frame = cropped[0, ...]

        video_pipe = write_frames_preview(
                os.path.join(images_dir, '{}.mp4'.format('extraction')),
                out_video_combined,
                pipe=video_pipe, close_pipe=False, fps=fps,
                frame_range=frame_idxs,
                depth_max=max_height, depth_min=min_height, tqdm_kwargs=vid_tqdm_opts)

    if video_pipe:
        video_pipe.stdin.close()
        video_pipe.wait()


if __name__ == '__main__':
    cli()
