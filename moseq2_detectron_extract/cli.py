import atexit
import cProfile
import datetime
import json
import math
import sys
import os
from copy import Error, deepcopy
from pstats import Stats

import click
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from detectron2.data.catalog import MetadataCatalog
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.utils.visualizer import ColorMode, Visualizer

from moseq2_detectron_extract.io.annot import (
    augment_annotations_with_rotation, default_keypoint_names,
    read_annotations, register_dataset_metadata, register_datasets,
    show_dataset_info)
from moseq2_detectron_extract.io.image import write_image
from moseq2_detectron_extract.io.proc import (apply_roi, colorize_video, get_frame_features,
                                              hampel_filter,
                                              hampel_filter_forloop,
                                              instances_to_features,
                                              overlay_video, rotate_points)
from moseq2_detectron_extract.io.session import Session
from moseq2_detectron_extract.io.util import ensure_dir, get_last_checkpoint
from moseq2_detectron_extract.io.video import write_frames_preview
from moseq2_detectron_extract.model.config import (add_dataset_cfg,
                                                   get_base_config, load_config)
from moseq2_detectron_extract.model.model import Predictor, Trainer
from moseq2_detectron_extract.model.util import select_frames_kmeans

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



@cli.command(name='train', help='run training')
@click.argument('annot_file', nargs=1, type=click.Path(exists=True))
@click.argument('model_dir', nargs=1, type=click.Path(exists=False))
@click.option('--config', default=None, type=click.Path(), help="Model configuration to override base configuration")
@click.option('--replace-data-path', default=(None, None), type=(str, str), help="Replace data path")
@click.option('--resume', is_flag=True, help='Resume training from a previous checkpoint')
@click.option('--auto-cd', is_flag=True, help='treat model_dir as a base directory and create a child dir for this specific run')
@click.option('--min-height', default=0, type=int, help='Min mouse height from floor (mm)')
@click.option('--max-height', default=100, type=int, help='Max mouse height from floor (mm)')
def train(annot_file, model_dir, config, replace_data_path, resume, auto_cd, min_height, max_height):
    print()
    if resume:
        print("Resuming Model Training from: {}".format(model_dir))
        cfg = load_config(os.path.join(model_dir, "config.yaml"))

        if config is not None:
            print("WARNING: Ignoring --config because you opted to resume training from a previous checkpoint!")
    else:
        cfg = get_base_config()

        if config is not None:
            print("Attempting to load your extra --config and merge with the base configuration")
            cfg.merge_from_file(config)
        
        if auto_cd:
            cfg.OUTPUT_DIR = os.path.join(model_dir, datetime.datetime.now().strftime("%Y-%m-%dT%H-%M_%S"))
        else:
            cfg.OUTPUT_DIR = model_dir

        if os.path.exists(os.path.join(cfg.OUTPUT_DIR, "config.yaml")):
            print("Hmmm... it looks like there is already a model located here.... ({})".format(cfg.OUTPUT_DIR))
            print("If you wish to resume training, please use the --resume flag")
            print("Otherwise please change the `model_dir` argument to another location, or utilize the --auto-cd option")
            print("Exiting...")
            return
        
        print("Model output: {}".format(cfg.OUTPUT_DIR))

    ensure_dir(cfg.OUTPUT_DIR)

    intensity_scale = (max_height/255)


    annotations = read_annotations(annot_file, default_keypoint_names, mask_format=cfg.INPUT.MASK_FORMAT, replace_path=replace_data_path, rescale=intensity_scale)
    annotations = augment_annotations_with_rotation(annotations)
    print('Dataset information:')
    show_dataset_info(annotations)
    register_datasets(annotations, default_keypoint_names)


    if not resume:
        cfg = add_dataset_cfg(cfg)
        with open(os.path.join(cfg.OUTPUT_DIR, "config.yaml"), 'w') as f:
            f.write(cfg.dump())

    print(cfg)

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=resume)
    trainer.train()



@cli.command(name='infer', help='run inference')
@click.argument('model_dir', nargs=1, type=click.Path(exists=True))
@click.argument('input_file', nargs=1, type=click.Path(exists=True))
@click.option('--frame-trim', default=(0, 0), type=(int, int), help='Frames to trim from beginning and end of data')
@click.option('--batch-size', default=10, type=int, help='Number of frames for each model inference iteration')
@click.option('--chunk-size', default=1000, type=int, help='Number of frames for each processing iteration')
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
def infer(model_dir, input_file, frame_trim, batch_size, chunk_size, chunk_overlap, bg_roi_dilate, bg_roi_shape, bg_roi_index, bg_roi_weights, bg_roi_depth_range,
          bg_roi_gradient_filter, bg_roi_gradient_threshold, bg_roi_gradient_kernel, bg_roi_fill_holes, use_plane_bground, frame_dtype, output_dir,
          min_height, max_height, fps, crop_size, profile):
    print("") # Empty line to give some breething room

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

    # Prepare keypoint data output file
    kp_out_file = open(os.path.join(images_dir, 'keypoints.tsv'), 'w')
    kp_out_file_headers = ["Frame_Idx"]
    for i, kp in enumerate(MetadataCatalog.get("moseq_train").keypoint_names):
        kp_out_file_headers.extend([
            "{}_X".format(kp),
            "{}_Y".format(kp),
            "{}_S".format(kp)
        ])
    kp_out_file.write("{}\n".format("\t".join(kp_out_file_headers)))


    # Iterate Frames and write images
    last_frame = None
    for frame_idxs, raw_frames in tqdm.tqdm(session.iterate(chunk_size, chunk_overlap), desc='Processing batches'):
        raw_frames = bground_im - raw_frames
        raw_frames[raw_frames < min_height] = 0
        raw_frames[raw_frames > max_height] = max_height
        #raw_frames = (raw_frames / max_height) * 255 # rescale to use full gammitt
        raw_frames = raw_frames.astype(frame_dtype)
        raw_frames = apply_roi(raw_frames, roi)

        

        # Do the inference
        outputs = []
        for i in tqdm.tqdm(range(0, raw_frames.shape[0], batch_size), desc="Inferring", leave=False):
            outputs.extend(predictor((raw_frames[i:i+batch_size,:,:,None] / max_height) * 255)) # rescale to use full gammitt

        
        angles, centroids, masks = instances_to_features(outputs, raw_frames)
        angles = hampel_filter(angles, 7, 3)
        # centroids = hampel_filter_forloop(centroids, 50, 3)[0]



        rfs = raw_frames.shape
        scale = 2.0
        out_video = np.zeros([rfs[0], int(rfs[1]*scale), int(rfs[2]*scale), 3], dtype='uint8')
        cropped_frames = np.zeros((rfs[0], crop_size[0], crop_size[1], 3), dtype='uint8')
        border = (crop_size[1], crop_size[1], crop_size[0], crop_size[0])
        for i, (raw_frame, output, angle, centroid) in enumerate(tqdm.tqdm(zip(raw_frames, outputs, angles, centroids), desc="Postprocessing", leave=False, total=raw_frames.shape[0])):
            im = raw_frame[:,:,None].copy().astype('uint8')
            im = (im-min_height)/(max_height-min_height)
            im[im < min_height] = 0
            im[im > max_height] = max_height
            im = im * 255
            v = Visualizer(convert_image_to_rgb(im, "L"),
                   metadata=MetadataCatalog.get("moseq_train"),
                   scale=scale,
                   instance_mode=ColorMode.SEGMENTATION   # remove the colors of unsegmented pixels. This option is only available for segmentation models
            )
            instances = output["instances"].to("cpu")
            out = v.draw_instance_predictions(instances)
            out_video[i,:,:,:] = out.get_image()




            if len(instances) > 0:
                keypoints = instances.pred_keypoints[0, :, :].numpy()
                kp_out_data = [str(frame_idxs[i])]
                for kpi in range(keypoints.shape[0]):
                    kp_out_data.extend([
                        "{}".format(keypoints[kpi, 0]),
                        "{}".format(keypoints[kpi, 1]),
                        "{}".format(keypoints[kpi, 2])
                    ])
                kp_out_file.write("{}\n".format("\t".join(kp_out_data)))

                


                rot_mat = cv2.getRotationMatrix2D((crop_size[0] // 2, crop_size[1] // 2), angle, 1)

                xmin = int(centroid[0] - crop_size[1] // 2) + crop_size[1]
                xmax = int(centroid[0] + crop_size[1] // 2) + crop_size[1]
                ymin = int(centroid[1] - crop_size[0] // 2) + crop_size[0]
                ymax = int(centroid[1] + crop_size[0] // 2) + crop_size[0]
                #plt.imshow(raw_frame)
                #plt.scatter(x=[x_center], y=[y_center], marker='*', c='r')
                #plt.scatter(x=instances.pred_keypoints[0, :, 0], y=instances.pred_keypoints[0, :, 1], c=instances.pred_keypoints[0, :, 2], cmap='inferno', label=MetadataCatalog.get("moseq_train").keypoint_names)
                #plt.legend()
                #plt.show()
                use_frame = cv2.copyMakeBorder(raw_frame, *border, cv2.BORDER_CONSTANT, 0)
                cropped = cv2.warpAffine(use_frame[ymin:ymax, xmin:xmax], rot_mat, (crop_size[0], crop_size[1]))

                use_mask = cv2.copyMakeBorder(masks[i], *border, cv2.BORDER_CONSTANT, 0)
                cropped_mask = cv2.warpAffine(use_mask[ymin:ymax, xmin:xmax], rot_mat, (crop_size[0], crop_size[1]))
                #cropped = cv2.warpAffine(raw_frame[int(box[1]):int(box[3]), int(box[0]):int(box[2])], rot_mat, (crop_size[0], crop_size[1]))
                #cropped = cropped * cropped_mask # mask the cropped image
                cropped_frames[i, :, :, :] = colorize_video(cropped)
            else:
                tqdm.tqdm.write("WARNING: No instances found for frame #{}".format(frame_idxs[i]))
                kp_out_data = [str(frame_idxs[i])]
                for kp in MetadataCatalog.get("moseq_train").keypoint_names:
                    kp_out_data.extend(['N/A', 'N/A', 'N/A'])
                kp_out_file.write("{}\n".format("\t".join(kp_out_data)))


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





@cli.command(name='generate-dataset', help='Generate images from a dataset')
@click.argument('input_file', nargs=-1, type=click.Path(exists=True))
@click.option('--num-samples', default=100, type=int, help='Total number of samples to draw')
@click.option('--sample-method', default='uniform', type=click.Choice(['random', 'uniform', 'kmeans']), help='Method to sample the data. Random chooses a random sample of frames. Uniform will produce a temporally uniform sample. Kmeans performs clustering on downsampled frames.')
@click.option('--chunk-size', default=1000, type=int, help='Number of frames for each processing iteration')
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
@click.option('--output-dir', type=click.Path(), help='Output directory to save the results')
@click.option('--min-height', default=0, type=int, help='Min mouse height from floor (mm)')
@click.option('--max-height', default=100, type=int, help='Max mouse height from floor (mm)')
@click.option('--stream', default=['depth'], multiple=True, type=click.Choice(['depth', 'rgb']), help='Data type for processed frames')
@click.option('--output-label-studio', is_flag=True, help='Output label-studio files')
def generate_dataset(input_file, num_samples, sample_method, chunk_size, chunk_overlap, bg_roi_dilate, bg_roi_shape, bg_roi_index, bg_roi_weights, bg_roi_depth_range,
            bg_roi_gradient_filter, bg_roi_gradient_threshold, bg_roi_gradient_kernel, bg_roi_fill_holes, use_plane_bground, output_dir, min_height, max_height, 
            stream, output_label_studio):

    num_samples_per_file = int(np.ceil(num_samples / len(input_file)))
    parameters = deepcopy(locals())


    output_dir = ensure_dir(output_dir)
    images_dir = ensure_dir(os.path.join(output_dir, 'images'))
    info_dir = ensure_dir(os.path.join(output_dir, '.info'))

    if len(stream) == 0:
        stream = ['depth']
    stream = list(set(stream))


    output_info = []
    for in_file in tqdm.tqdm(input_file, desc='Datasets'):
        #load session
        #print('Processing: {}'.format(in_file))
        session = Session(in_file)

        session_info_dir = ensure_dir(os.path.join(info_dir, session.session_id))

        # Find image background and ROI
        bground_im, roi, true_depth = session.find_roi(bg_roi_dilate, bg_roi_shape, bg_roi_index, bg_roi_weights, bg_roi_depth_range,
                bg_roi_gradient_filter, bg_roi_gradient_threshold, bg_roi_gradient_kernel, bg_roi_fill_holes, use_plane_bground, cache_dir=session_info_dir)


        # Dump status information
        with open(os.path.join(session_info_dir, 'info.json'), 'w') as sf:
            json.dump({
                'parameters': parameters,
                'session_id': session.session_id,
                'metadata': session.load_metadata(),
                'true_depth': true_depth,
            }, sf, indent='\t')

        if sample_method == 'random':
            iterator = session.sample(num_samples_per_file, chunk_size=chunk_size, streams=stream)
        
        elif sample_method == 'uniform':
            step = session.nframes // num_samples_per_file
            iterator = session.index(np.arange(step, session.nframes, step), chunk_size=chunk_size, streams=stream)

        elif sample_method == 'kmeans':
            kmeans_selected_frames = select_frames_kmeans(session, num_samples_per_file, chunk_size=chunk_size, min_height=min_height, max_height=max_height)
            iterator = session.index(kmeans_selected_frames, chunk_size=chunk_size, streams=stream)
        
        else:
            raise Error('unknown sample_method "{}"'.format(sample_method))

        session_data = {}
        # Iterate Frames and write images
        for data in tqdm.tqdm(iterator, desc='Processing batches', leave=False):
            frame_idxs = data[0]
            
            for fidx in frame_idxs:
                session_data[fidx] = {
                    'data': {
                        'images': []
                    },
                    'meta_info': {
                        'frame_idx': int(fidx),
                        'session_id': session.session_id,
                        'true_depth': true_depth,
                        **session.load_metadata()
                    }
                }

            if 'depth' in stream:
                raw_frames = data[stream.index('depth')+1]
                raw_frames = bground_im - raw_frames
                raw_frames[raw_frames < min_height] = 0
                raw_frames[raw_frames > max_height] = max_height
                raw_frames = apply_roi(raw_frames, roi)

                for idx, rf in zip(frame_idxs, raw_frames):
                    dest = os.path.join(images_dir, '{}_{}_{}.png'.format(session.session_id, 'depth', idx))
                    write_image(dest, rf, scale=True, scale_factor=(0, max_height))
                    session_data[idx]['data']['depth_image'] = dest
                    session_data[idx]['data']['images'].append(dest)

            if 'rgb' in stream:
                rgb_frames = data[stream.index('rgb')+1]
                rgb_frames = apply_roi(rgb_frames, roi)

                for idx, rf in zip(frame_idxs, rgb_frames):
                    dest = os.path.join(images_dir, '{}_{}_{}.png'.format(session.session_id, 'rgb', idx))
                    write_image(dest, rf, scale=False, dtype='uint8')
                    session_data[idx]['data']['rgb_image'] = dest
                    session_data[idx]['data']['images'].append(dest)

        
        output_info.extend(list(session_data.values()))

    print('Wrote dataset to "{}" '.format(output_dir))

    if output_label_studio:
        ls_task_dest = os.path.join(output_dir, 'tasks.json')
        with open(ls_task_dest, 'w') as f:
            json.dump(output_info, f, indent='\t')
        print('Wrote label-studio tasks to "{}" '.format(ls_task_dest))



# end generate_dataset()

if __name__ == '__main__':
    cli()
