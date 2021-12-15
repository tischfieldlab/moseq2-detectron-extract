import atexit
import cProfile
import datetime
import json
import logging
import os
import time
import uuid
import warnings
from copy import deepcopy
from pstats import Stats

import click
import h5py
import numpy as np
import pandas as pd
import tqdm
from click_option_group import optgroup
from detectron2.data.catalog import MetadataCatalog
from detectron2.utils.env import seed_all_rng

from moseq2_detectron_extract.extract import extract_session
from moseq2_detectron_extract.io.annot import (
    default_keypoint_names, read_annotations, register_dataset_metadata,
    register_datasets, replace_data_path_in_annotations, show_dataset_info,
    validate_annotations)
from moseq2_detectron_extract.io.image import write_image
from moseq2_detectron_extract.io.result import (create_extract_h5,
                                                write_extracted_chunk_to_h5)
from moseq2_detectron_extract.io.session import Session, Stream
from moseq2_detectron_extract.io.util import Tee, ensure_dir, setup_logging
from moseq2_detectron_extract.io.video import PreviewVideoWriter
from moseq2_detectron_extract.model import Evaluator, Predictor, Trainer
from moseq2_detectron_extract.model.config import (add_dataset_cfg,
                                                   get_base_config,
                                                   load_config)
from moseq2_detectron_extract.model.deploy import export_model
from moseq2_detectron_extract.model.util import (get_last_checkpoint,
                                                 get_specific_checkpoint)
from moseq2_detectron_extract.proc.keypoints import keypoints_to_dict
from moseq2_detectron_extract.proc.kmeans import select_frames_kmeans
from moseq2_detectron_extract.proc.proc import (colorize_video,
                                                crop_and_rotate_frame,
                                                instances_to_features,
                                                prep_raw_frames,
                                                scale_raw_frames, stack_videos)
from moseq2_detectron_extract.proc.roi import apply_roi
from moseq2_detectron_extract.proc.scalars import compute_scalars
from moseq2_detectron_extract.quality import find_outliers_h5
from moseq2_detectron_extract.viz import draw_instances_fast

warnings.filterwarnings("ignore", category=UserWarning, module='torch') # disable UserWarning: floor_divide is deprecated

orig_init = click.core.Option.__init__
def new_init(self, *args, **kwargs):
    orig_init(self, *args, **kwargs)
    self.show_default = True
# end new_init()
click.core.Option.__init__ = new_init

def enable_profiling():
    logging.info("Enabling profiling...")
    pr = cProfile.Profile()
    pr.enable()

    def exit():
        pr.disable()
        logging.info("Profiling completed")
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
@click.argument('annot_file', required=True, nargs=-1, type=click.Path(exists=True))
@click.argument('model_dir', nargs=1, type=click.Path(exists=False))
@click.option('--config', default=None, type=click.Path(), help="Model configuration to override base configuration, in yaml format.")
@click.option('--replace-data-path', multiple=True, default=[], type=(str, str), help="Replace path to data image items in `annot_file`. Specify <search> <replace>")
@click.option('--resume', is_flag=True, help='Resume training from a previous checkpoint')
@click.option('--auto-cd', is_flag=True, help='treat model_dir as a base directory and create a child dir for this specific run')
@click.option('--min-height', default=0, type=int, help='Min mouse height from floor (mm)')
@click.option('--max-height', default=100, type=int, help='Max mouse height from floor (mm)')
def train(annot_file, model_dir, config, replace_data_path, resume, auto_cd, min_height, max_height):
    logging.info("")
    if resume:
        logging.info("Resuming Model Training from: {}".format(model_dir))
        cfg = load_config(os.path.join(model_dir, "config.yaml"))

        if config is not None:
            logging.warning("WARNING: Ignoring --config because you opted to resume training from a previous checkpoint!")
    else:
        cfg = get_base_config()

        if config is not None:
            logging.info("Attempting to load your extra --config and merge with the base configuration")
            cfg.merge_from_file(config)

        if auto_cd:
            cfg.OUTPUT_DIR = os.path.join(model_dir, datetime.datetime.now().strftime("%Y-%m-%dT%H-%M_%S"))
        else:
            cfg.OUTPUT_DIR = model_dir

        if os.path.exists(os.path.join(cfg.OUTPUT_DIR, "config.yaml")):
            logging.info("Hmmm... it looks like there is already a model located here.... ({})".format(cfg.OUTPUT_DIR))
            logging.info("If you wish to resume training, please use the --resume flag")
            logging.info("Otherwise please change the `model_dir` argument to another location, or utilize the --auto-cd option")
            logging.info("Exiting...")
            return

        logging.info("Model output: {}".format(cfg.OUTPUT_DIR))

    ensure_dir(cfg.OUTPUT_DIR)
    tee = Tee(os.path.join(cfg.OUTPUT_DIR, 'train.log'), mode='a')
    tee.attach()

    seed_all_rng(None if cfg.SEED < 0 else cfg.SEED) # Seed the random number generators

    intensity_scale = (max_height/255)

    annotations = []
    for anot_f in annot_file:
        annot = read_annotations(anot_f, default_keypoint_names, mask_format=cfg.INPUT.MASK_FORMAT, rescale=intensity_scale)
        annotations.extend(annot)

    for search, replace in replace_data_path:
        replace_data_path_in_annotations(annotations, search, replace)
    validate_annotations(annotations)
    logging.info('Dataset information:')
    show_dataset_info(annotations)
    register_datasets(annotations, default_keypoint_names)


    if not resume:
        cfg = add_dataset_cfg(cfg)
        with open(os.path.join(cfg.OUTPUT_DIR, "config.yaml"), 'w') as f:
            f.write(cfg.dump())

    logging.info(cfg)

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=resume)
    trainer.train()



@cli.command(name='evaluate', help='run evaluation of a model on a test dataset')
@click.argument('model_dir', nargs=1, type=click.Path(exists=True))
@click.argument('annot_file', required=True, nargs=-1, type=click.Path(exists=True))
@click.option('--replace-data-path', multiple=True, default=[], type=(str, str), help="Replace path to data image items in `annot_file`. Specify <search> <replace>")
@click.option('--min-height', default=0, type=int, help='Min mouse height from floor (mm)')
@click.option('--max-height', default=100, type=int, help='Max mouse height from floor (mm)')
@click.option("--profile", is_flag=True)
def evaluate(model_dir, annot_file, replace_data_path, min_height, max_height, profile):
    logging.info("") # Empty line to give some breething room

    if profile:
        enable_profiling()

    logging.info('Loading model configuration....')
    register_dataset_metadata("moseq_train", default_keypoint_names)
    cfg = get_base_config()
    with open(os.path.join(model_dir, 'config.yaml'), 'r') as cfg_file:
        cfg = cfg.load_cfg(cfg_file)
    cfg.MODEL.WEIGHTS = get_last_checkpoint(model_dir)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6  # set a custom testing threshold
    cfg.TEST.DETECTIONS_PER_IMAGE = 1


    logging.info('Loading annotations....')
    intensity_scale = (max_height/255)
    annotations = []
    for anot_f in annot_file:
        annot = read_annotations(anot_f, default_keypoint_names, mask_format=cfg.INPUT.MASK_FORMAT, rescale=intensity_scale)
        annotations.extend(annot)

    for search, replace in replace_data_path:
        replace_data_path_in_annotations(annotations, search, replace)
    validate_annotations(annotations)

    logging.info('Dataset information:')
    show_dataset_info(annotations)
    register_datasets(annotations, default_keypoint_names, split=False)

    evaluator = Evaluator(cfg)
    evaluator()


@cli.command(name='extract', help='run extraction')
@click.argument('model_dir', nargs=1, type=click.Path(exists=True))
@click.argument('input_file', nargs=1, type=click.Path(exists=True))
@click.option('--checkpoint', default='last', help='Model checkpoint to load. Use "last" to load the last checkpoint.')
@click.option('--frame-trim', default=(0, 0), type=(int, int), help='Frames to trim from beginning and end of data')
@click.option('--batch-size', default=10, type=int, help='Number of frames for each model inference iteration')
@click.option('--chunk-size', default=1000, type=int, help='Number of frames for each processing iteration')
@click.option('--chunk-overlap', default=0, type=int, help='Frames overlapped in each chunk. Useful for cable tracking')
@optgroup.group('Background Detection', help='These options deal with background detection')
@optgroup.option('--bg-roi-dilate', default=(10, 10), type=(int, int), help='Size of the mask dilation (to include environment walls)')
@optgroup.option('--bg-roi-shape', default='ellipse', type=str, help='Shape to use for the mask dilation (ellipse or rect)')
@optgroup.option('--bg-roi-index', default=0, type=int, help='Index of which background mask(s) to use')
@optgroup.option('--bg-roi-weights', default=(1, .1, 1), type=(float, float, float), help='Feature weighting (area, extent, dist) of the background mask')
@optgroup.option('--bg-roi-depth-range', default=(650, 750), type=(float, float), help='Range to search for floor of arena (in mm)')
@optgroup.option('--bg-roi-gradient-filter', default=False, type=bool, help='Exclude walls with gradient filtering')
@optgroup.option('--bg-roi-gradient-threshold', default=3000, type=float, help='Gradient must be < this to include points')
@optgroup.option('--bg-roi-gradient-kernel', default=7, type=int, help='Kernel size for Sobel gradient filtering')
@optgroup.option('--bg-roi-fill-holes', default=True, type=bool, help='Fill holes in ROI')
@optgroup.option('--use-plane-bground', is_flag=True, help='Use a plane fit for the background. Useful for mice that don\'t move much')
@click.option('--frame-dtype', default='uint8', type=click.Choice(['uint8', 'uint16']), help='Data type for processed frames')
@click.option('--output-dir', default=None, help='Output directory to save the results h5 file')
@click.option('--min-height', default=0, type=int, help='Min mouse height from floor (mm)')
@click.option('--max-height', default=100, type=int, help='Max mouse height from floor (mm)')
@click.option('--fps', default=30, type=int, help='Frame rate of camera')
@click.option('--crop-size', default=(80, 80), type=(int, int), help='size of crop region')
@click.option("--profile", is_flag=True)
@click.option("--report-outliers", is_flag=True)
def extract(model_dir, input_file, checkpoint, frame_trim, batch_size, chunk_size, chunk_overlap, bg_roi_dilate, bg_roi_shape, bg_roi_index, bg_roi_weights, bg_roi_depth_range,
          bg_roi_gradient_filter, bg_roi_gradient_threshold, bg_roi_gradient_kernel, bg_roi_fill_holes, use_plane_bground, frame_dtype, output_dir,
          min_height, max_height, fps, crop_size, profile, report_outliers):

    print("") # Empty line to give some breething room

    if profile:
        enable_profiling()

    config_data = locals()
    config_data.update({
        'use_tracking_model': False,
        'flip_classifier': model_dir
    })

    session = Session(input_file, frame_trim=frame_trim)

    status_filename = extract_session(session=session, config=config_data)

    if report_outliers:
        logging.info("")
        logging.info("Searching for outlier frames....")
        result_filename = os.path.splitext(status_filename)[0] + '.h5'
        kpt_names = [kp for kp in default_keypoint_names if kp != 'TailTip']
        find_outliers_h5(result_filename, keypoint_names=kpt_names)



@cli.command(name='infer', help='run inference')
@click.argument('model_dir', nargs=1, type=click.Path(exists=True))
@click.argument('input_file', nargs=1, type=click.Path(exists=True))
@click.option('--checkpoint', default='last', help='Model checkpoint to load. Use "last" to load the last checkpoint.')
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
def infer(model_dir, input_file, checkpoint, frame_trim, batch_size, chunk_size, chunk_overlap, bg_roi_dilate, bg_roi_shape, bg_roi_index, bg_roi_weights, bg_roi_depth_range,
          bg_roi_gradient_filter, bg_roi_gradient_threshold, bg_roi_gradient_kernel, bg_roi_fill_holes, use_plane_bground, frame_dtype, output_dir,
          min_height, max_height, fps, crop_size, profile):
    logging.info("") # Empty line to give some breething room

    if profile:
        enable_profiling()

    config_data = locals()
    config_data.update({
        'use_tracking_model': False,
        'flip_classifier': model_dir,
    })

    status_dict = {
        'complete': False,
        'skip': False,
        'uuid': str(uuid.uuid4()),
        'metadata': '',
        'parameters': deepcopy(config_data)
    }

    session = Session(input_file, frame_trim=frame_trim)

    # set up the output directory
    if output_dir is None:
        output_dir = os.path.join(session.dirname, 'proc')
    else:
        output_dir = os.path.join(session.dirname, output_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    tee = Tee(os.path.join(output_dir, 'infer.log'))
    tee.attach()

    info_dir = os.path.join(output_dir, '.info')
    if not os.path.exists(info_dir):
        os.makedirs(info_dir)

    logging.info('Loading model....')
    register_dataset_metadata("moseq_train", default_keypoint_names)
    cfg = get_base_config()
    with open(os.path.join(model_dir, 'config.yaml'), 'r') as cfg_file:
        cfg = cfg.load_cfg(cfg_file)
    if checkpoint == 'last':
        logging.info(' -> Using last model checkpoint....')
        cfg.MODEL.WEIGHTS = get_last_checkpoint(model_dir)
    else:
        logging.info(f' -> Using model checkpoint at iteration {checkpoint}....')
        cfg.MODEL.WEIGHTS = get_specific_checkpoint(model_dir, checkpoint)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6  # set a custom testing threshold
    cfg.TEST.DETECTIONS_PER_IMAGE = 1
    predictor = Predictor(cfg)

    logging.info('Processing: {}'.format(input_file))
    # Find image background and ROI
    first_frame, bground_im, roi, true_depth = session.find_roi(bg_roi_dilate, bg_roi_shape, bg_roi_index, bg_roi_weights, bg_roi_depth_range,
            bg_roi_gradient_filter, bg_roi_gradient_threshold, bg_roi_gradient_kernel, bg_roi_fill_holes, use_plane_bground, cache_dir=info_dir)
    logging.info(f'Found true depth: {true_depth}')
    config_data.update({
        'true_depth': true_depth,
    })

    preview_video_dest = os.path.join(output_dir, '{}.mp4'.format('extraction'))
    video_pipe = PreviewVideoWriter(preview_video_dest, fps=fps, vmin=min_height, vmax=max_height)

    result_h5_dest = os.path.join(output_dir, '{}.h5'.format('result'))
    result_h5 = h5py.File(result_h5_dest, mode='w')
    create_extract_h5(result_h5, acquisition_metadata=session.load_metadata(), config_data=config_data, status_dict=status_dict,
                      nframes=session.nframes, roi=roi, bground_im=bground_im, first_frame=first_frame, timestamps=session.load_timestamps(Stream.Depth))

    times = {
        'prepare_data': [],
        'inference': [],
        'features': [],
        'draw_instances': [],
        'write_keypoints': [],
        'crop_rotate': [],
        'colorize': [],
        'write_video': []
    }

    # Iterate Frames and write images
    last_frame = None
    kp_out_data = []
    keypoint_names = MetadataCatalog.get("moseq_train").keypoint_names
    for i, (frame_idxs, raw_frames) in enumerate(tqdm.tqdm(session.iterate(chunk_size, chunk_overlap), desc='Processing batches')):
        offset = chunk_overlap if i > 0 else 0
        start = time.time()
        raw_frames = prep_raw_frames(raw_frames, bground_im=bground_im, roi=roi, vmin=min_height, vmax=max_height, scale=255)
        times['prepare_data'].append(time.time() - start)



        # Do the inference
        start = time.time()
        outputs = []
        for i in tqdm.tqdm(range(0, raw_frames.shape[0], batch_size), desc="Inferring", leave=False):
            outputs.extend(predictor(scale_raw_frames(raw_frames[i:i+batch_size,:,:,None], vmin=min_height, vmax=max_height)))
        times['inference'].append(time.time() - start)

        # Post process results and extract features
        start = time.time()
        features = instances_to_features(outputs, raw_frames)
        times['features'].append(time.time() - start)


        sub_times = {
            'draw_instances': [],
            'write_keypoints': [],
            'crop_rotate': [],
        }
        rfs = raw_frames.shape
        scale = 2.0
        out_video = np.zeros((rfs[0], int(rfs[1]*scale), int(rfs[2]*scale), 3), dtype='uint8')
        cropped_frames = np.zeros((rfs[0], crop_size[0], crop_size[1]), dtype='uint8')
        cropped_masks = np.zeros((rfs[0], crop_size[0], crop_size[1]), dtype='uint8')
        for i in tqdm.tqdm(range(raw_frames.shape[0]), desc="Postprocessing", leave=False):
            raw_frame = raw_frames[i]
            clean_frame = features['cleaned_frames'][i]
            mask = features['masks'][i]
            output = outputs[i]
            angle = features['features']['orientation'][i]
            centroid = features['features']['centroid'][i]
            flip = features['flips'][i]
            allosteric_keypoints = features['allosteric_keypoints'][i, 0]
            rotated_keypoints = features['rotated_keypoints'][i, 0]
            
            
            

            if len(instances) <= 0:
                tqdm.tqdm.write("WARNING: No instances found for frame #{}".format(frame_idxs[i]))

            start = time.time()
            kp_out_data.append({
                'Frame_Idx': frame_idxs[i],
                'Flip': flip,
                'Centroid_X': centroid[0],
                'Centroid_Y': centroid[1],
                'Angle': angle,
                **keypoints_to_dict(keypoint_names, allosteric_keypoints),
                **keypoints_to_dict(keypoint_names, rotated_keypoints, prefix='rot_')
            })
            sub_times['write_keypoints'].append(time.time() - start)

            instances = output["instances"].to('cpu')
            start = time.time()
            out_video[i,:,:,:] = draw_instances_fast(raw_frame[:,:,None].copy(), instances, scale=scale)
            sub_times['draw_instances'].append(time.time() - start)

            start = time.time()
            cropped = crop_and_rotate_frame(raw_frame, centroid, angle, crop_size)
            cropped_mask = crop_and_rotate_frame(mask, centroid, angle, crop_size)
            cropped = cropped * cropped_mask # mask the cropped image
            cropped_frames[i] = cropped
            cropped_masks[i] = cropped_mask
            sub_times['crop_rotate'].append(time.time() - start)

        results = {
            'chunk': raw_frames,
            'depth_frames': cropped_frames,
            'mask_frames': cropped_masks,
            'scalars': compute_scalars(raw_frames * features['masks'], features['features'], min_height=min_height, max_height=max_height, true_depth=true_depth),
            'flips': features['flips'],
            'parameters': None # only not None if EM tracking was used (we don't support that here)
        }

        write_extracted_chunk_to_h5(result_h5, results=results, frame_range=frame_idxs, offset=offset)



        start = time.time()
        out_video_combined = stack_videos([out_video, colorize_video(cropped_frames, vmax=255)], orientation='diagional')
        video_pipe.write_frames(frame_idxs, out_video_combined)
        times['write_video'].append(time.time() - start)

        for k, v in sub_times.items():
            times[k].append(np.sum(v))

    pd.DataFrame(kp_out_data).to_csv(os.path.join(output_dir, 'keypoints.tsv'), sep='\t', index=False)
    result_h5.close()
    video_pipe.close()

    logging.info('Processing Times:')
    for k, v in times.items():
        logging.info(f'{k}: {np.sum(v)}')
    logging.info(f'Total: {np.sum(list(times.values()))}')




@cli.command(name='generate-dataset', help='Generate images from a dataset')
@click.argument('input_file', nargs=-1, type=click.Path(exists=True))
@click.option('--num-samples', default=100, type=int, help='Total number of samples to draw')
@click.option('--indices', default=None, type=str, help='A comma separated list of indices, or path to a file containing one index per line. When --sample-method=list, the indicies to directly pick. When --sample-method=random or --sample-method=kmeans, limit selection to this set of indicies. Otherwise unused.')
@click.option('--sample-method', default='uniform', type=click.Choice(['random', 'uniform', 'kmeans', 'list']), help='Method to sample the data. Random chooses a random sample of frames. Uniform will produce a temporally uniform sample. Kmeans performs clustering on downsampled frames. List interprets --indices as a comma separated list of indices to extract.')
@click.option('--chunk-size', default=1000, type=int, help='Number of frames for each processing iteration')
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
def generate_dataset(input_file, num_samples, indices, sample_method, chunk_size, bg_roi_dilate, bg_roi_shape, bg_roi_index, bg_roi_weights, bg_roi_depth_range,
            bg_roi_gradient_filter, bg_roi_gradient_threshold, bg_roi_gradient_kernel, bg_roi_fill_holes, use_plane_bground, output_dir, min_height, max_height,
            stream, output_label_studio):

    setup_logging()

    if indices is not None:
        if os.path.exists(indices):
            with open(indices, mode='r') as ind_file:
                indices = sorted([int(l) for l in ind_file.readlines()])
            del ind_file
        else:
            indices = sorted([int(i) for i in indices.split(',')])

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
        #logging.info('Processing: {}'.format(in_file))
        session = Session(in_file)

        session_info_dir = ensure_dir(os.path.join(info_dir, session.session_id))

        # Find image background and ROI
        first_frame, bground_im, roi, true_depth = session.find_roi(bg_roi_dilate, bg_roi_shape, bg_roi_index, bg_roi_weights, bg_roi_depth_range,
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
            if indices is not None:
                seq = np.random.choice(indices, num_samples_per_file, replace=False)
                iterator = session.index(seq, chunk_size=chunk_size, streams=stream)
            else:
                iterator = session.sample(num_samples_per_file, chunk_size=chunk_size, streams=stream)

        elif sample_method == 'uniform':
            step = session.nframes // num_samples_per_file
            iterator = session.index(np.arange(step, session.nframes, step), chunk_size=chunk_size, streams=stream)

        elif sample_method == 'kmeans':
            kmeans_selected_frames = select_frames_kmeans(session, num_samples_per_file, indices=indices, chunk_size=chunk_size, min_height=min_height, max_height=max_height)
            iterator = session.index(kmeans_selected_frames, chunk_size=chunk_size, streams=stream)

        elif sample_method == 'list':
            iterator = session.index(indices, chunk_size=chunk_size, streams=stream)

        else:
            raise ValueError('Unknown sample_method "{}"'.format(sample_method))

        session_data = {}
        # Iterate Frames and write images
        for data in tqdm.tqdm(iterator, desc='Processing batches', leave=False):
            frame_idxs = data[0]
            
            for fidx in frame_idxs:
                session_data[fidx] = {
                    'data': {
                        'images': []
                    },
                    'meta': {
                        'frame_idx': int(fidx),
                        'session_id': session.session_id,
                        'true_depth': true_depth,
                        **session.load_metadata()
                    }
                }

            if 'depth' in stream:
                raw_frames = prep_raw_frames(data[stream.index('depth')+1], bground_im=bground_im, roi=roi, vmin=min_height, vmax=max_height)

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

    logging.info('Wrote dataset to "{}" '.format(output_dir))

    if output_label_studio:
        ls_task_dest = os.path.join(output_dir, 'tasks.json')
        if os.path.exists(ls_task_dest):
            logging.warn('label-studio tasks file "{}" seems to already exist! Will append the new tasks to this existing file'.format(ls_task_dest))
            with open(ls_task_dest, 'r') as f:
                existing_tasks = json.load(f)
                output_info = existing_tasks + output_info

        with open(ls_task_dest, 'w') as f:
            json.dump(output_info, f, indent='\t')
        logging.info('Wrote label-studio tasks to "{}" '.format(ls_task_dest))

# end generate_dataset()


@cli.command(name='dataset-info', help='interogate the dataset for information')
@click.argument('annot_file', required=True, nargs=-1, type=click.Path(exists=True))
@click.option('--replace-data-path', multiple=True, default=[], type=(str, str), help="Replace path to data image items in `annot_file`. Specify <search> <replace>")
@click.option('--min-height', default=0, type=int, help='Min mouse height from floor (mm)')
@click.option('--max-height', default=100, type=int, help='Max mouse height from floor (mm)')
def dataset_info(annot_file, replace_data_path, min_height, max_height):

    logging.info('Loading annotations....')
    intensity_scale = (max_height/255)
    annotations = []
    for anot_f in annot_file:
        annot = read_annotations(anot_f, default_keypoint_names, mask_format='polygon', rescale=intensity_scale)
        annotations.extend(annot)

    for search, replace in replace_data_path:
        replace_data_path_in_annotations(annotations, search, replace)
    validate_annotations(annotations)

    logging.info('Dataset information:')
    show_dataset_info(annotations)



@cli.command(name='compile', help='compile a model for deployment')
@click.argument('model_dir', nargs=1, type=click.Path(exists=True))
@click.argument('annot_file', required=True, nargs=-1, type=click.Path(exists=True))
@click.option('--replace-data-path', multiple=True, default=[], type=(str, str), help="Replace path to data image items in `annot_file`. Specify <search> <replace>")
@click.option('--min-height', default=0, type=int, help='Min mouse height from floor (mm)')
@click.option('--max-height', default=100, type=int, help='Max mouse height from floor (mm)')
@click.option('--checkpoint', default='last', help='Model checkpoint to load. Use "last" to load the last checkpoint.')
@click.option('--evaluate', is_flag=True, help='Run COCO evaluation metrics on supplied annotations.')
def dataset_info(model_dir, annot_file, replace_data_path, min_height, max_height, checkpoint, evaluate):

    logging.info('Loading model....')
    register_dataset_metadata("moseq_train", default_keypoint_names)
    cfg = get_base_config()
    with open(os.path.join(model_dir, 'config.yaml'), 'r') as cfg_file:
        cfg = cfg.load_cfg(cfg_file)

    if checkpoint == 'last':
        logging.info(' -> Using last model checkpoint....')
        cfg.MODEL.WEIGHTS = get_last_checkpoint(model_dir)
    else:
        logging.info(f' -> Using model checkpoint at iteration {checkpoint}....')
        cfg.MODEL.WEIGHTS = get_specific_checkpoint(model_dir, checkpoint)

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6  # set a custom testing threshold
    cfg.TEST.DETECTIONS_PER_IMAGE = 1


    logging.info('Loading annotations....')
    intensity_scale = (max_height/255)
    annotations = []
    for anot_f in annot_file:
        annot = read_annotations(anot_f, default_keypoint_names, mask_format=cfg.INPUT.MASK_FORMAT, rescale=intensity_scale)
        annotations.extend(annot)

    for search, replace in replace_data_path:
        replace_data_path_in_annotations(annotations, search, replace)
    validate_annotations(annotations)
    register_datasets(annotations, default_keypoint_names)

    logging.info('Exporting model....')
    export_model(cfg, model_dir, run_eval=evaluate)



@cli.command(name='find-outliers', help='find putative outlier frames')
@click.argument('result_h5', required=True, nargs=-1, type=click.Path(exists=True))
@click.option('--window', default=6, type=int, help='sliding window size for jumping algorithm')
@click.option('--threshold', default=10, type=float, help='threshold for jumping algorithm')
def find_outliers(result_h5, window, threshold):
    kpt_names = [kp for kp in default_keypoint_names if kp != 'TailTip']

    for h5_path in result_h5:
        find_outliers_h5(h5_path, keypoint_names=kpt_names, jump_win=window, jump_thresh=threshold)




if __name__ == '__main__':
    cli()
