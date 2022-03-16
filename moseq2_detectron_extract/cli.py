import datetime
import logging
import os

import click
from click_option_group import optgroup
from detectron2.utils.env import seed_all_rng
from tabulate import tabulate
from moseq2_detectron_extract.dataset import generate_dataset_for_sessions, write_label_studio_tasks

from moseq2_detectron_extract.extract import extract_session
from moseq2_detectron_extract.io.annot import (default_keypoint_names, load_annotations_helper,
                                               register_dataset_metadata,
                                               register_datasets)
from moseq2_detectron_extract.io.session import Session
from moseq2_detectron_extract.io.util import (
    attach_file_logger, click_monkey_patch_option_show_defaults,
    enable_profiling, ensure_dir, setup_logging)
from moseq2_detectron_extract.model import Evaluator, Trainer
from moseq2_detectron_extract.model.config import (add_dataset_cfg,
                                                   get_base_config,
                                                   load_config)
from moseq2_detectron_extract.model.deploy import export_model
from moseq2_detectron_extract.model.util import (get_available_device_info,
                                                 get_available_devices,
                                                 get_default_device,
                                                 get_last_checkpoint,
                                                 get_specific_checkpoint,
                                                 get_system_versions)
from moseq2_detectron_extract.proc.util import check_completion_status
from moseq2_detectron_extract.quality import find_outliers_h5

# import warnings
# warnings.filterwarnings('ignore', category=UserWarning, module='torch') # disable UserWarning: floor_divide is deprecated
# warnings.showwarning = warn_with_traceback
# np.seterr(all='raise')

if os.getenv('MOSEQ_DETECTRON_PROFILE', 'False').lower() in ('true', '1', 't'):
    enable_profiling()


# Show click option defaults
click_monkey_patch_option_show_defaults()

@click.group()
def cli():
    ''' Toolbox for training and using Detectron2 models for
        moseq raw data processing
    '''
    pass # pylint: disable=unnecessary-pass


# pylint: disable=unused-argument

@cli.command(name='train', help='Train a model')
@click.argument('annot_file', required=True, nargs=-1, type=click.Path(exists=True))
@click.argument('model_dir', nargs=1, type=click.Path(exists=False))
@click.option('--config', default=None, type=click.Path(), help='Model configuration to override base configuration, in yaml format.')
@click.option('--replace-data-path', multiple=True, default=[], type=(str, str),
    help='Replace path to data image items in `annot_file`. Specify <search> <replace>')
@click.option('--resume', is_flag=True, help='Resume training from a previous checkpoint')
@click.option('--auto-cd', is_flag=True, help='treat model_dir as a base directory and create a child dir for this specific run')
def train(annot_file, model_dir, config, replace_data_path, resume, auto_cd):
    ''' CLI entrypoint for model training '''
    setup_logging(add_defered_file_handler=True)
    logging.info('')
    if resume:
        logging.info(f'Resuming Model Training from: {model_dir}')
        cfg = load_config(os.path.join(model_dir, 'config.yaml'))

        if config is not None:
            logging.warning('WARNING: Ignoring --config because you opted to resume training from a previous checkpoint!')
    else:
        cfg = get_base_config()

        if config is not None:
            logging.info('Attempting to load your extra --config and merge with the base configuration')
            cfg.merge_from_file(config)

        if auto_cd:
            cfg.OUTPUT_DIR = os.path.join(model_dir, datetime.datetime.now().strftime('%Y-%m-%dT%H-%M_%S'))
        else:
            cfg.OUTPUT_DIR = model_dir

        if os.path.exists(os.path.join(cfg.OUTPUT_DIR, 'config.yaml')):
            logging.info(f'Hmmm... it looks like there is already a model located here.... ({cfg.OUTPUT_DIR})')
            logging.info('If you wish to resume training, please use the --resume flag')
            logging.info('Otherwise please change the `model_dir` argument to another location, or utilize the --auto-cd option')
            logging.info('Exiting...')
            return

        logging.info(f'Model training output directory: {cfg.OUTPUT_DIR}')

    ensure_dir(cfg.OUTPUT_DIR)
    attach_file_logger(os.path.join(cfg.OUTPUT_DIR, 'train.log'))

    seed_all_rng(None if cfg.SEED < 0 else cfg.SEED) # Seed the random number generators


    load_annotations_helper(annot_file,
                            replace_data_path,
                            mask_format=cfg.INPUT.MASK_FORMAT,
                            register=True,
                            show_info=True)


    if not resume:
        cfg = add_dataset_cfg(cfg)
        with open(os.path.join(cfg.OUTPUT_DIR, 'config.yaml'), 'w', encoding='utf-8') as config_file:
            config_file.write(cfg.dump())

    logging.info(cfg)

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=resume)
    trainer.train()



@cli.command(name='evaluate', help='Evaluate a model given a test dataset')
@click.argument('model_dir', nargs=1, type=click.Path(exists=True))
@click.argument('annot_file', required=True, nargs=-1, type=click.Path(exists=True))
@click.option('--replace-data-path', multiple=True, default=[], type=(str, str),
    help='Replace path to data image items in `annot_file`. Specify <search> <replace>')
def evaluate(model_dir, annot_file, replace_data_path):
    ''' CLI entrypoint for model evaluation '''
    setup_logging()
    logging.info('') # Empty line to give some breething room

    logging.info('Loading model configuration....')
    register_dataset_metadata('moseq_train', default_keypoint_names)
    cfg = get_base_config()
    with open(os.path.join(model_dir, 'config.yaml'), 'r', encoding='utf-8') as cfg_file:
        cfg = cfg.load_cfg(cfg_file)
    cfg.MODEL.WEIGHTS = get_last_checkpoint(model_dir)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6  # set a custom testing threshold
    cfg.TEST.DETECTIONS_PER_IMAGE = 1


    annotations = load_annotations_helper(annot_file,
                                          replace_data_path,
                                          mask_format=cfg.INPUT.MASK_FORMAT,
                                          register=False,
                                          show_info=True)
    register_datasets(annotations, default_keypoint_names, split=False)

    evaluator = Evaluator(cfg)
    evaluator()



@cli.command(name='extract', short_help='Extract a moseq session raw data')
@click.argument('model', nargs=1, type=click.Path(exists=True))
@click.argument('input_file', nargs=1, type=click.Path(exists=True))
@optgroup.group('Model Inference')
@optgroup.option('--device', default=get_default_device(), type=click.Choice(get_available_devices()), help='Device to run model inference on')
@optgroup.option('--checkpoint', default='last', help='Model checkpoint to load. Use "last" to load the last checkpoint')
@optgroup.option('--batch-size', default=10, type=int, help='Number of frames for each model inference iteration')
@optgroup.group('Background Detection')
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
@optgroup.group('Output')
@optgroup.option('--output-dir', default=None, help='Output directory to save the extraction output files')
@optgroup.option('--frame-dtype', default='uint8', type=click.Choice(['uint8', 'uint16']), help='Data type for processed frames')
@optgroup.option('--min-height', default=0, type=int, help='Min mouse height from floor (mm)')
@optgroup.option('--max-height', default=100, type=int, help='Max mouse height from floor (mm)')
@optgroup.option('--crop-size', default=(80, 80), type=(int, int), help='Size of crop region')
@optgroup.option('--report-outliers', is_flag=True, help='Report outliers in extracted data')
@optgroup.group('Input and Processing')
@optgroup.option('--frame-trim', default=(0, 0), type=(int, int), help='Frames to trim from beginning and end of data')
@optgroup.option('--chunk-size', default=1000, type=int, help='Number of frames for each processing iteration')
@optgroup.option('--chunk-overlap', default=0, type=int, help='Frames overlapped in each chunk')
@optgroup.option('--fps', default=30, type=int, help='Frame rate of camera')
def extract(model, input_file, device, checkpoint, batch_size, frame_trim, chunk_size, chunk_overlap, bg_roi_dilate, bg_roi_shape, bg_roi_index,
          bg_roi_weights, bg_roi_depth_range, bg_roi_gradient_filter, bg_roi_gradient_threshold, bg_roi_gradient_kernel, bg_roi_fill_holes,
          use_plane_bground, frame_dtype, output_dir, min_height, max_height, fps, crop_size, report_outliers):
    ''' Extract a moseq session with a trained detectron2 model

    \b
    MODEL is a path to a model, which could be:
        A) Path to a directory containing model files. It is expected that the directory contain a file `config.yaml`
           containing the model configuration. The model which is ultimatly loaded is affected by --checkpoint.
           This method allows dynamically setting the --device parameter.
        B) Path to a file with a `*.ts` extension whose contents are a compiled torchscript model.
           The parameters --device and --checkpoint have no effect.
    \b
    INPUT_FILE is a path to moseq raw depth data, which could be:
        A) Path to a compressed moseq session in tar.gz format which contains a depth.dat file,
           ex: /path/to/session_1234567890.tar.gz
        B) Path to an uncompressed moseq session depth.dat file,
           ex: /path/to/session_1234567890/depth.dat
    '''
    setup_logging(add_defered_file_handler=True)
    print('') # Empty line to give some breething room

    config_data = locals()
    config_data.update({
        'use_tracking_model': False,
        'flip_classifier': model
    })

    session = Session(input_file, frame_trim=frame_trim)

    status_filename = extract_session(session=session, config=config_data)

    if report_outliers:
        if not check_completion_status(status_filename):
            logging.info('')
            logging.info('Skipping search for outlier frames because sessoion extraction was not completed!')
        else:
            logging.info('')
            logging.info('Searching for outlier frames....')
            result_filename = os.path.splitext(status_filename)[0] + '.h5'
            kpt_names = [kp for kp in default_keypoint_names if kp != 'TailTip']
            find_outliers_h5(result_filename, keypoint_names=kpt_names)



@cli.command(name='generate-dataset', help='Generate dataset samples from a moseq session')
@click.argument('input_file', nargs=-1, type=click.Path(exists=True))
@optgroup.group('Sample Selection')
@optgroup.option('--num-samples', default=100, type=int, help='Total number of samples to draw')
@optgroup.option('--indices', default=None, type=str, help='A comma separated list of indices, or path to a file containing one index per line. '
    'When --sample-method=list, the indicies to directly pick. When --sample-method=random or --sample-method=kmeans, '
    'limit selection to this set of indicies. Otherwise unused.')
@optgroup.option('--sample-method', default='uniform', type=click.Choice(['random', 'uniform', 'kmeans', 'list']),
    help='Method to sample the data. Random chooses a random sample of frames. Uniform will produce a temporally uniform sample. '
    'Kmeans performs clustering on downsampled frames. List interprets --indices as a comma separated list of indices to extract.')
@optgroup.group('Background Detection')
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
@optgroup.group('Output')
@optgroup.option('--output-dir', type=click.Path(), help='Output directory to save the results')
@optgroup.option('--min-height', default=0, type=int, help='Min mouse height from floor (mm)')
@optgroup.option('--max-height', default=100, type=int, help='Max mouse height from floor (mm)')
@optgroup.option('--stream', default=['depth'], multiple=True, type=click.Choice(['depth', 'rgb']),
    help='Stream types to output, specify multiple times for multiple streams')
@optgroup.option('--output-label-studio', is_flag=True, help='Output label-studio files')
@optgroup.group('Input and Processing')
@optgroup.option('--chunk-size', default=1000, type=int, help='Number of frames for each processing iteration')
def generate_dataset(input_file, num_samples, indices, sample_method, chunk_size, bg_roi_dilate, bg_roi_shape, bg_roi_index, bg_roi_weights,
            bg_roi_depth_range, bg_roi_gradient_filter, bg_roi_gradient_threshold, bg_roi_gradient_kernel, bg_roi_fill_holes, use_plane_bground,
            output_dir, min_height, max_height, stream, output_label_studio):
    ''' CLI entrypoint for generating a dataset for model training '''
    setup_logging()

    if indices is not None:
        if os.path.exists(indices):
            with open(indices, mode='r', encoding='utf-8') as ind_file:
                indices = sorted([int(l) for l in ind_file.readlines()])
            del ind_file
        else:
            indices = sorted([int(i) for i in indices.split(',')])

    roi_params = {
        'bg_roi_dilate': bg_roi_dilate,
        'bg_roi_shape': bg_roi_shape,
        'bg_roi_index': bg_roi_index,
        'bg_roi_weights': bg_roi_weights,
        'bg_roi_depth_range': bg_roi_depth_range,
        'bg_roi_gradient_filter': bg_roi_gradient_filter,
        'bg_roi_gradient_threshold': bg_roi_gradient_threshold,
        'bg_roi_gradient_kernel': bg_roi_gradient_kernel,
        'bg_roi_fill_holes': bg_roi_fill_holes,
        'use_plane_bground': use_plane_bground,
    }

    if len(stream) == 0:
        stream = ['depth']
    stream = list(set(stream))

    output_info = generate_dataset_for_sessions(input_files=input_file,
                                                streams=stream,
                                                num_samples=num_samples,
                                                indices=indices,
                                                sample_method=sample_method,
                                                roi_params=roi_params,
                                                output_dir=output_dir,
                                                min_height=min_height,
                                                max_height=max_height,
                                                chunk_size=chunk_size)

    if output_label_studio:
        ls_task_dest = os.path.join(output_dir, 'tasks.json')
        write_label_studio_tasks(output_info, ls_task_dest)



@cli.command(name='dataset-info', short_help='Interogate datasets for information')
@click.argument('annot_file', required=True, nargs=-1, type=click.Path(exists=True))
@click.option('--replace-data-path', multiple=True, default=[], type=(str, str),
    help='Replace path to data image items in `annot_file`. Specify <search> <replace>')
def dataset_info(annot_file, replace_data_path):
    ''' Interrogate datasets and show statistics.

        \b
        Includes the following information:
         -> Number of annotations in each [sub-]dataset
         -> Total number of annotations across all datasets
         -> Size range of images in the dataset
         -> Statistics on the size and ratio of instance bounding boxes
         -> Statistics on the image pixel intensities
    '''
    setup_logging()
    load_annotations_helper(annot_file, replace_data_path, register=False, show_info=True)



@cli.command(name='compile-model', help='Compile a model for deployment')
@click.argument('model_dir', nargs=1, type=click.Path(exists=True))
@click.argument('annot_file', required=True, nargs=-1, type=click.Path(exists=True))
@click.option('--replace-data-path', multiple=True, default=[], type=(str, str),
    help='Replace path to data image items in `annot_file`. Specify <search> <replace>')
@click.option('--checkpoint', default='last', help='Model checkpoint to load. Use "last" to load the last checkpoint.')
@click.option('--device', default=get_default_device(), type=click.Choice(get_available_devices()), help='Device to compile model for.')
@click.option('--eval-model', is_flag=True, help='Run COCO evaluation metrics on supplied annotations.')
def compile_model(model_dir, annot_file, replace_data_path, checkpoint, device, eval_model):
    ''' CLI entrypoint for compiling a model to torchscript '''
    setup_logging()

    logging.info('Loading model....')
    register_dataset_metadata('moseq_train', default_keypoint_names)
    cfg = get_base_config()
    with open(os.path.join(model_dir, 'config.yaml'), 'r', encoding='utf-8') as cfg_file:
        cfg = cfg.load_cfg(cfg_file)

    if checkpoint == 'last':
        logging.info(' -> Using last model checkpoint....')
        cfg.MODEL.WEIGHTS = get_last_checkpoint(model_dir)
    else:
        logging.info(f' -> Using model checkpoint at iteration {checkpoint}....')
        cfg.MODEL.WEIGHTS = get_specific_checkpoint(model_dir, checkpoint)

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6  # set a custom testing threshold
    cfg.TEST.DETECTIONS_PER_IMAGE = 1

    logging.info(f' -> Setting device to \'{device}\'')
    cfg.MODEL.DEVICE = device

    load_annotations_helper(annot_file,
                            replace_data_path,
                            mask_format=cfg.INPUT.MASK_FORMAT,
                            register=True,
                            show_info=True)

    logging.info('Exporting model....')
    export_model(cfg, model_dir, run_eval=eval_model)



@cli.command(name='find-outliers', help='Find putative outlier frames in an extracted session')
@click.argument('result_h5', required=True, nargs=-1, type=click.Path(exists=True))
@click.option('--window', default=6, type=int, help='sliding window size for jumping algorithm')
@click.option('--threshold', default=10, type=float, help='threshold for jumping algorithm')
def find_outliers(result_h5, window, threshold):
    ''' CLI entrypoint for finding outliers in extracted session data '''
    setup_logging()
    kpt_names = [kp for kp in default_keypoint_names if kp != 'TailTip']

    for h5_path in result_h5:
        find_outliers_h5(h5_path, keypoint_names=kpt_names, jump_win=window, jump_thresh=threshold)



@cli.command(name='system-info', short_help='Show relevant system information')
def system_info():
    ''' Show relevant system information
    '''
    setup_logging()
    logging.info('')
    logging.info('System Framework Versions:')
    vdata = get_system_versions()
    logging.info(tabulate(vdata, headers='keys'))

    logging.info('\n')

    logging.info(f'System Devices (default is {get_default_device()}):')
    dinfo = get_available_device_info()
    if dinfo is not None:
        logging.info(tabulate(dinfo, headers='keys'))
    else:
        logging.info('No devices found')
    logging.info('\n')



if __name__ == '__main__':
    cli()
