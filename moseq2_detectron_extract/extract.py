import logging
import os
import time
import uuid
from copy import deepcopy
from datetime import timedelta

from moseq2_detectron_extract.io.annot import (
    default_keypoint_colors, default_keypoint_connection_rules,
    default_keypoint_names)
from moseq2_detectron_extract.io.session import Session
from moseq2_detectron_extract.io.util import attach_file_logger, ensure_dir, write_yaml
from moseq2_detectron_extract.pipeline import (ExtractFeaturesStep,
                                               FrameCropStep, InferenceStep,
                                               KeypointWriterStep, Pipeline,
                                               PreviewVideoWriterStep,
                                               ResultH5WriterStep, ProduceFramesStep)
from moseq2_detectron_extract.pipeline.progress import WorkerError
from moseq2_detectron_extract.proc.util import check_completion_status


def extract_session(session: Session, config: dict):
    ''' Primary entrypoint to extraction pipeline

    Parameters:
    session (Session): moseq session to extract
    config (dict): extraction configuration

    Returns:
    str: path to status dictionary file
    '''

    overall_time = time.time()

    # set up the output directory
    if config['output_dir'] is None:
        output_dir = os.path.join(session.dirname, 'proc')
        config['output_dir'] = output_dir
    else:
        output_dir = config['output_dir']
    ensure_dir(output_dir)

    # Attach log file to logging module
    attach_file_logger(os.path.join(output_dir, f"results_{config['bg_roi_index']:02d}.log"))

    status_filename = os.path.join(output_dir, f"results_{config['bg_roi_index']:02d}.yaml")
    if check_completion_status(status_filename):
        logging.warning('WARNING: Session appears to already be extracted, so skipping!')
        return status_filename

    status_dict = {
        'complete': False,
        'skip': False,
        'uuid': str(uuid.uuid4()),
        'metadata': session.load_metadata(),
        'parameters': deepcopy(config)
    }
    write_yaml(status_filename, status_dict)


    # Find image background and ROI
    first_frame, bground_im, roi, true_depth = session.find_roi(bg_roi_dilate=config['bg_roi_dilate'], bg_roi_shape=config['bg_roi_shape'],
        bg_roi_index=config['bg_roi_index'], bg_roi_weights=config['bg_roi_weights'], bg_roi_depth_range=config['bg_roi_depth_range'],
        bg_roi_gradient_filter=config['bg_roi_gradient_filter'], bg_roi_gradient_threshold=config['bg_roi_gradient_threshold'],
        bg_roi_gradient_kernel=config['bg_roi_gradient_kernel'], bg_roi_fill_holes=config['bg_roi_fill_holes'],
        use_plane_bground=config['use_plane_bground'], cache_dir=output_dir, verbose=True)
    logging.info("")
    config.update({
        'nframes': session.nframes,
        'true_depth': true_depth,
        'keypoint_names': default_keypoint_names,
        'keypoint_connection_rules': default_keypoint_connection_rules,
        'keypoint_colors': default_keypoint_colors,
        'first_frame': first_frame,
        'bground_im': bground_im,
        'roi': roi,
        'session': session,
        'status_dict': status_dict,
    })

    try:
        pipeline = Pipeline()
        step0  = pipeline.add_step(' Read Depth Data', ProduceFramesStep, config=config)
        step1  = pipeline.add_step(' Model Inference', InferenceStep, config=config)
        step2  = pipeline.add_step('Extract Features', ExtractFeaturesStep, show_progress=True, config=config)
        step3  = pipeline.add_step(' Crop and Rotate', FrameCropStep, config=config)
        step4a = pipeline.add_step('   Preview Video', PreviewVideoWriterStep, config=config)
        step4b = pipeline.add_step(' Write Keypoints', KeypointWriterStep, show_progress=True, config=config)
        step4c = pipeline.add_step(' Write H5 Result', ResultH5WriterStep, show_progress=True, config=config)
        pipeline.link(step0, step1)
        pipeline.link(step1, step2)
        pipeline.link(step2, step3)
        pipeline.link(step3, step4a, step4b, step4c)
        pipeline.add_timed_callback(30.0, log_processing_status)
        pipeline.start()
        # logging.info('all workers started')

        while pipeline.is_running():
            time.sleep(0.1)

        # shutdown the pipeline
        pipeline.shutdown()
    except WorkerError as work_error:
        logging.error('')
        logging.error('One or more workers encountered an error during extraction:\n')
        for err in work_error.error_info:
            logging.error(f'Worker "{err.name.strip()}" raised an exception:\n{err.message}')
            logging.error('')

    except Exception: # pylint: disable=broad-except
        logging.error('')
        logging.error('Error during extraction', exc_info=True)
        logging.error('')

    else:
        # mark status as complete and flush to filesystem
        status_dict['complete'] = True
        write_yaml(status_filename, status_dict)

        # show overall processing statistics
        extract_duration = (time.time() - overall_time)
        extract_fps = session.nframes / extract_duration
        logging.info(f'Finished processing {session.nframes} frames in {timedelta(seconds=round(extract_duration))} (approx. {extract_fps:.2f} fps overall)')


    return status_filename


def log_processing_status(pipeline: Pipeline):
    ''' Log status of pipeline in a way that if friendly to log files
    '''
    producer_progress = pipeline.progress.get_stats(pipeline.steps[0].name)
    complete_progress = pipeline.progress.get_stats(pipeline.steps[-1].name)

    sec_elapsed = producer_progress['elapsed']
    total_frames = producer_progress['total']
    produced_frames = (producer_progress['completed'] or 0)
    completed_frames = (complete_progress['completed'] or 0)
    in_progress_frames = produced_frames - completed_frames
    percent_str = f'{completed_frames/total_frames:.1%}'.rjust(6)
    nchar = len(str(total_frames))

    msg = f'Completed processing {str(completed_frames).rjust(nchar)} / {total_frames} frames ({percent_str}) ' \
          f'in {timedelta(seconds=round(sec_elapsed))}, another {str(in_progress_frames).rjust(nchar)} frames in progress'
    logging.info(msg, extra={'nostream': True})
