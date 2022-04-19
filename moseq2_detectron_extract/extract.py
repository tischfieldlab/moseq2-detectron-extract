import logging
import os
import time
import uuid
from copy import deepcopy
from datetime import timedelta

from detectron2.data import MetadataCatalog

from moseq2_detectron_extract.io.annot import register_dataset_metadata
from moseq2_detectron_extract.io.session import Session, Stream
from moseq2_detectron_extract.io.util import (attach_file_logger, ensure_dir,
                                              write_yaml)
from moseq2_detectron_extract.pipeline import (InferenceStep,Pipeline,
                                               PreviewVideoWriterStep,
                                               ProduceFramesStep,
                                               ProcessFeaturesStep,
                                               ResultWriterStep, WorkerError)
from moseq2_detectron_extract.proc.util import check_completion_status


def extract_session(session: Session, config: dict):
    ''' Primary entrypoint to extraction pipeline

    Parameters:
    session (Session): moseq session to extract
    config (dict): extraction configuration

    Returns:
    str: path to status dictionary file
    '''

    # Record the time we started, we can compute overall time spent at the end
    start_time = time.time()

    # set up the output directory
    if config['output_dir'] is None:
        output_dir = os.path.join(session.dirname, 'proc')
        config['output_dir'] = output_dir
    else:
        output_dir = config['output_dir']
    ensure_dir(output_dir)

    # Attach log file to logging module, dependent on having setup output_dir
    attach_file_logger(os.path.join(output_dir, f"results_{config['bg_roi_index']:02d}.log"))

    # Make status filename, and check if session looks to be already extracted
    # we usally don't want to overwrite already processed data
    status_filename = os.path.join(output_dir, f"results_{config['bg_roi_index']:02d}.yaml")
    if check_completion_status(status_filename):
        logging.warning('WARNING: Session appears to already be extracted, so skipping!')
        return status_filename

    # Create status dictionary and write out to file system
    status_dict = {
        'complete': False,
        'skip': False,
        'uuid': str(uuid.uuid4()),
        'metadata': session.load_metadata(),
        'parameters': deepcopy(config)
    }
    write_yaml(status_filename, status_dict)


    # Find image background and ROI
    session.find_roi(bg_roi_dilate=config['bg_roi_dilate'],
                     bg_roi_shape=config['bg_roi_shape'],
                     bg_roi_index=config['bg_roi_index'],
                     bg_roi_weights=config['bg_roi_weights'],
                     bg_roi_depth_range=config['bg_roi_depth_range'],
                     bg_roi_gradient_filter=config['bg_roi_gradient_filter'],
                     bg_roi_gradient_threshold=config['bg_roi_gradient_threshold'],
                     bg_roi_gradient_kernel=config['bg_roi_gradient_kernel'],
                     bg_roi_fill_holes=config['bg_roi_fill_holes'],
                     use_plane_bground=config['use_plane_bground'],
                     cache_dir=output_dir,
                     verbose=True)
    logging.info("")
    config.update({
        'nframes': session.nframes,
        'true_depth': session.true_depth,
        'roi': session.roi,
        'first_frame': session.first_frame,
        'bground_im': session.bground_im,
        'status_dict': status_dict,
        'timestamps': session.load_timestamps(Stream.DEPTH),
    })

    if config['dataset_name'] not in MetadataCatalog:
        register_dataset_metadata(config['dataset_name'])

    try:
        # Create processing pipeline
        pipeline = Pipeline()
        step0  = pipeline.add_step(' Read Depth Data', ProduceFramesStep, session=session, config=config)
        step1  = pipeline.add_step(' Model Inference', InferenceStep, config=config)
        step2  = pipeline.add_step('Process Features', ProcessFeaturesStep, show_progress=True, config=config)
        step3a = pipeline.add_step('   Preview Video', PreviewVideoWriterStep, config=config)
        step3b = pipeline.add_step('    Write Reults', ResultWriterStep, show_progress=True, config=config)
        pipeline.link(step0, step1)
        pipeline.link(step1, step2)
        pipeline.link(step2, step3a, step3b)
        pipeline.add_timed_callback(30.0, log_processing_status)

        # Start processing
        pipeline.start()

        # poll while running pipeline
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
        extract_duration = (time.time() - start_time)
        extract_fps = session.nframes / extract_duration
        logging.info(f'Finished processing {session.nframes} frames in {timedelta(seconds=round(extract_duration))} (approx. {extract_fps:.2f} fps overall)')


    return status_filename


def log_processing_status(pipeline: Pipeline):
    ''' Log status of pipeline in a way that if friendly to log files
    '''
    try:
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
    except: # pylint: disable=bare-except
        pass
