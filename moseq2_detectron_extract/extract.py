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
from moseq2_detectron_extract.io.util import setup_logging, write_yaml
from moseq2_detectron_extract.pipeline import (ExtractFeaturesStep,
                                               FrameCropStep, InferenceStep,
                                               KeypointWriterStep, Pipeline,
                                               PreviewVideoWriterStep,
                                               ResultH5WriterStep)
from moseq2_detectron_extract.proc.proc import prep_raw_frames
from moseq2_detectron_extract.proc.util import check_completion_status


def extract_session(session: Session, config: dict):

    overall_time = time.time()

    # set up the output directory
    if config['output_dir'] is None:
        output_dir = os.path.join(session.dirname, 'proc')
        config['output_dir'] = output_dir
    else:
        output_dir = config['output_dir']

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    setup_logging(os.path.join(output_dir, 'results_{:02d}.log'.format(config['bg_roi_index'])))

    status_filename = os.path.join(output_dir, 'results_{:02d}.yaml'.format(config['bg_roi_index']))
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
    })

    try:
        pipeline = Pipeline()
        reader_pbar = pipeline.progress.add(name='producer', desc='Processing batches', total=session.nframes)
        out = pipeline.add_step('Inferring',InferenceStep, pipeline.input, config=config)
        out = pipeline.add_step('Extract Features', ExtractFeaturesStep, out[0], show_progress=False, config=config)
        out = pipeline.add_step('Crop and Rotate', FrameCropStep, out[0], num_listners=3, config=config)
        pipeline.add_step('Preview Video', PreviewVideoWriterStep, out[0], config=config)
        pipeline.add_step('Write Keypoints', KeypointWriterStep, out[1], show_progress=False, config=config)
        pipeline.add_step('Write H5 Result', ResultH5WriterStep, out[2], show_progress=False, session=session, config=config, status_dict=status_dict)
        pipeline.start()

        for i, (frame_idxs, raw_frames) in enumerate(session.iterate(config['chunk_size'], config['chunk_overlap'])):
            offset = config['chunk_overlap'] if i > 0 else 0
            raw_frames = prep_raw_frames(raw_frames, bground_im=bground_im, roi=roi, vmin=config['min_height'], vmax=config['max_height'])
            shm = { 'batch': i, 'chunk': raw_frames, 'frame_idxs': frame_idxs, 'offset': offset }
            while not pipeline.input.empty():
                time.sleep(0.1)
            pipeline.input.put(shm)
            reader_pbar.put({'update': raw_frames.shape[0]})
            cp = pipeline.progress.get_tqdm('producer')
            if cp is not None:
                n_frames = (cp.format_dict["n"] or 0)
                t_frames = cp.format_dict["total"]
                s_elapsed = cp.format_dict["elapsed"]
                logging.info(f'Processed {n_frames} / {t_frames} frames ({n_frames/t_frames:.2%}) in {timedelta(seconds=s_elapsed)}', extra={'nostream': True})
        pipeline.input.put(None) # signal we are done

        # join the threads
        pipeline.shutdown()
    except:
        logging.error('Error during extraction', exc_info=True)

    # mark status as complete and flush to filesystem
    status_dict['complete'] = True
    write_yaml(status_filename, status_dict)

    extract_duration = (time.time() - overall_time)
    extract_fps = session.nframes / extract_duration
    logging.info(f'Finished processing {session.nframes} frames in {timedelta(seconds=extract_duration)} (approx. {extract_fps:.2f} fps overall)')

    return status_filename
