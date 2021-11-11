import os
import time
import uuid
from copy import deepcopy
from threading import Thread
from typing import List, Union

import tqdm
from torch.multiprocessing import Process, Queue, SimpleQueue

from moseq2_detectron_extract.io.annot import (
    default_keypoint_connection_rules, default_keypoint_names)
from moseq2_detectron_extract.io.memory import SharedMemoryDict
from moseq2_detectron_extract.io.session import Session
from moseq2_detectron_extract.io.util import write_yaml
from moseq2_detectron_extract.pipeline import (ExtractFeaturesStep,
                                               FrameCropStep, InferenceStep,
                                               KeypointWriterStep,
                                               PreviewVideoWriterStep,
                                               ProcessProgress,
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

    status_filename = os.path.join(output_dir, 'results_{:02d}.yaml'.format(config['bg_roi_index']))
    if check_completion_status(status_filename):
        print('WARNING: Sessions appears to already be extracted, so skipping!')
        return

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
        use_plane_bground=config['use_plane_bground'], cache_dir=output_dir)
    print(f'Found true depth: {true_depth}')
    config.update({
        'nframes': session.nframes,
        'true_depth': true_depth,
        'keypoint_names': default_keypoint_names,
        'keypoint_connection_rules': default_keypoint_connection_rules,
    })


    progress = ProcessProgress()
    reader_pbar = progress.add(desc='Processing batches', total=session.nframes)

    inference_in: Queue = Queue(maxsize=1)
    inference_out: Queue = SimpleQueue()
    inference_pbar = progress.add(desc='Inferring')
    inference_thread = InferenceStep(config=config, progress=inference_pbar, in_queue=inference_in, out_queue=inference_out)

    extract_features_out: Queue = SimpleQueue()
    extract_features_pbar = None #progress.add(desc='Extracting Features')
    extract_features_thread = ExtractFeaturesStep(config=config, progress=extract_features_pbar, in_queue=inference_out, out_queue=extract_features_out)

    frame_crop_out: List[Queue] = [
        SimpleQueue(),
        SimpleQueue(),
        SimpleQueue(),
    ]
    frame_crop_pbar = progress.add(desc='Crop and Rotate')
    frame_crop_thread = FrameCropStep(config=config, progress=frame_crop_pbar, in_queue=extract_features_out, out_queue=frame_crop_out)

    preview_vid_pbar = progress.add(desc='Preview Video')
    preview_vid_thread = PreviewVideoWriterStep(config=config, progress=preview_vid_pbar, in_queue=frame_crop_out[0], out_queue=None)

    keypoint_writer_pbar = None #progress.add(desc='Writing Keypoints')
    keypoint_writer_thread = KeypointWriterStep(config=config, progress=keypoint_writer_pbar, in_queue=frame_crop_out[1], out_queue=None)

    result_writer_thread = ResultH5WriterStep(session, roi=roi, bground_im=bground_im, first_frame=first_frame,
        config=config, status_dict=status_dict, in_queue=frame_crop_out[2], out_queue=None)

    all_threads: List[Union[Thread, Process]] = [
        inference_thread,
        extract_features_thread,
        frame_crop_thread,
        preview_vid_thread,
        keypoint_writer_thread,
        result_writer_thread,
    ]
    # start the threads
    for t in all_threads:
        t.start()
    progress.start()



    for i, (frame_idxs, raw_frames) in enumerate(session.iterate(config['chunk_size'], config['chunk_overlap'])):
        offset = config['chunk_overlap'] if i > 0 else 0
        raw_frames = prep_raw_frames(raw_frames, bground_im=bground_im, roi=roi, vmin=config['min_height'], vmax=config['max_height'])
        shm = { 'batch': i, 'chunk': raw_frames, 'frame_idxs': frame_idxs, 'offset': offset }
        inference_in.put(SharedMemoryDict(shm))
        reader_pbar.put({'update': raw_frames.shape[0]})
    inference_in.put(None) # signal we are done

    # join the threads
    for t in all_threads:
        t.join()
    progress.shutdown()
    progress.join()

    # mark status as complete and flush to filesystem
    status_dict['complete'] = True
    write_yaml(status_filename, status_dict)

    tqdm.tqdm.write(f'Finished processing {session.nframes} frames in {time.time() - overall_time} seconds')
