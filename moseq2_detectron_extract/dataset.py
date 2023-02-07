import json
import logging
import os
from copy import deepcopy
from functools import partial
from typing import Optional, Dict, Iterable, List, Literal, Sequence

import numpy as np
import tqdm
from skimage.transform import resize
from sklearn.cluster import MiniBatchKMeans

from moseq2_detectron_extract.io.image import write_image
from moseq2_detectron_extract.io.session import (Session,
                                                 SessionFramesIterator, Stream)
from moseq2_detectron_extract.io.util import ensure_dir
from moseq2_detectron_extract.proc.proc import (prep_raw_frames,
                                                scale_raw_frames)
from moseq2_detectron_extract.proc.roi import apply_roi



SampleMethod = Literal['random', 'uniform', 'kmeans', 'list']


def generate_dataset_for_sessions(input_files: Sequence[str], streams: Sequence[Stream], num_samples: int,  sample_method: SampleMethod, roi_params: dict,
                      output_dir: str, indices: Optional[Sequence[int]]=None, min_height: int=0, max_height: int=100, chunk_size: int=1000) -> List[dict]:
    ''' Generate a dataset for multiple sessions

    Parameters:
    input_files (Iterable[str]): files from which to generate a dataset from
    streams (Iterable[Stream]): Streams to produce
    num_samples (int): Total number of samples to draw across all `input_files`
    sample_method (SampleMethod): Method used for sampling
    roi_params (dict): dictionary with params needed by Session.find_roi()
    output_dir (str): path to a directory to store the dataset
    indicies (None|Iterable[int]): If not None, indicies to limit samples to
    min_height (int): minimum animal height
    max_height (int): maximum animal height
    chunk_size (int): number of frames to read per iteration

    Returns:
    List[dict] - list of dicts containing data about each sample generated
    '''

    num_samples_per_file = int(np.ceil(num_samples / len(input_files)))
    output_dir = ensure_dir(output_dir)

    if streams is None or len(streams) == 0:
        streams = [Stream.DEPTH]
    streams = list(set(streams))


    output_info = []
    for in_file in tqdm.tqdm(input_files, desc='Datasets'):
        session_data = generate_dataset_for_session(in_file=in_file,
                                                    streams=streams,
                                                    sample_method=sample_method,
                                                    n_samples=num_samples_per_file,
                                                    output_dir=output_dir,
                                                    roi_params=roi_params,
                                                    indices=indices,
                                                    min_height=min_height,
                                                    max_height=max_height,
                                                    chunk_size=chunk_size)
        output_info.extend(session_data)

    logging.info(f'Finished writing dataset to "{output_dir}"')
    return output_info



def generate_dataset_for_session(in_file: str, streams: Iterable[Stream], n_samples: int, sample_method: SampleMethod, output_dir: str,
                                 roi_params: dict, indices: Optional[Sequence[int]]=None, min_height=0, max_height=0, chunk_size=1000):
    ''' Generate a dataset for a single session

    Parameters:
    in_file (str): file from which to generate a dataset from
    streams (Iterable[Stream]): Streams to produce
    n_samples (int): Total number of samples to draw across all `input_files`
    sample_method (SampleMethod): Method used for sampling
    output_dir (str): path to a directory to store the dataset
    roi_params (dict): dictionary with params needed by Session.find_roi()
    indicies (None|Iterable[int]): If not None, indicies to limit samples to
    min_height (int): minimum animal height
    max_height (int): maximum animal height
    chunk_size (int): number of frames to read per iteration

    Returns:
    List[dict] - list of dicts containing data about each sample generated
    '''
    parameters = deepcopy(locals())
    images_dir = ensure_dir(os.path.join(output_dir, 'images'))
    info_dir = ensure_dir(os.path.join(output_dir, '.info'))
    session = Session(in_file)

    session_info_dir = ensure_dir(os.path.join(info_dir, session.session_id))

    # Find image background and ROI
    session.find_roi(**roi_params, cache_dir=session_info_dir)

    # Dump status information
    with open(os.path.join(session_info_dir, 'info.json'), 'w', encoding='utf-8') as status_file:
        json.dump({
            'parameters': parameters,
            'session_id': session.session_id,
            'metadata': session.load_metadata(),
            'true_depth': session.true_depth,
        }, status_file, indent='\t')


    def __attach_frames_iterator_filters(iterator):
        iterator.attach_filter([Stream.DEPTH], partial(prep_raw_frames, bground_im=session.bground_im, roi=session.roi, vmin=min_height, vmax=max_height))
        iterator.attach_filter([Stream.DEPTH], partial(scale_raw_frames, vmin=min_height, vmax=max_height))
        iterator.attach_filter([Stream.RGB], partial(apply_roi, roi=session.roi))


    if sample_method == 'kmeans':
        if indices is None:
            kmeans_iter = session.iterate(chunk_size=chunk_size)
        else:
            kmeans_iter = session.index(indices, chunk_size=chunk_size)
        __attach_frames_iterator_filters(kmeans_iter)
        indices = select_frames_kmeans(kmeans_iter, n_samples)
        sample_method = 'list'


    iterator = prepare_session_iterator(session, streams, sample_method, n_samples, indices, chunk_size)
    __attach_frames_iterator_filters(iterator)
    session_data = produce_frames(iterator, images_dir)

    return session_data


def prepare_session_iterator(session: Session, streams: Iterable[Stream], sample_method: SampleMethod, n_samples: int, indices: Optional[Sequence[int]]=None,
                             chunk_size: int=1000) -> SessionFramesIterator:
    ''' Prepare a session iterator

    Parameters:
    session (Session): Session from wich to construct an iterator for
    streams (Iterable[Stream]): Streams the iterator should produce
    sample_method (SampleMethod): Method for sampling frames
    n_samples (int): How many frames to produce
    indicies (Iterable[int]): specific indicies to sample from
    chunk_size (int): how many frames to produce in one iteration of the iterator

    Returns:
    SessionFramesIterator - iterator producing frames
    '''

    # NOTE: we handle k-means outside of this function!!

    if sample_method == 'random':
        if indices is not None:
            seq = list(np.random.choice(indices, n_samples, replace=False))
            iterator = session.index(seq, chunk_size=chunk_size, streams=streams)
        else:
            iterator = session.sample(n_samples, chunk_size=chunk_size, streams=streams)

    elif sample_method == 'uniform':
        step = session.nframes // n_samples
        iterator = session.index(np.arange(step, session.nframes, step), chunk_size=chunk_size, streams=streams)

    elif sample_method == 'list':
        assert indices is not None
        iterator = session.index(indices, chunk_size=chunk_size, streams=streams)

    else:
        raise ValueError(f'Unknown sample_method "{sample_method}"')

    return iterator



def produce_frames(iterator: SessionFramesIterator, dest_directory: str) -> List[dict]:
    ''' Produce frames from `iterator`, save them to `dest_directory`, and return information about the frames

    Parameters:
    iterator (SessionFramesIterator): iterator which should yield frames
    dest_directory (str): directory where frames should be saved as images

    Returns:
    List[dict] - data about the frames which were produced
    '''
    session_data: Dict = {}
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
                    'session_id': iterator.session.session_id,
                    'true_depth': iterator.session.true_depth,
                    **iterator.session.load_metadata()
                }
            }

        if 'depth' in iterator.streams:
            for idx, raw_frame in zip(frame_idxs, data[iterator.streams.index('depth')+1]):
                dest = os.path.join(dest_directory, f'{iterator.session.session_id}_depth_{idx}.png')
                write_image(dest, raw_frame, scale=False, dtype='uint8')
                session_data[idx]['data']['depth_image'] = dest
                session_data[idx]['data']['images'].append(dest)

        if 'rgb' in iterator.streams:
            for idx, raw_frame in zip(frame_idxs, data[iterator.streams.index('rgb')+1]):
                dest = os.path.join(dest_directory, f'{iterator.session.session_id}_rgb_{idx}.png')
                write_image(dest, raw_frame, scale=False, dtype='uint8')
                session_data[idx]['data']['rgb_image'] = dest
                session_data[idx]['data']['images'].append(dest)

    return list(session_data.values())



def write_label_studio_tasks(tasks: List[dict], dest: str):
    ''' Write `tasks` to a json file
    '''
    if os.path.exists(dest):
        logging.warning(f'label-studio tasks file "{dest}" seems to already exist! Will append the new tasks to this existing file')
        with open(dest, 'r', encoding='utf-8') as task_file:
            existing_tasks = json.load(task_file)
            logging.warning(f' -> Read {len(existing_tasks)} existing tasks')
            tasks = existing_tasks + tasks

    with open(dest, 'w', encoding='utf-8') as task_file:
        json.dump(tasks, task_file, indent='\t')
    logging.info(f'Wrote {len(tasks)} label-studio tasks to "{dest}"')


def select_frames_kmeans(frames_iterator: SessionFramesIterator, num_frames_to_pick: int, num_clusters: Optional[int]=None,
                         scale: float=4, kmeans_batchsize: int=100, kmeans_max_iter: int=50) -> List[int]:
    ''' Select frames from a session using k-means clustering to pick dissimilar frames

    Parameters:
    frames_iterator (SessionFramesIterator): Iterator which yields depth frames
    num_frames_to_pick (int): Total number of frames to pick from this session
    num_clusters (int): Number of (k-means) clusters to use. If none, then `num_frames_to_pick` clusters are used
    scale (float): Down-sample images by this scale factor
    kmeans_batchsize (int): Batch size for fitting k-means model
    kmeans_max_iter (int): Max iterations for fitting k-means model

    Returns:
    list of frame indices which were selected
    '''

    roi = frames_iterator.session.roi
    downsampled = np.zeros((frames_iterator.nframes, int(roi.shape[0] / scale), int(roi.shape[1] / scale)))
    all_frame_idxs = np.empty((frames_iterator.nframes,), dtype=int)
    for b, (frame_idxs, raw_frames) in enumerate(tqdm.tqdm(frames_iterator, desc='Processing batches', leave=False)):
        for i, idx in enumerate(tqdm.tqdm(frame_idxs, desc='Resizing Frames', leave=False, disable=False)):
            sparse_idx = ((b * frames_iterator.chunk_size) + i)
            downsampled[sparse_idx, :, :] = resize(raw_frames[i], downsampled.shape[1:], anti_aliasing=True, preserve_range=True, mode='constant')
            all_frame_idxs[sparse_idx] = idx

    with tqdm.tqdm(total=1, leave=False, desc="Kmeans clustering ... (this might take a while)") as pbar:
        data = downsampled - downsampled.mean(axis=0)
        data = data.reshape(data.shape[0], -1)  # stacking

        if num_clusters is None:
            num_clusters = num_frames_to_pick

        kmeans = MiniBatchKMeans(
            n_clusters=num_clusters,
            tol=1e-3,
            batch_size=kmeans_batchsize,
            max_iter=kmeans_max_iter
        )
        kmeans.fit(data)
        pbar.update(1)

    num_frames_per_cluster = max(num_frames_to_pick // num_clusters, 1)

    selected_frames = []
    for cluster_id in range(num_clusters):  # pick one frame per cluster
        cluster_ids = np.where(cluster_id == kmeans.labels_)[0]
        cluster_ids = all_frame_idxs[cluster_ids]

        num_images_in_cluster = len(cluster_ids)
        if num_images_in_cluster > 0:
            selected_frames.extend(list(cluster_ids[np.random.choice(num_images_in_cluster, size=num_frames_per_cluster, replace=False)]))

    return selected_frames
