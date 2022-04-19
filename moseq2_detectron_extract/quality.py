import logging
import os
from typing import Iterable, List

import h5py
import numpy as np

from moseq2_detectron_extract.io.annot import default_keypoint_names
from moseq2_detectron_extract.io.video import collapse_consecutive_values
from moseq2_detectron_extract.proc.keypoints import (
    find_nan_keypoints, find_outliers_jumping, load_keypoint_data_from_h5)
from moseq2_detectron_extract.proc.proc import flips_from_keypoints


def find_outliers_h5(result_h5: str, dest: str=None, keypoint_names: List[str]=None, jump_win: int=6, jump_thresh: float=10.0) -> List[int]:
    ''' Find outliers from a h5 results file

    Parameters:
    result_h5 (str): path to a h5 file containing extraction results
    dest (str|None): base path to write results to. If None, use the same path as the result h5 file
    keypoint_names (List[str]): keypoints to inspect. If None, use the default keypoint names
    jump_win (int): window for jumping algorithm
    jump_thresh (float): threshold for jumping algorithm

    Returns:
    List[int]: combined and deduplicated indicies marked as outliers by any of the algorithms
    '''
    if keypoint_names is None:
        keypoint_names = [kp for kp in default_keypoint_names if kp != 'TailTip']

    if dest is None:
        dest = os.path.splitext(result_h5)[0] + '.outlier_idxs.{}.txt'

    with h5py.File(result_h5, 'r') as h5:
        kpts = load_keypoint_data_from_h5(h5, keypoint_names)

        logging.info('Searching for frames with NAN keypoints...')
        nan_keypoints = find_nan_keypoints(kpts)
        write_indicies(dest.format('NaNs'), nan_keypoints)
        logging.info(f' -> Found {len(nan_keypoints)} frames with NAN keypoints.\n')

        logging.info('Searching for frames with jumping algorithm...')
        ind, _, _ = find_outliers_jumping(kpts, window=jump_win, thresh=jump_thresh)
        write_indicies(dest.format('jumping'), nan_keypoints)
        logging.info(f' -> Found {len(ind)} frames via jumping algorithm.\n')

        logging.info('Searching for frames with flip disagreements...')
        centroids = np.column_stack((h5['/scalars/centroid_x_px'][()], h5['/scalars/centroid_y_px'][()]))
        angles = h5['/scalars/angle'][()]
        flips_mask = flips_from_keypoints(kpts, centroids, angles)
        flips = np.nonzero(flips_mask)[0]
        write_indicies(dest.format('flips'), flips)
        logging.info(f' -> Found {len(flips)} frames with flip disagreements.\n')

        # Combine output from all algorithms
        nframes = h5['/frames'].shape[0] # pylint: disable=no-member
        final_indices = sorted(set(np.concatenate([nan_keypoints, ind])))
        write_indicies(dest.format('combined'), final_indices)
        logging.info(f"Found {len(final_indices)} putative outlier frames out of {nframes} extracted frames ({len(final_indices)/nframes:.2%})")

        return final_indices


def write_indicies(filename: str, indicies: Iterable[int], collapse: bool = True):
    ''' Write indicies to a file

    Parameters:
    filename (str): path to write to
    indicies (Iterable[int]): indicies to write
    collapse (bool): If True, collapse indicies ranges before writing
    '''
    if collapse:
        ranges = collapse_consecutive_values(indicies)
        to_write = [f'{start} - {start + span}\n' for start, span in ranges]
    else:
        to_write = [f'{idx}\n' for idx in indicies]

    with open(filename, 'w', encoding='utf-8') as outlier_idx_file:
        outlier_idx_file.writelines(to_write)
