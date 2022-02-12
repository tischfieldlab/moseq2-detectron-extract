import logging
import os
from typing import List

import h5py
import numpy as np

from moseq2_detectron_extract.io.annot import default_keypoint_names
from moseq2_detectron_extract.proc.keypoints import (
    find_nan_keypoints, find_outliers_jumping, load_keypoint_data_from_h5)


def find_outliers_h5(result_h5: str, dest=None, keypoint_names: List[str]=None, jump_win:int=6, jump_thresh: float=10.0):
    if keypoint_names is None:
        keypoint_names = [kp for kp in default_keypoint_names if kp != 'TailTip']

    with h5py.File(result_h5, 'r') as h5:
        kpts = load_keypoint_data_from_h5(h5, keypoint_names)

        logging.info('Searching for frames with NAN keypoints...')
        nan_keypoints = find_nan_keypoints(kpts)
        logging.info(f'Found {len(nan_keypoints)} frames with NAN keypoints.\n')

        logging.info('Searching for frames with jumping algorithm...')
        ind, dist, outliers = find_outliers_jumping(kpts, window=jump_win, thresh=jump_thresh)
        logging.info(f'Found {len(ind)} frames via jumping algorithm.\n')

        final_indices = sorted(set(np.concatenate([nan_keypoints, ind])))

        nframes = h5['/frames'].shape[0]
        logging.info(f"Found {len(final_indices)} putative outlier frames out of {nframes} extracted frames ({len(final_indices)/nframes:.2%})")

        if dest is None:
            dest = os.path.splitext(result_h5)[0] + '.outlier_idxs.txt'
        with open(dest, 'w') as f:
            f.writelines(f'{idx}\n' for idx in final_indices)

        return final_indices
