import logging
import os
from typing import List

import h5py
import numpy as np

from moseq2_detectron_extract.io.annot import default_keypoint_names
from moseq2_detectron_extract.proc.keypoints import (
    find_nan_keypoints, find_outliers_jumping, load_keypoint_data_from_h5)
from moseq2_detectron_extract.proc.proc import flips_from_keypoints


def find_outliers_h5(result_h5: str, dest=None, keypoint_names: List[str]=None, jump_win:int=6, jump_thresh: float=10.0):
    ''' Find outliers from a h5 results file
    '''
    if keypoint_names is None:
        keypoint_names = [kp for kp in default_keypoint_names if kp != 'TailTip']

    with h5py.File(result_h5, 'r') as h5:
        kpts = load_keypoint_data_from_h5(h5, keypoint_names)

        logging.info('Searching for frames with NAN keypoints...')
        nan_keypoints = find_nan_keypoints(kpts)
        logging.info(f' -> Found {len(nan_keypoints)} frames with NAN keypoints.\n')

        logging.info('Searching for frames with jumping algorithm...')
        ind, _, _ = find_outliers_jumping(kpts, window=jump_win, thresh=jump_thresh)
        logging.info(f' -> Found {len(ind)} frames via jumping algorithm.\n')

        logging.info('Searching for frames with flip disagreements...')
        centroids = np.column_stack((h5['/scalars/centroid_x_px'][()], h5['/scalars/centroid_y_px'][()]))
        angles = h5['/scalars/angle'][()]
        flips_mask = flips_from_keypoints(kpts, centroids, angles)
        flips = np.nonzero(flips_mask)[0]
        print(flips)
        logging.info(f' -> Found {len(flips)} frames with flip disagreements.\n')

        final_indices = sorted(set(np.concatenate([nan_keypoints, ind])))

        nframes = h5['/frames'].shape[0] # pylint: disable=no-member
        logging.info(f"Found {len(final_indices)} putative outlier frames out of {nframes} extracted frames ({len(final_indices)/nframes:.2%})")

        if dest is None:
            dest = os.path.splitext(result_h5)[0] + '.outlier_idxs.txt'
        with open(dest, 'w', encoding='utf-8') as outlier_idx_file:
            outlier_idx_file.writelines(f'{idx}\n' for idx in final_indices)

        return final_indices
