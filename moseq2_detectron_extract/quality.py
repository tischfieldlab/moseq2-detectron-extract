


import os
from typing import List

import h5py

from moseq2_detectron_extract.io.annot import default_keypoint_names
from moseq2_detectron_extract.proc.keypoints import (
    find_outliers_jumping, load_keypoint_data_from_h5)


def find_outliers_h5(result_h5: str, dest=None, keypoint_names: List[str]=None):
    if keypoint_names is None:
        keypoint_names = [kp for kp in default_keypoint_names if kp != 'TailTip']

    with h5py.File(result_h5, 'r') as h5:
        kpts = load_keypoint_data_from_h5(h5, keypoint_names)
        ind, dist, outliers = find_outliers_jumping(kpts, window=6, thresh=10)

        if dest is None:
            dest = os.path.splitext(result_h5)[0] + '.outlier_idxs.txt'
        with open(dest, 'w') as f:
            f.writelines(f'{idx}\n' for idx in ind)

        return ind
