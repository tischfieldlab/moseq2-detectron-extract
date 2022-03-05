import os
from typing import Tuple

import cv2
import numpy as np
from moseq2_detectron_extract.io.util import read_yaml


def select_strel(shape: str='e', size: Tuple[int, int]=(10, 10)):
    ''' Create a CV2 structuring element

    Parameters:
    shape (string): shape of the structuring element
    size (Tuple[int, int]): size of the structuring element

    Returns:
    cv2 structuring element
    '''
    if shape[0].lower() == 'e':
        strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, size)
    elif shape[0].lower() == 'r':
        strel = cv2.getStructuringElement(cv2.MORPH_RECT, size)
    else:
        strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, size)

    return strel


def convert_pxs_to_mm(coords: np.ndarray, resolution: Tuple[int, int]=(512, 424), field_of_view: Tuple[float, float]=(70.6, 60),
                      true_depth: float=673.1) -> np.ndarray:
    '''
    Converts x, y coordinates in pixel space to mm.
    http://stackoverflow.com/questions/17832238/kinect-intrinsic-parameters-from-field-of-view/18199938#18199938
    http://www.imaginativeuniversal.com/blog/post/2014/03/05/quick-reference-kinect-1-vs-kinect-2.aspx
    http://smeenk.com/kinect-field-of-view-comparison/

    Parameters:
    coords (np.ndarray): list of x,y pixel coordinates
    resolution (Tuple[int, int]): image dimensions
    field_of_view (Tuple[float, float]): width and height scaling params
    true_depth (float): detected true depth

    Returns:
    new_coords (list): x,y coordinates in mm
    '''

    cx = resolution[0] // 2
    cy = resolution[1] // 2

    xhat = coords[:, 0] - cx
    yhat = coords[:, 1] - cy

    fw = resolution[0] / (2 * np.deg2rad(field_of_view[0] / 2))
    fh = resolution[1] / (2 * np.deg2rad(field_of_view[1] / 2))

    new_coords = np.zeros_like(coords)
    new_coords[:, 0] = true_depth * xhat / fw
    new_coords[:, 1] = true_depth * yhat / fh

    return new_coords


def check_completion_status(status_filename: str) -> bool:
    '''
    Reads a results_00.yaml (status file) and checks whether the session has been
    fully extracted. Returns True if yes, and False if not and if the file doesn't exist.

    Parameters:
    status_filename (str): path to results_00.yaml containing extraction status

    Returns:
    complete (bool): If True, data has been extracted to completion.
    '''

    if os.path.exists(status_filename):
        return read_yaml(status_filename)['complete']
    return False


def slice_dict(data: dict, index: int) -> dict:
    ''' Extract a slice out of a dict of numpy arrays

    Parameters:
    data (dict): dict with
    index (int): index on axis 0 to extract

    Returns:
    dict with the same keys as `data` and values as the `i`-th index of values from `data`
    '''
    out = {}
    for k, v in data.items():
        out[k] = v[index]
    return out
