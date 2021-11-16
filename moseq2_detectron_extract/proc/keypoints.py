from typing import Dict, List, Literal, Tuple, Union
import h5py
import numpy as np
from bottleneck import move_median
from moseq2_detectron_extract.io.annot import default_keypoint_names
from moseq2_detectron_extract.proc.util import convert_pxs_to_mm
from moseq2_detectron_extract.stats import is_outlier



def rotate_points(points: np.ndarray, center: Tuple[float, float]=(0, 0), angle: float=0) -> np.ndarray:
    ''' Rotate a set of `points` around `origin` by `degrees`

    Parameters:
    points (np.ndarray): Array of shape (nkeypoints, 2|3 [x, y, s?]). Values in index 2 of axis 1 are assumed to be weights.
    center (Tuple[float, float]): center point of the rotation transform
    angle (float): number of degrees to rotate the points

    Returns:
    np.ndarray containing rotated points of same shape as `points`
    '''
    if points.shape[1] == 3:
        weights = points[:, 2]
        points = points[:, :2]
    else:
        weights = None

    angle = np.deg2rad(-angle)
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    o = np.atleast_2d(center)
    p = np.atleast_2d(points)
    rotated = np.squeeze((R @ (p.T-o.T) + o.T).T)

    if weights is not None:
        rotated = np.append(rotated, weights[..., None], 1)
    return rotated


def rotate_points_batch(points: np.ndarray, centers: np.ndarray, angles: Union[np.ndarray, float]) -> np.ndarray:
    ''' Rotate points over several frames

    Parameters:
    points (np.ndarray): array of shape (nframes, nkeypoints, 2|3 [x, y, s?]) to be rotated
    centers (np.ndarray): array of rotation centers of shape (nframes, 2 [x, y])
    angles (Union[np.ndarray, float]): number of degrees to rotate the points. If a scalar, same angle is used for all frames. Otherwise should be np.ndarray of shape (nframes,)

    Returns:
    np.ndarray containing rotated points of same shape as `points`
    '''
    # expects (frames x keypoints x 3 [x,, y, s])
    if isinstance(angles, (int, float)):
        angles = [angles] * points.shape[0]

    for i in range(points.shape[0]):
        points[i, 0] = rotate_points(points[i, 0], centers[i], angles[i])
    return points


def keypoint_attributes(keypoint_names: List[str]=None) -> Dict[str, str]:
    ''' Gets keypoint attributes dict with names paired with descriptions.

    Parameters:
    keypoint_names (List[str]|None): List of keypoint names to use. If none, will use `moseq2_detectron_extract.io.annot.default_keypoint_names`

    Returns:
    attributes (Dict[str, str]): collection of metadata keys and descriptions.
    '''

    if keypoint_names is None:
        keypoint_names = default_keypoint_names

    attributes = {}
    for kpn in keypoint_names:
        for cs in ['reference', 'rotated']:
            attributes[f'{cs}/{kpn}_x_px'] = f'X position of {kpn} (pixels) in {cs} coordinate system.'
            attributes[f'{cs}/{kpn}_y_px'] = f'Y position of {kpn} (pixels) in {cs} coordinate system.'
            attributes[f'{cs}/{kpn}_x_mm'] = f'X position of {kpn} (mm) in {cs} coordinate system.'
            attributes[f'{cs}/{kpn}_y_mm'] = f'Y position of {kpn} (mm) in {cs} coordinate system.'
            attributes[f'{cs}/{kpn}_z_mm'] = f'Z position of {kpn} (mm) in {cs} coordinate system.'
            attributes[f'{cs}/{kpn}_score'] = f'Inference score of {kpn}.'

    return attributes


def keypoints_to_dict(keypoints: np.ndarray, frames: np.ndarray, centers: np.ndarray, angles: np.ndarray, true_depth: float=673.1,
    keypoint_names: List[str]=None) -> Dict[str, np.ndarray]:
    ''' Convert keypoint data to a dict format

    Also converts keypoints to following:
    a) keypoints in reference coordinate system and units of pixels, (0,0) indicates top left
    b) keypoints in reference coordinate system and units of millimeters, (0,0) indicates center of Kinect field of view
    c) keypoints in rotated coordinate system and units of pixels, (0,0) indicates centroid of animal
    d) keypoints in rotated coordinate system and units of millimeters, (0,0) indicates centroid of animal

    We also grab the putative Z-position (in millimeters) for each keypoint.

    Parameters:
    keypoints (np.ndarray): array of keypoint data of shape (nframes, nkeypoints, 3 [x, y, s])
    frames (np.ndarray): depth frames, used to compute z-position of keypoints
    centers (np.ndarray): centroids to use for rotating keypoints, of shape (nframes, 2 [x, y])
    angles (np.ndarray): angles to use for rotating keypoints, of shape (nframes,)
    true_depth (float): true depth to the floor, use for converting to mm units
    keypoint_names (List[str]|None): List of keypoint names to use. If none, will use `moseq2_detectron_extract.io.annot.default_keypoint_names`

    Returns:
    Dict[str, np.ndarray] containing keypoint data in various coordinate systems and units of measure
    '''

    if keypoint_names is None:
        keypoint_names = default_keypoint_names

    # Collect Z-height for each keypoint, and convert to mm
    x_coords = np.clip(np.floor(keypoints[:, 0, :, 0]).astype(int), 0, frames.shape[2]-1)
    y_coords = np.clip(np.floor(keypoints[:, 0, :, 1]).astype(int), 0, frames.shape[1]-1)
    z_data = np.zeros((frames.shape[0], keypoints.shape[2]))
    ref_kpts_px = np.copy(keypoints)
    ref_kpts_mm = np.zeros_like(keypoints)
    ref_kpts_mm[:, :, :, 2] = keypoints[:, :, :, 2] # copy over scores
    for kpi in range(keypoints.shape[2]):
        # fetch z height
        z_data[:, kpi] = frames[np.arange(frames.shape[0]), y_coords[:, kpi], x_coords[:, kpi]]

        # convert coordinates from px to mm
        ref_kpts_mm[:, 0, kpi, :2] = convert_pxs_to_mm(keypoints[:, 0, kpi, :2], true_depth=true_depth)

    # Rotated keypoints in px, relative to centroid
    rot_kpts_px = rotate_points_batch(np.copy(keypoints), centers=centers, angles=angles)
    rot_kpts_px[:, 0, :, :2] -= np.expand_dims(centers, axis=1)

    # Rotated keypoints in mm, relative to centroid
    centroid_mm = convert_pxs_to_mm(centers, true_depth=true_depth)
    rot_kpts_mm = rotate_points_batch(np.copy(ref_kpts_mm), centers=centroid_mm, angles=angles)
    rot_kpts_mm[:, 0, :, :2] -= np.expand_dims(centroid_mm, axis=1)

    # Record keypoint positions in both coordinate systems and units of measure
    out = {}
    for kpi, kpn in enumerate(default_keypoint_names):
        out[f'reference/{kpn}_x_px'] = ref_kpts_px[:, 0, kpi, 0]
        out[f'reference/{kpn}_y_px'] = ref_kpts_px[:, 0, kpi, 1]
        out[f'reference/{kpn}_score'] = ref_kpts_px[:, 0, kpi, 2]

        out[f'reference/{kpn}_x_mm'] = ref_kpts_mm[:, 0, kpi, 0]
        out[f'reference/{kpn}_y_mm'] = ref_kpts_mm[:, 0, kpi, 1]
        out[f'reference/{kpn}_z_mm'] = z_data[:, kpi]

        out[f'rotated/{kpn}_x_px'] = rot_kpts_px[:, 0, kpi, 0]
        out[f'rotated/{kpn}_y_px'] = rot_kpts_px[:, 0, kpi, 1]
        out[f'rotated/{kpn}_score'] = rot_kpts_px[:, 0, kpi, 2]

        out[f'rotated/{kpn}_x_mm'] = rot_kpts_mm[:, 0, kpi, 0]
        out[f'rotated/{kpn}_y_mm'] = rot_kpts_mm[:, 0, kpi, 1]
        out[f'rotated/{kpn}_z_mm'] = z_data[:, kpi]

    return out


def load_keypoint_data_from_h5(h5: h5py.File, keypoints: List[str]=None, coord_system: Literal['reference', 'rotated']='reference',
                               units: Literal['px','mm']='px'):
    ''' Load keypoint data from a result h5 files.

    Parameters:
    h5 (h5py.File): h5 file create as a result of the extraction function
    keypoints (Optional[List[str]]): keypoint names to load. If none, will use default keypoint names
    coord_system (Literal['reference', 'rotated']): Coordinate system to use
    units (Literal['px','mm']): units of measurement to use

    Returns:
    numpy.ndarray of shape (nframes, nkeypoints, 3 [x, y, s])
    '''
    if keypoints is None:
        keypoints = default_keypoint_names

    keys = [f'/keypoints/{coord_system}/{kp}' for kp in keypoints]
    data = np.ndarray((h5['frames'].shape[0], len(keys), 3), dtype=float)
    for kpi, kp in enumerate(keys):
        data[:, kpi, 0] = h5[f'{kp}_x_{units}'][()]
        data[:, kpi, 1] = h5[f'{kp}_y_{units}'][()]
        data[:, kpi, 2] = h5[f'{kp}_score'][()]
    return data


def find_outliers_jumping(data: np.ndarray, window: int=4, thresh: float=10):
    ''' Find outlier frames in data based on deviation from expected position

    Basic alogrithm:
    - compute sliding window median (size `win`) over data
    - compute distance of actual position to modelled position
    - detect outlier keypoints using median absolute deviation (MAD) method with threshold `thresh`
    - detect outlier frames as frames with any outlier keypoints

    Parameters:
    `data` (np.ndarray): a numpy.ndarray with shape like (nframes, nkeypoints, 3 [x, y, s])
    `window` (int): Sliding window size
    `thresh` (float): modified z-score to use as threshold for calling outliers

    Returns:
    A tuple containing (`indicies`, `distances`, `outliers`), where:
    `indicies` - 1D numpy array of frame indicies called as outliers
    `distances` - a (nframes, nkeypoints) sized dataframe containing keypoint distances to modelled position
    `outliers` - a (nframes, nkeypoints) sized boolean numpy array containing outliers calls for each keypoint and frame

    '''
    data = np.copy(data[:,:,:2]) # drop scores
    windows = move_median(data, window=window, min_count=1, axis=0)
    diff = (data - windows) ** 2
    dist = np.sqrt(np.sum(diff, axis=2))

    outliers = np.zeros(dist.shape[:2], dtype=bool)
    for i in range(dist.shape[1]):
        outliers[:,i] = is_outlier(dist[:,i], thresh=thresh)
    ind = np.where(outliers.any(axis=1))[0]

    return ind, dist, outliers
