import numpy as np
from moseq2_detectron_extract.io.annot import default_keypoint_names
from moseq2_detectron_extract.proc.util import convert_pxs_to_mm


def crop_points(points, center, crop_size=(80, 80)):
    points[:, 0, :, 0] -= np.expand_dims((crop_size[0] - center[:, 0] // 2), axis=1)
    points[:, 0, :, 1] -= np.expand_dims((crop_size[1] - center[:, 1] // 2), axis=1)

    return points


def rotate_points(points, center=(0, 0), angle=0):
    ''' Rotate a set of `points` around `origin` by `degrees`
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


def rotate_points_batch(points, centers, angles):
    # expects (frames x keypoints x 3 [x,, y, s])
    if isinstance(angles, (int, float)):
        angles = [angles] * points.shape[0]

    for i in range(points.shape[0]):
        points[i, 0] = rotate_points(points[i, 0], centers[i], angles[i])
    return points


def keypoint_attributes(keypoint_names=None):
    '''
    Gets keypoint attributes dict with names paired with descriptions.
    Returns
    -------
    attributes (dict): collection of metadata keys and descriptions.
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


def keypoints_to_dict(keypoints, frames, centers, angles, true_depth=673.1, keypoint_names=None):

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

    # Record keypoint positions in reference coordinate system
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
