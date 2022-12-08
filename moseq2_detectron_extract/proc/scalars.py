from typing import Dict
import numpy as np
from moseq2_detectron_extract.proc.util import convert_pxs_to_mm


def scalar_attributes() -> Dict[str, str]:
    ''' Gets scalar attributes dict with names paired with descriptions.

    Returns:
    attributes (dict): collection of metadata keys and descriptions.
    '''

    attributes = {
        'centroid_x_px': 'X centroid (pixels)',
        'centroid_y_px': 'Y centroid (pixels)',
        'velocity_2d_px': '2D velocity (pixels / frame), note that missing frames are not accounted for',
        'velocity_3d_px': '3D velocity (pixels / frame), note that missing frames are not accounted for, also height is in mm, not pixels for calculation',
        'width_px': 'Mouse width (pixels)',
        'length_px': 'Mouse length (pixels)',
        'area_px': 'Mouse area (pixels)',
        'centroid_x_mm': 'X centroid (mm)',
        'centroid_y_mm': 'Y centroid (mm)',
        'velocity_2d_mm': '2D velocity (mm / frame), note that missing frames are not accounted for',
        'velocity_3d_mm': '3D velocity (mm / frame), note that missing frames are not accounted for',
        'width_mm': 'Mouse width (mm)',
        'length_mm': 'Mouse length (mm)',
        'area_mm': 'Mouse area (mm)',
        'height_ave_mm': 'Mouse average height (mm)',
        'angle': 'Angle (radians, unwrapped)',
        'velocity_theta': 'Angular component of velocity (arctan(vel_x, vel_y))'
    }

    return attributes


def compute_scalars(frames: np.ndarray, track_features: dict, min_height: float=10, max_height: float=100, true_depth: float=673.1) -> Dict[str, np.ndarray]:
    ''' Computes scalars.

    Parameters:
    frames (3d np.ndarray): (frames x r x c), uncropped mouse
    track_features (dict):  dictionary with tracking variables (centroid and orientation)
    min_height (float): minimum height of the mouse
    max_height (float): maximum height of the mouse
    true_depth (float): detected true depth

    Returns:
    features (dict): dictionary of scalars
    '''

    nframes = frames.shape[0]

    # Pack features into dict
    features = {
        'centroid_x_px': np.zeros((nframes,), 'float32'),
        'centroid_y_px': np.zeros((nframes,), 'float32'),
        'velocity_2d_px': np.zeros((nframes,), 'float32'),
        'velocity_3d_px': np.zeros((nframes,), 'float32'),
        'width_px': np.zeros((nframes,), 'float32'),
        'length_px': np.zeros((nframes,), 'float32'),
        'area_px': np.zeros((nframes,)),
        'centroid_x_mm': np.zeros((nframes,), 'float32'),
        'centroid_y_mm': np.zeros((nframes,), 'float32'),
        'velocity_2d_mm': np.zeros((nframes,), 'float32'),
        'velocity_3d_mm': np.zeros((nframes,), 'float32'),
        'width_mm': np.zeros((nframes,), 'float32'),
        'length_mm': np.zeros((nframes,), 'float32'),
        'area_mm': np.zeros((nframes,)),
        'height_ave_mm': np.zeros((nframes,), 'float32'),
        'angle': np.zeros((nframes,), 'float32'),
        'velocity_theta': np.zeros((nframes,)),
    }

    # Get mm centroid
    centroid_mm = convert_pxs_to_mm(track_features['centroid'], true_depth=true_depth)
    centroid_mm_shift = convert_pxs_to_mm(track_features['centroid'] + 1, true_depth=true_depth)

    # Based on the centroid of the mouse, get the mm_to_px conversion
    px_to_mm = np.abs(centroid_mm_shift - centroid_mm)
    masked_frames = np.logical_and(frames > min_height, frames < max_height)

    features['centroid_x_px'] = track_features['centroid'][:, 0]
    features['centroid_y_px'] = track_features['centroid'][:, 1]

    features['centroid_x_mm'] = centroid_mm[:, 0]
    features['centroid_y_mm'] = centroid_mm[:, 1]

    # based on the centroid of the mouse, get the mm_to_px conversion

    features['width_px'] = np.min(track_features['axis_length'], axis=1)
    features['length_px'] = np.max(track_features['axis_length'], axis=1)
    features['area_px'] = np.sum(masked_frames, axis=(1, 2))

    features['width_mm'] = features['width_px'] * px_to_mm[:, 1]
    features['length_mm'] = features['length_px'] * px_to_mm[:, 0]
    features['area_mm'] = features['area_px'] * px_to_mm.mean(axis=1)

    features['angle'] = np.deg2rad(track_features['orientation'])

    nmask = np.sum(masked_frames, axis=(1, 2))

    for i in range(nframes):
        if nmask[i] > 0:
            features['height_ave_mm'][i] = np.mean(frames[i, masked_frames[i]])

    vel_x = np.diff(np.concatenate((features['centroid_x_px'][:1], features['centroid_x_px'])))
    vel_y = np.diff(np.concatenate((features['centroid_y_px'][:1], features['centroid_y_px'])))
    vel_z = np.diff(np.concatenate((features['height_ave_mm'][:1], features['height_ave_mm'])))

    features['velocity_2d_px'] = np.hypot(vel_x, vel_y)
    features['velocity_3d_px'] = np.sqrt(np.square(vel_x)+np.square(vel_y)+np.square(vel_z))

    vel_x = np.diff(np.concatenate((features['centroid_x_mm'][:1], features['centroid_x_mm'])))
    vel_y = np.diff(np.concatenate((features['centroid_y_mm'][:1], features['centroid_y_mm'])))

    features['velocity_2d_mm'] = np.hypot(vel_x, vel_y)
    features['velocity_3d_mm'] = np.sqrt(np.square(vel_x)+np.square(vel_y)+np.square(vel_z))

    features['velocity_theta'] = np.arctan2(vel_y, vel_x)

    return features
