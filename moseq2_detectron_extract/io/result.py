import h5py
import numpy as np
from pkg_resources import get_distribution

from moseq2_detectron_extract.io.util import click_param_annot, dict_to_h5
from moseq2_detectron_extract.proc.keypoints import keypoint_attributes
from moseq2_detectron_extract.proc.scalars import scalar_attributes



def create_extract_h5(h5_file: h5py.File, acquisition_metadata: dict, config_data: dict, status_dict: dict,
                      nframes: int, roi: np.ndarray, bground_im: np.ndarray, first_frame: np.ndarray, timestamps: np.ndarray) -> None:
    ''' Prepare a h5 file for writing extraction results. Creates the necessary datasets and writes initial metadata:
     - Acquisition metadata
     - extraction metadata
     - computed scalars
     - computed keypoints
     - timestamps
     - original frames/frames_mask.

    Parameters:
    h5_file (h5py.File): opened h5 file object to write to.
    acquisition_metadata (dict): Dictionary containing extracted session acquisition metadata.
    config_data (dict): dictionary object containing all required extraction parameters. (auto generated)
    status_dict (dict): dictionary that helps indicate if the session has been extracted fully.
    nframes (int): number of frames being recorded
    roi (2d np.ndarray): Computed 2D ROI Image.
    bground_im (2d np.ndarray): Computed 2D Background Image.
    first_frame (2d np.ndarray): Computed 2D First Frame Image.
    timestamps (np.array): Array of session timestamps.
    kwargs (dict): additional keyword arguments.
    '''

    h5_file.create_dataset('metadata/uuid', data=status_dict['uuid'])

    # Creating scalar dataset
    scalars_attrs = scalar_attributes()
    for scalar in list(scalars_attrs.keys()):
        h5_file.create_dataset(f'scalars/{scalar}', (nframes,), 'float32', compression='gzip')
        h5_file[f'scalars/{scalar}'].attrs['description'] = scalars_attrs[scalar]

    # Creating keypoints dataset
    keypoint_attrs = keypoint_attributes()
    for kp in list(keypoint_attrs.keys()):
        h5_file.create_dataset(f'keypoints/{kp}', (nframes,), 'float32', compression='gzip')
        h5_file[f'keypoints/{kp}'].attrs['description'] = keypoint_attrs[kp]

    # Timestamps
    if timestamps is not None:
        h5_file.create_dataset('timestamps', compression='gzip', data=timestamps)
        h5_file['timestamps'].attrs['description'] = "Depth video timestamps"

    # Cropped Frames
    h5_file.create_dataset('frames', (nframes, config_data['crop_size'][0], config_data['crop_size'][1]),
                     config_data['frame_dtype'], compression='gzip')
    h5_file['frames'].attrs['description'] = '3D Numpy array of depth frames (nframes x w x h, in mm)'

    # Frame Masks for EM Tracking
    if config_data['use_tracking_model']:
        h5_file.create_dataset('frames_mask', (nframes, config_data['crop_size'][0], config_data['crop_size'][1]), 'float32',
                         compression='gzip')
        h5_file['frames_mask'].attrs['description'] = 'Log-likelihood values from the tracking model (nframes x w x h)'
    else:
        h5_file.create_dataset('frames_mask', (nframes, config_data['crop_size'][0], config_data['crop_size'][1]), 'bool',
                         compression='gzip')
        h5_file['frames_mask'].attrs['description'] = 'Boolean mask, false=not mouse, true=mouse'

    # Flip Classifier
    if config_data['flip_classifier'] is not None:
        h5_file.create_dataset('metadata/extraction/flips', (nframes,), 'bool', compression='gzip')
        h5_file['metadata/extraction/flips'].attrs['description'] = 'Output from flip classifier, false=no flip, true=flip'

    # True Depth
    h5_file.create_dataset('metadata/extraction/true_depth', data=config_data['true_depth'])
    h5_file['metadata/extraction/true_depth'].attrs['description'] = 'Detected true depth of arena floor in mm'

    # ROI
    h5_file.create_dataset('metadata/extraction/roi', data=roi, compression='gzip')
    h5_file['metadata/extraction/roi'].attrs['description'] = 'ROI mask'

    # First Frame
    h5_file.create_dataset('metadata/extraction/first_frame', data=first_frame[0], compression='gzip')
    h5_file['metadata/extraction/first_frame'].attrs['description'] = 'First frame of depth dataset'

    # Background
    h5_file.create_dataset('metadata/extraction/background', data=bground_im, compression='gzip')
    h5_file['metadata/extraction/background'].attrs['description'] = 'Computed background image'

    # Extract Version
    package_name = 'moseq2-detectron-extract'
    extract_version = f'{package_name} v{np.string_(get_distribution(package_name).version)}'
    h5_file.create_dataset('metadata/extraction/extract_version', data=extract_version)
    h5_file['metadata/extraction/extract_version'].attrs['description'] = 'Version of moseq2-extract'

    # Extraction Parameters
    from moseq2_detectron_extract.cli import extract # pylint: disable=import-outside-toplevel
    dict_to_h5(h5_file, status_dict['parameters'], 'metadata/extraction/parameters', click_param_annot(extract))

    # Acquisition Metadata
    for key, value in acquisition_metadata.items():
        if isinstance(value, list) and len(value) > 0 and isinstance(value[0], str):
            value = [n.encode('utf8') for n in value]

        if value is not None:
            h5_file.create_dataset(f'metadata/acquisition/{key}', data=value)
        else:
            h5_file.create_dataset(f'metadata/acquisition/{key}', dtype="f")


def write_extracted_chunk_to_h5(h5_file: h5py.File, results: dict) -> None:
    ''' Write extracted frames, frame masks, scalars, and keypoints to an open h5 file.

    Parameters:
    h5_file (H5py.File): open results_00 h5 file to save data in.
    results (dict): extraction results dict.
    '''
    frame_range = results['frame_idxs']
    offset = results['offset']

    # Writing computed scalars to h5 file
    for scalar in results['scalars'].keys():
        h5_file[f'scalars/{scalar}'][frame_range] = results['scalars'][scalar][offset:]

    # Writing frames and mask to h5
    h5_file['frames'][frame_range] = results['depth_frames'][offset:]
    h5_file['frames_mask'][frame_range] = results['mask_frames'][offset:]

    # Writing flip classifier results to h5
    if 'flips' in results and results['flips'] is not None:
        h5_file['metadata/extraction/flips'][frame_range] = results['features']['flips'][offset:]

    # Writing keypoints dataset
    for kp in results['keypoints'].keys():
        h5_file[f'keypoints/{kp}'][frame_range] = results['keypoints'][kp][offset:]
