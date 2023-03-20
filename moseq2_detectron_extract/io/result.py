from importlib.metadata import version
import os
from typing import Union

import h5py
from moseq2_detectron_extract.io.click import click_param_annot

from moseq2_detectron_extract.io.util import dict_to_h5
from moseq2_detectron_extract.proc.keypoints import keypoint_attributes
from moseq2_detectron_extract.proc.scalars import scalar_attributes



def create_extract_h5(h5_file: h5py.File, config_data: dict, status_dict: dict) -> None:
    ''' Prepare a h5 file for writing extraction results. Creates the necessary datasets and writes initial metadata:
     - Acquisition metadata
     - extraction metadata
     - computed scalars
     - computed keypoints
     - timestamps
     - original frames/frames_mask

    Parameters:
    h5_file (h5py.File): opened h5 file object to write to.
    config_data (dict): dictionary object containing all required extraction parameters. (auto generated)
    status_dict (dict): dictionary that helps indicate if the session has been extracted fully.
    kwargs (dict): additional keyword arguments.
    '''
    nframes = config_data['nframes']

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
    h5_file.create_dataset('timestamps', compression='gzip', data=config_data['timestamps'])
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
    h5_file.create_dataset('metadata/extraction/roi', data=config_data['roi'], compression='gzip')
    h5_file['metadata/extraction/roi'].attrs['description'] = 'ROI mask'

    # First Frame
    h5_file.create_dataset('metadata/extraction/first_frame', data=config_data['first_frame'], compression='gzip')
    h5_file['metadata/extraction/first_frame'].attrs['description'] = 'First frame of depth dataset'

    # Background
    h5_file.create_dataset('metadata/extraction/background', data=config_data['bground_im'], compression='gzip')
    h5_file['metadata/extraction/background'].attrs['description'] = 'Computed background image'

    # Extract Version
    package_name = 'moseq2-detectron-extract'
    extract_version = f'{package_name} v{version(package_name)}'
    h5_file.create_dataset('metadata/extraction/extract_version', data=extract_version)
    h5_file['metadata/extraction/extract_version'].attrs['description'] = 'Version of moseq2-extract'

    # Extraction Parameters
    from moseq2_detectron_extract.cli import extract # pylint: disable=import-outside-toplevel
    dict_to_h5(h5_file, status_dict['parameters'], 'metadata/extraction/parameters', click_param_annot(extract))

    # Acquisition Metadata
    for key, value in status_dict['metadata'].items():
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


def copy_frame(h5_file: h5py.File, src_frame: int, dst_frame: int) -> None:
    ''' Copy data from `scr_frame` and write to `dst_frame`

    Parameters:
    h5_file (h5py.File): results file to operate upon
    scr_frame (int): index of the frame to copy
    dst_frame (int): index of the destination frame
    '''

    # copy over extracted frames and masks
    h5_file['frames'][dst_frame] = h5_file['frames'][src_frame]
    h5_file['frames_mask'][dst_frame] = h5_file['frames_mask'][src_frame]

    # copy over scalars and keypoints
    bases = [
        '/scalars',
        '/keypoints/reference',
        '/keypoints/rotated'
    ]
    for base in bases:
        for key in h5_file[base].keys():
            h5_file[f'{base}/{key}'][dst_frame] = h5_file[f'{base}/{key}'][src_frame]

    # copy over flips
    flip_keys = [f'/metadata/extraction/{key}' for key in h5_file['/metadata/extraction'].keys() if key.startswith('flips')]
    for key in flip_keys:
        h5_file[key][dst_frame] = h5_file[key][src_frame]


def trim_results(h5_file: Union[str, h5py.File], start: int, stop: int) -> None:
    ''' Trim all datasets (excluding datasets under `/metadata`) to the range specified by `start` and `stop`

    Parameters:
    h5_file (str|h5py.File): results file to operate upon
    start (int): first index to keep
    stop (int): last index to keep (follows normal python indexing semantics)
    '''
    h5: h5py.File
    need_to_close = False
    if isinstance(h5_file, (str, os.PathLike)):
        h5 = h5py.File(h5_file, mode='r+')
        need_to_close = True
    elif isinstance(h5_file, h5py.File):
        h5 = h5_file
        if h5.mode != 'r+':
            raise ValueError('Expecting the h5 file to have been opened in writable mode, but it is read-only!')
    else:
        raise ValueError(f'Was expecting a string path to h5 file or a h5py.File object, but got {type(h5_file)}')


    def _do_trim(name: str, node: Union[h5py.Group, h5py.Dataset]):
        if isinstance(node, h5py.Dataset):
            # node is a dataset
            if 'metadata' not in name:
                data = node[start:stop]
                node.resize(stop-start, axis=0)
                node[:] = data[:]
        else:
            # node is a group
            pass


    h5.visititems(_do_trim)
    h5.flush()

    if need_to_close:
        h5.close()
