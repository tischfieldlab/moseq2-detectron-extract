import os
import sys
from typing import Union
import click
import numpy as np
import json
import errno
import h5py

def gen_batch_sequence(nframes: int, chunk_size: int, overlap: int, offset: int=0):
    """Generate a sequence with overlap
    """
    seq = range(offset, nframes)
    for i in range(offset, len(seq)-overlap, chunk_size-overlap):
        yield seq[i:i+chunk_size]


def load_timestamps(timestamp_file: str, col: int=0) -> Union[np.array, None]:
    """Read timestamps from space delimited text file
    """

    ts = []
    try:
        with open(timestamp_file, 'r') as f:
            for line in f:
                cols = line.split()
                ts.append(float(cols[col]))
        ts = np.array(ts)
    except TypeError as e:
        # try iterating directly
        for line in timestamp_file:
            cols = line.split()
            ts.append(float(cols[col]))
        ts = np.array(ts)
    except FileNotFoundError as e:
        ts = None

    return ts


def load_metadata(metadata_file: str) -> dict:
    metadata = {}
    try:
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
    except TypeError as e:
        # try loading directly
        metadata = json.load(metadata_file)

    return metadata


def get_last_checkpoint(path: str) -> str:
    ''' Get the path to the last model checkpoint in a model directory
        Looks at the "last_checkpoint" file in the directory

        Parameters:
            path (str): directory containing the modelling results

        Returns:
            Path to the last checkpoint
    '''
    with open(os.path.join(path, 'last_checkpoint'), 'r') as f:
        last_checkpoint = f.read()
    return os.path.join(path, last_checkpoint)


def keypoints_to_dict(keypoint_names, kp_data, prefix=''):
    out = {}
    for ki, kp in enumerate(keypoint_names):
        out.update({
            k: v for k, v in zip([f"{prefix}{kp}_X", f"{prefix}{kp}_Y", f"{prefix}{kp}_S"], kp_data[ki])
        })
    return out


def ensure_dir(path: str) -> str:
    """ Ensures the path exists by creating the directories specified 
    by path if they do not already exist.
    
    Parameters:
    path (string): path for which to ensure directories exist

    Raises:
    OSError: any OSError raised by os.makedirs, except for the EEXIST condition

    Returns:
    path (string): the ensured path
    """
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as exception:
            #if the exception is raised because the directory already exits,
            #than our work is done and everything is OK, otherwise re-raise the error
            #THIS CAN OCCUR FROM A POSSIBLE RACE CONDITION!!!
            if exception.errno != errno.EEXIST:
                raise
    return path
#end ensure_dir()


def dict_to_h5(h5, dic, root='/', annotations=None):
    '''
    Save an dict to an h5 file, mounting at root.
    Keys are mapped to group names recursively.
    Parameters
    ----------
    h5 (h5py.File instance): h5py.file object to operate on
    dic (dict): dictionary of data to write
    root (string): group on which to add additional groups and datasets
    annotations (dict): annotation data to add to corresponding h5 datasets. Should contain same keys as dic.
    Returns
    -------
    None
    '''

    if not root.endswith('/'):
        root = root + '/'

    if annotations is None:
        annotations = {} #empty dict is better than None, but dicts shouldn't be default parameters

    for key, item in dic.items():
        dest = root + key
        try:
            if isinstance(item, (np.ndarray, np.int64, np.float64, str, bytes)):
                h5[dest] = item
            elif isinstance(item, (tuple, list)):
                h5[dest] = np.asarray(item)
            elif isinstance(item, (int, float)):
                h5[dest] = np.asarray([item])[0]
            elif item is None:
                h5.create_dataset(dest, data=h5py.Empty(dtype=h5py.special_dtype(vlen=str)))
            elif isinstance(item, dict):
                dict_to_h5(h5, item, dest)
            else:
                raise ValueError('Cannot save {} type to key {}'.format(type(item), dest))
        except Exception as e:
            print(e)
            if key != 'inputs':
                print('h5py could not encode key:', key)

        if key in annotations:
            if annotations[key] is None:
                h5[dest].attrs['description'] = ""
            else:
                h5[dest].attrs['description'] = annotations[key]


class Tee(object):
    ''' Pipes stdout/stderr to a file and stdout/stderr
    '''
    def __init__(self, name, mode='w'):
        self.name = name
        self.mode = mode
        self.file = None
        self.stdout = None
        self.stderr = None

    def attach(self):
        ''' Attach onto stderr/stdout
        '''
        self.file = open(self.name, self.mode, encoding='utf-8')

        self.stdout = sys.stdout
        sys.stdout = self

        self.stderr = sys.stderr
        sys.stderr = self

    def detach(self):
        ''' Detach from stderr/stdout
        '''
        sys.stdout = self.stdout
        sys.stderr = self.stderr
        self.file.close()

    def __del__(self):
        self.detach()

    def write(self, data):
        ''' Write data
        '''
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        ''' Flush output
        '''
        self.file.flush()



def click_param_annot(click_cmd):
    '''
    Given a click.Command instance, return a dict that maps option names to help strings.
    Currently skips click.Arguments, as they do not have help strings.
    Parameters
    ----------
    click_cmd (click.Command): command to introspect
    Returns
    -------
    annotations (dict): click.Option.human_readable_name as keys; click.Option.help as values
    '''

    annotations = {}
    for p in click_cmd.params:
        if isinstance(p, click.Option):
            annotations[p.human_readable_name] = p.help
    return annotations