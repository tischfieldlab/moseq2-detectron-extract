import os
from typing import Union
import numpy as np
import json
import errno

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