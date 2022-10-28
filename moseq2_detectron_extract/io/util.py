import atexit
import cProfile
import errno
import io
import json
import logging
import os
import shutil
import sys
import traceback
import warnings
from logging.handlers import MemoryHandler
from pstats import Stats
from typing import IO, Dict, Optional, Union

import click
import h5py
import numpy as np
import ruamel.yaml as yaml
import tqdm
from tqdm.contrib.logging import _TqdmLoggingHandler


def gen_batch_sequence(nframes: int, chunk_size: int, overlap: int, offset: int=0):
    ''' Generate a sequence with overlap

    Parameters:
    nframes (int): number of frames to produce
    chunk_size (int): size of each chunk
    overlap (int): overlap between successive chunks
    offset (int): offset of the initial chunk
    '''
    seq = range(offset, nframes)
    for i in range(offset, len(seq)-overlap, chunk_size-overlap):
        yield seq[i:i+chunk_size]


def load_timestamps(timestamp_file: Union[str, IO[bytes]], col: int=0) -> np.ndarray:
    ''' Read timestamps from space delimited text file

    Parameters:
    timestamp_file (str): path to a file containing timestamp data
    col (int): column of the file containing timestamp data

    Returns:
    np.ndarray containing timestamp data.
    '''

    timestamps = []
    if isinstance(timestamp_file, str):
        with open(timestamp_file, 'r', encoding='utf-8') as ts_file:
            for line_str in ts_file:
                cols = line_str.split()
                timestamps.append(float(cols[col]))
        return np.array(timestamps)
    elif isinstance(timestamp_file, io.BufferedReader):
        # try iterating directly
        for line_bytes in timestamp_file:
            cols = line_bytes.decode().split()
            timestamps.append(float(cols[col]))
        return np.array(timestamps)
    else:
        raise ValueError('Could not understand parameter timestamp_file!')


def load_metadata(metadata_file: Union[str, IO[bytes]]) -> dict:
    ''' Load session metadata from a json file

    Parameters:
    metadata_file (str): path to the file containing metadata in json format

    Returns:
    dict containing session metadata
    '''
    if isinstance(metadata_file, str):
        with open(metadata_file, 'r', encoding='utf-8') as md_file:
            return json.load(md_file)
    elif isinstance(metadata_file, io.BufferedReader):
        return json.load(metadata_file)
    else:
        raise ValueError(f'Could not load metadata file "{metadata_file}"')



def read_yaml(yaml_file: str) -> dict:
    ''' Read a yaml file into dict object

    Parameters:
    yaml_file (str): path to yaml file

    Returns:
    return_dict (dict): dict of yaml contents
    '''
    with open(yaml_file, 'r', encoding='utf-8') as y_file:
        yml = yaml.YAML(typ='safe')
        return yml.load(y_file)


def write_yaml(yaml_file: str, data: dict) -> None:
    ''' Write a dict object into a yaml file

    Parameters:
    yaml_file (str): path to yaml file
    data (dict): dict of data to write to `yaml_file`
    '''
    with open(yaml_file, 'w', encoding='utf-8') as y_file:
        yml = yaml.YAML(typ='safe')
        yml.default_flow_style = False
        yml.dump(data, y_file)


def ensure_dir(path: str) -> str:
    ''' Ensures the path exists by creating the directories specified by path if they do not already exist.

    Parameters:
    path (string): path for which to ensure directories exist

    Raises:
    OSError: any OSError raised by os.makedirs, except for the EEXIST condition

    Returns:
    path (string): the ensured path
    '''
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


def dict_to_h5(h5_file: h5py.File, data: dict, root: str='/', annotations: dict=None) -> None:
    ''' Save an dict to an h5 file, mounting at root. Keys are mapped to group names recursively.

    Parameters:
    h5_file (h5py.File): h5py.file object to operate on
    data (dict): dictionary of data to write
    root (string): group on which to add additional groups and datasets
    annotations (dict): annotation data to add to corresponding h5 datasets. Should contain same keys as data.
    '''

    if not root.endswith('/'):
        root = root + '/'

    if annotations is None:
        annotations = {} #empty dict is better than None, but dicts shouldn't be default parameters

    for key, item in data.items():
        dest = root + key
        try:
            if isinstance(item, (np.ndarray, np.int64, np.float64, str, bytes)):
                h5_file[dest] = item
            elif isinstance(item, (tuple, list)):
                h5_file[dest] = np.asarray(item)
            elif isinstance(item, (int, float)):
                h5_file[dest] = np.asarray([item])[0]
            elif item is None:
                h5_file.create_dataset(dest, data=h5py.Empty(dtype=h5py.special_dtype(vlen=str)))
            elif isinstance(item, dict):
                dict_to_h5(h5_file, item, dest)
            else:
                raise ValueError(f'Cannot save {type(item)} type to key {dest}')
        except Exception as exc: # pylint: disable=broad-except
            logging.error(exc, exc_info=True)
            if key != 'inputs':
                logging.error(f'h5py could not encode key: "{key}"')

        if key in annotations:
            if annotations[key] is None:
                h5_file[dest].attrs['description'] = ""
            else:
                h5_file[dest].attrs['description'] = annotations[key]


def setup_logging(name: str=None, level: Union[str, int]=logging.INFO, add_defered_file_handler: bool=False):
    ''' Setup the logging module
    A) set logging level to `level` (default logging.INFO)
    b) optionally attach a memory handler, if add_defered_file_handler is True (enables adding a file handler later without losing records)
    C) attach stream handler to pump messages through tqdm.write
        a) filter LogRecords which have `nostream` attribute attached

    Parameters:
    name (str|None): Name of the Logger instance to operate on
    level (str|int): Set logging level to this level
    add_defered_file_handler (bool): If true, add a handler to buffer log records in memory.
        Use in combination with `attach_file_logger()` to later point these buffered records to a file
    '''
    logging.captureWarnings(True)
    #logging.lastResort = logging.NullHandler()

    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(level)

    # attach defered file handler, if requested
    if add_defered_file_handler:
        mem_handler = MemoryHandler(0)
        mem_handler.name = 'DEFERED_FILE_HANDLER'
        mem_handler.setLevel(level)
        logger.addHandler(mem_handler)

    # attach stream handler
    stream_handler = _TqdmLoggingHandler()
    stream_handler.setLevel(level)
    def filter_progress_records(record: logging.LogRecord):
        return not getattr(record, 'nostream', False)
    stream_handler.addFilter(filter_progress_records)
    logger.addHandler(stream_handler)


def attach_file_logger(log_filename: str, logger: logging.Logger=None):
    ''' Attach a logging.FileHandler to an existing logging.MemoryHandler

    Parameters:
    log_filename (str): path for FileHandler to write
    logger (logging.Logger): Operate on this Logger instance, otherwise use the root Logger
    '''
    if logger is None:
        logger = logging.getLogger()

    for handler in logger.handlers:
        if isinstance(handler, MemoryHandler) and handler.name == 'DEFERED_FILE_HANDLER':
            if handler.target is not None:
                raise RuntimeError('MemoryHandler already has a target!')
            file_handler = logging.FileHandler(log_filename)
            file_handler.setLevel(handler.level)
            handler.setTarget(file_handler)
            handler.flush()
            return

    # If we got here, then we could not find the MemoryHandler, we should warn the user!
    raise RuntimeError('Could not find a suitable handler to attach target!')




def click_param_annot(click_cmd: click.Command) -> Dict[str, Optional[str]]:
    ''' Given a click.Command instance, return a dict that maps option names to help strings.
    Currently skips click.Arguments, as they do not have help strings.

    Parameters:
    click_cmd (click.Command): command to introspect

    Returns:
    annotations (dict): click.Option.human_readable_name as keys; click.Option.help as values
    '''
    annotations = {}
    for param in click_cmd.params:
        if isinstance(param, click.Option):
            annotations[param.human_readable_name] = param.help
    return annotations


def click_monkey_patch_option_show_defaults():
    ''' Monkey patch click.core.Option to turn on showing default values.
    '''
    orig_init = click.core.Option.__init__
    def new_init(self, *args, **kwargs):
        ''' This version of click.core.Option.__init__ will set show default values to True
        '''
        orig_init(self, *args, **kwargs)
        self.show_default = True
    # end new_init()
    click.core.Option.__init__ = new_init # type: ignore


def enable_profiling():
    ''' Enable application profiling via cProfile
    '''
    logging.info("Enabling profiling...")
    profiler = cProfile.Profile()
    profiler.enable()

    def profile_exit():
        profiler.disable()
        logging.info("Profiling completed")
        with open('profiling_stats.txt', 'w', encoding='utf-8') as stream:
            stats = Stats(profiler, stream=stream)
            stats.strip_dirs()
            stats.sort_stats('time')
            stats.dump_stats('.prof_stats')
            stats.print_stats()
    atexit.register(profile_exit)


def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
    ''' write warning with stacktrace information
    '''

    log = file if hasattr(file, 'write') else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))


class ProgressFileObject(io.FileIO):
    """ Class used to provide provide file read progress updates """
    def __init__(self, path, *args, progress=None, tqdm_kwargs=None, **kwargs):
        """ Construct an instance of ProgressFileObject

        Parameters:
        path (string): Path of the file to open
        progress (tqdm instance): An (optional) instance of tqdm. If None, one is constructed for you
        *args: additional arguments passed to io.FileIO
        **kwargs: additional kwargs passed to io.FileIO

        """
        if tqdm_kwargs is None:
            tqdm_kwargs = {}

        self._total_size = os.path.getsize(path)
        if progress is not None:
            assert(isinstance(progress, (tqdm.tqdm,)))
            self.progress = progress
            self.is_progress_external = True
        else:
            self.progress = tqdm.tqdm(total=self._total_size, unit='bytes', unit_scale=True, **tqdm_kwargs)
            self.is_progress_external = False

        super().__init__(path, *args, **kwargs)

    def detach_progress(self):
        ''' Detach and return progress object
        '''
        progress = self.progress
        self.progress = None
        self.is_progress_external = False
        return progress

    def read(self, size):
        if self.progress:
            self.progress.update(size)
        return super().read(size)

    def close(self):
        if self.progress and not self.is_progress_external:
            self.progress.close()
        return super().close()
#end class ProgressFileObject


def backup_existing_file(origional_path: str) -> str:
    ''' Backup a file, ensuring no filename clashes

    Given a path, while that path exists, we append an incrementing suffix
    until the path no longer exists on the file system. The file is then
    copied to the new path. The new path is returned.

    Parameters:
    origional_path (str): name of the file that should be backed up

    Returns
    str - new file path
    '''

    counter = 0
    new_path = origional_path
    base, ext = os.path.splitext(origional_path)
    while os.path.exists(new_path):
        counter += 1
        new_path = f'{base}.backup-{counter}{ext}'

    shutil.copy2(origional_path, new_path)
    return new_path


def find_unused_file_path(path: str) -> str:
    ''' Find an unused file path on the filesystem

    Parameters:
    path (str): path to a file to find an unused version of
    '''
    if not os.path.exists(path):
        return path

    name, ext = os.path.splitext(path)
    i = 1
    new_fname = f"{name}-{i}{ext}"
    while os.path.exists(new_fname):
        i += 1
        new_fname = f"{name}-{i}{ext}"
    return new_fname


def find_unused_dataset_path(h5_file: str, path: str) -> str:
    ''' Find an unused dataset path in an h5 file

    Parameters:
    h5_file (str): Path to an h5 file to interrogate
    path (str): dataset path to find an unused name for

    Returns:
    potential name for a dataset which does not yet exist in `h5_file`
    '''
    new_path = path
    with h5py.File(h5_file, mode='r') as h5:
        if new_path in h5:
            i = 0
            while True:
                new_path = f'{path}_{i}'
                if new_path not in h5:
                    break
                i += 1
    return new_path
