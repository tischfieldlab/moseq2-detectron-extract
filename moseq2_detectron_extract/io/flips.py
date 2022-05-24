from datetime import datetime
import itertools
import sys

import h5py
import numpy as np


def read_flips_file(file_path):
    ''' Read a file containing flip annotations and return a list of flips

    Parameters:
    file_path (string): path to the filps file to parse

    Returns:
    ranges (list of list): list of flip ranges

    '''
    flips = []
    with(open(file_path, 'r', encoding='utf-8')) as flip_file:
        for lno, line in enumerate(flip_file):
            line = line.strip()
            if len(line) > 0:
                if line[0] == '#':
                    # ignore comment lines
                    continue

                try:
                    parts = [int(i.strip()) for i in line.split('-')]
                except ValueError as e:
                    raise RuntimeError(f'File {file_path} line {lno + 1}: Expected only integer indicies! "{line}"') from e

                if len(parts) != 2:
                    raise RuntimeError(f'File {file_path} line {lno + 1}: Expected only 2 indicies, but recieved {len(parts)}! "{line}"')

                flips.append(parts[:2])

    try:
        verify_ranges(flips)
    except RuntimeError as e:
        raise RuntimeError(f'File {file_path}:\n{str(e)}') from e

    return flips


def verify_ranges(ranges, vmin=0, vmax=sys.maxsize):
    ''' Verify that ranges is within bounds and there are no overlapping intervals

    Raises `RuntimeError` if any violations are found, otherwise return `True`

    Parameters:
    ranges (list): list of ranges to verify such as: [[0, 6], [8, 10]]
    vmin (int): minimum value allowed
    vmax (int): maximum value allowed

    Returns:
    True if all ranges are valid, otherwise raises a `RuntimeError`
    '''
    errors = []
    for start, stop in ranges:
        if stop < start:
            errors.append(f'Range ({start}, {stop}) stop cannot be less than start')
        if start < vmin:
            errors.append(f'Range ({start}, {stop}) start cannot be less than {vmin}')
        if stop > vmax:
            errors.append(f'Range ({start}, {stop}) stop cannot be greater than {vmax}')

    for r1, r2 in itertools.combinations(ranges, 2):
        if max(r1[0], r2[0]) <= min(r1[1], r2[1]):
            errors.append(f'Range ({r1[0]}, {r1[1]}) overlaps with range ({r2[0]}, {r2[1]})')

    if len(errors) > 0:
        raise RuntimeError('\n'.join(errors))

    return True


def flip_dataset(h5_file, flip_mask=None, flip_ranges=None, frames_path='/frames', frames_mask_path='/frames_mask',
                 angle_path='/scalars/angle', flips_path='/metadata/extraction/flips', flip_class=1):
    ''' Flip a dataset according to either `flip_mask` xor `flip_ranges`

    If `flip_ranges` is provided, it is converted to a `flip_mask` first. You cannot provide both `flip_mask` and `flip_ranges`,
    only one or the other. The value of `flip_ranges` is expected to be a list of lists of `start` and `stop` indicies which form slices,
    and must pass validation by the method `verify_ranges()`:
    ```
    flip_ranges = [
        [0, 10],  # start, stop
        [500, 1002]
    ]
    ```

    If `flips_path` already exists within the h5 file, a unique suffix will be appended

    Parameters:
        h5_file (string): Path to the h5 file to flip
        flip_mask (ndarray): mask of indicies to flip
        flip_ranges (list<list<int>>): list of frame ranges to flip
        frames_path (str): Path to the frames dataset within the h5 file
        frames_mask_path (str): Path to the frames_mask dataset within the h5 file
        angle_path (str): Path to the angle dataset within the h5 file
        flips_path (str): Path to the flips dataset within the h5 file
        flip_class (int): Value indicating a flip in flip_mask
    '''

    if flip_ranges is None and flip_mask is None:
        raise RuntimeError('One of flip_mask or flip_ranges must be supplied!')

    if flip_ranges is not None and flip_mask is not None:
        raise RuntimeError('Cannot supply both flip_mask and flip_ranges!')

    with(h5py.File(h5_file, 'r+')) as h5:
        nframes = h5[frames_path].shape[0]

        # if we were given flip_ranges, convert to flip_mask
        if flip_ranges is not None:
            verify_ranges(flip_ranges, vmax=nframes)
            flip_mask = np.zeros(nframes, dtype=np.bool)
            for start, stop in flip_ranges:
                flip_mask[start:stop] = flip_class
        else:
            flip_mask = flip_mask == flip_class

        # find a path for flips that is not already in use
        if flips_path in h5:
            i = 1
            while True:
                new_flips_path = f'{flips_path}_{i}'
                if new_flips_path not in h5:
                    flips_path = new_flips_path
                    break
                i += 1

        # create and set the flips dataset
        h5.create_dataset(flips_path, data=flip_mask, dtype='bool', compression='gzip')
        h5[flips_path].attrs['description'] = 'Output from flip classifier, false=no flip, true=flip'
        h5[flips_path].attrs['description'] = f'Created by moseq2-detectron-extract, manual flips, on {datetime.now()}'

        # apply flips to the datasets
        h5[frames_path][flip_mask, ...] = flip_horizontal(h5[frames_path][flip_mask, ...])
        h5[frames_mask_path][flip_mask, ...] = flip_horizontal(h5[frames_mask_path][flip_mask, ...])
        h5[angle_path][flip_mask] += np.pi

        # TODO: flip also keypoints!

        h5.flush()


def flip_horizontal(data: np.ndarray) -> np.ndarray:
    ''' Flip an array of frames horizontally

    Parameters:
    data (ndarray): data to flip

    Returns:
    ndarray: data flipped horizontally
    '''
    return np.rot90(data, k=2, axes=(-2, -1))


def flip_vertical(data: np.ndarray) -> np.ndarray:
    ''' Flip an array of frames vertically

    Parameters:
    data (ndarray): data to flip

    Returns:
    ndarray: data flipped vertically
    '''
    return np.flip(data, axis=-2)
