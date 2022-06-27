

import tarfile
from typing import Callable, Iterable

import numpy as np
from moseq2_detectron_extract.io.video import load_movie_data




class Driver():
    ''' Base driver '''

    def __init__(self) -> None:
        self.filters = []

    def attach_filter(self, filterer: Callable[[np.ndarray], np.ndarray]):
        ''' Attach a filter to this frames iterator. A filter is simply a callable accepting a
            numpy array of data, performs some operation upon it, and returns and output array.
            Multiple filters can be attached, and they are called in the order in which they were
            attachd. You may also specify the streams upon which the filter should apply.

        Parameters:
        filterer (Callable[[np.ndarray], np.ndarray]): callable used to filter data
        '''
        self.filters.append(filterer)

    def read(self, frame_idxs: Iterable[int]):
        ''' Read frames corresponding to `frame_idxs`
        '''
        data = self._read(frame_idxs)
        return self._run_filters(data)

    def _read(self, frame_idxs: Iterable[int]):
        raise NotImplementedError()

    def _run_filters(self, data: np.ndarray) -> np.ndarray:
        for filt in self.filters:
            data = filt(data)
        return data


class DepthReader(Driver):
    ''' Driver supporting reading depth video data
    '''

    def __init__(self, depth_file: str, tar_object: tarfile.TarFile=None) -> None:
        super().__init__()
        self.depth_file = depth_file
        self.tar_object = tar_object

    def _read(self, frame_idxs: Iterable[int]):
        return load_movie_data(self.depth_file, frame_idxs, tar_object=self.tar_object)


class ColorReader(Driver):
    ''' Driver supporting reading RGB video data
    '''

    def __init__(self, rgb_file: str, tar_object: tarfile.TarFile=None, pixel_format:str='rgb24') -> None:
        super().__init__()
        self.rgb_file = rgb_file
        self.tar_object = tar_object
        self.pixel_format = pixel_format

    def _read(self, frame_idxs: Iterable[int]):
        return load_movie_data(self.rgb_file, frame_idxs, pixel_format='rgb24', tar_object=self.tar_object)
