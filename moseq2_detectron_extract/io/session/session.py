import logging
import os
import tarfile
from enum import Enum
from typing import IO, Callable, Iterable, List, Sequence, Tuple, TypedDict, Union

import numpy as np
from moseq2_detectron_extract.io.image import read_tiff_image, write_image
from moseq2_detectron_extract.io.util import (ProgressFileObject,
                                              load_metadata, load_timestamps)
from moseq2_detectron_extract.io.video import get_movie_info, load_movie_data
from moseq2_detectron_extract.proc.roi import (apply_roi, get_bground_im_file,
                                               get_roi)
from moseq2_detectron_extract.proc.util import select_strel

from .drivers import ColorReader, DepthReader
from .time_map import TimestampMapper


class Stream(str, Enum):
    ''' Represents a specific stream contained within a session
    '''
    DEPTH = 'depth'
    RGB = 'rgb'


class _FilterItem(TypedDict):
    filter: Callable[[np.ndarray], np.ndarray]
    streams: Iterable[Stream]


class Session(object):
    ''' Represents a (possibly compressed) Moseq Session
    '''

    def __init__(self, path: str, frame_trim: Tuple[int, int]=(0, 0)):
        self.session_path = path
        self.__init_session(path)
        self.__trim_frames(frame_trim)

        self.tar: Union[None, tarfile.TarFile]
        self.tar_members: Union[None, List[tarfile.TarInfo]]
        self.depth_file: Union[str, tarfile.TarInfo]
        self.rgb_file: Union[str, tarfile.TarInfo]

        self._true_depth: Union[None, float] = None
        self._first_frame: Union[None, np.ndarray] = None
        self._bground_im: Union[None, np.ndarray] = None
        self._roi: Union[None, np.ndarray] = None
        self._roi_bground_im: Union[None, np.ndarray] = None

    def __init_session(self, input_file: str):
        self.dirname = os.path.dirname(input_file)

        if input_file.endswith('.tar.gz') or input_file.endswith('.tgz'):
            tqdm_args = {
                'disable': False,
                'desc': f'Scanning {os.path.basename(input_file)}',
                'leave': False
            }
            #compute NEW psuedo-dirname now, `input_file` gets overwritten below with depth.dat tarinfo...
            self.dirname = os.path.join(self.dirname, os.path.basename(input_file).replace('.tar.gz', '').replace('.tgz', ''))

            pfo = ProgressFileObject(input_file, tqdm_kwargs=tqdm_args)
            self.tar = tarfile.open(fileobj=pfo, mode='r:*')
            self.tar_members = self.tar.getmembers()
            self.tar_names = [_.name for _ in self.tar_members]
            self.depth_file = self.tar_members[self.tar_names.index('depth.dat')]
            self.rgb_file = self.tar_members[self.tar_names.index('rgb.mp4')]
            self.session_id = os.path.basename(input_file).split('.')[0]

            pfo.detach_progress().close()

        else:
            self.tar = None
            self.tar_members = None
            self.depth_file = input_file
            self.rgb_file = os.path.join(self.dirname, 'rgb.mp4')
            self.session_id = os.path.basename(self.dirname)

        self.depth_metadata = get_movie_info(self.depth_file, tar_object=self.tar)
        self.rgb_metadata = get_movie_info(self.rgb_file, tar_object=self.tar)


    def __trim_frames(self, frame_trim: Tuple[int, int]):
        self.frame_trim = frame_trim
        self.nframes = self.depth_metadata['nframes']

        if frame_trim[0] > 0 and frame_trim[0] < self.nframes:
            self.first_frame_idx = frame_trim[0]
        else:
            self.first_frame_idx = 0

        if self.nframes - frame_trim[1] > self.first_frame_idx:
            self.last_frame_idx = self.nframes - frame_trim[1]
        else:
            self.last_frame_idx = self.nframes

        self.nframes = self.last_frame_idx - self.first_frame_idx


    @property
    def is_compressed(self) -> bool:
        ''' Tells if this session is compressed (True) or not (False)

        Returns:
        True if this session is compressed; False if this session is not compressed
        '''
        return self.tar is not None


    def load_metadata(self) -> dict:
        ''' Load metadata from this session (metadata.json)

        Returns:
        dict, metadata for this session
        '''
        metadata_path: Union[str, IO[bytes]]
        if self.tar is not None and self.tar_members is not None:
            tinfo = self.tar_members[self.tar_names.index('metadata.json')]
            efile = self.tar.extractfile(tinfo)
            if efile is not None:
                metadata_path = efile
            else:
                raise ValueError('Could not find metadata in tar!')
        else:
            metadata_path = os.path.join(self.dirname, 'metadata.json')
        return load_metadata(metadata_path)


    def load_timestamps(self, stream: Stream) -> np.ndarray:
        ''' Load timestamps for `stream` from this session

        Parameters:
        stream (Stream): stream for which to retrieve timestamps for

        Returns:
        np.ndarray, array of timestamps
        '''
        timestamp_path: Union[str, IO[bytes], None] = None
        correction_factor = 1.0
        ts_search = []

        if stream == Stream.DEPTH:
            ts_search.append(('depth_ts.txt', 1.0))
            ts_search.append(('timestamps.csv', 1000.0))
        elif stream == Stream.RGB:
            ts_search.append(('rgb_ts.txt', 1.0))
        else:
            raise ValueError(f"unknown stream {stream}")

        if self.tar is not None and self.tar_members is not None:
            for name, corr_factor in ts_search:
                if name in self.tar_names:
                    timestamp_path = self.tar.extractfile(self.tar_members[self.tar_names.index(name)])
                    correction_factor = corr_factor
                    break

        else:
            for name, corr_factor in ts_search:
                ts_path = os.path.join(self.dirname, name)
                if os.path.exists(ts_path):
                    timestamp_path = ts_path
                    correction_factor = corr_factor
                    break

        if timestamp_path is None:
            raise ValueError('Could not locate timestamp file!')
        else:
            timestamps = load_timestamps(timestamp_path, col=0)

            if timestamps is not None:
                timestamps = timestamps[self.first_frame_idx:self.last_frame_idx]

            timestamps *= correction_factor

            return timestamps


    def find_roi(self, bg_roi_dilate: Tuple[int, int]=(10,10), bg_roi_shape='ellipse', bg_roi_index: int=0, bg_roi_weights=(1, .1, 1),
                 bg_roi_depth_range: Tuple[int, int]=(650, 750), bg_roi_gradient_filter: bool=False, bg_roi_gradient_threshold: int=3000,
                 bg_roi_gradient_kernel: int=7, bg_roi_fill_holes: bool=True, use_plane_bground: bool=False, verbose: bool=False,
                 cache_dir: Union[None, str]=None):
        ''' Find a region of interest in depth frames
        '''
        if cache_dir is None:
            use_cache = False
            cache_dir = ''
        else:
            use_cache = True

        # Grab the first frame of the video and write it out to a file, but only if we have a place to save it
        ff_filename = os.path.join(cache_dir, 'first_frame.tiff')
        if use_cache and os.path.exists(ff_filename):
            first_frame = read_tiff_image(ff_filename, scale=True)
        else:
            first_frame = load_movie_data(self.depth_file, 0, tar_object=self.tar)
            if use_cache:
                write_image(ff_filename, first_frame[0], scale=True, scale_factor=bg_roi_depth_range)


        # compute the background, or load one from the cache
        bg_filename = os.path.join(cache_dir, 'bground.tiff')
        if use_cache and os.path.exists(bg_filename):
            if verbose:
                logging.info('Loading background...')
            bground_im = read_tiff_image(bg_filename, scale=True)
        else:
            if verbose:
                logging.info('Computing background...')
            bground_im = get_bground_im_file(self.depth_file, tar_object=self.tar)

            if use_cache and not use_plane_bground:
                write_image(bg_filename, bground_im, scale=True)


        # compute the region of interest, or load one from the cache
        roi_filename = os.path.join(cache_dir, f'roi_{bg_roi_index:02d}.tiff')
        if use_cache and os.path.exists(roi_filename):
            if verbose:
                logging.info('Loading ROI...')
            roi = read_tiff_image(roi_filename, scale=True) > 0
        else:
            if verbose:
                logging.info('Computing roi...')
            strel_dilate = select_strel(bg_roi_shape, bg_roi_dilate)
            rois, plane, _, _, _, _ = get_roi(bground_im,
                                            strel_dilate=strel_dilate,
                                            weights=bg_roi_weights,
                                            depth_range=bg_roi_depth_range,
                                            gradient_filter=bg_roi_gradient_filter,
                                            gradient_threshold=bg_roi_gradient_threshold,
                                            gradient_kernel=bg_roi_gradient_kernel,
                                            fill_holes=bg_roi_fill_holes,
                                            progress_bar=verbose)

            if use_plane_bground:
                if verbose:
                    logging.info('Using plane fit for background...')
                xx, yy = np.meshgrid(np.arange(bground_im.shape[1]), np.arange(bground_im.shape[0]))
                coords = np.vstack((xx.ravel(), yy.ravel()))
                plane_im = (np.dot(coords.T, plane[:2]) + plane[3]) / -plane[2]
                plane_im = plane_im.reshape(bground_im.shape)
                if use_cache:
                    write_image(bg_filename, plane_im, scale=True)
                bground_im = plane_im

            roi = rois[bg_roi_index]
            if use_cache:
                write_image(roi_filename, roi, scale=True, dtype='uint8')

        true_depth = np.median(bground_im[roi > 0])
        if verbose:
            logging.info(f'Detected true depth: {true_depth}')

        self._true_depth = true_depth
        self._first_frame = first_frame
        self._bground_im = bground_im
        self._roi = roi
        self._roi_bground_im = apply_roi(np.copy(bground_im)[None, :, :], roi)[0]

        return first_frame, bground_im, roi, true_depth


    @property
    def true_depth(self) -> float:
        ''' Gets the detected true depth for this session. Necessitates calling Session.find_roi() first
        '''
        if self._true_depth is not None:
            return self._true_depth
        raise RuntimeError('You must first call Session.find_roi() to use this property!')


    @property
    def first_frame(self) -> np.ndarray:
        ''' Gets the first frame of this session. Necessitates calling Session.find_roi() first
        '''
        if self._first_frame is not None:
            return self._first_frame
        raise RuntimeError('You must first call Session.find_roi() to use this property!')


    @property
    def bground_im(self) -> np.ndarray:
        ''' Gets the background image for this session. Necessitates calling Session.find_roi() first
        '''
        if self._bground_im is not None:
            return self._bground_im
        raise RuntimeError('You must first call Session.find_roi() to use this property!')


    @property
    def roi(self) -> np.ndarray:
        ''' Gets the ROI for this session. Necessitates calling Session.find_roi() first
        '''
        if self._roi is not None:
            return self._roi
        raise RuntimeError('You must first call Session.find_roi() to use this property!')


    @property
    def roi_bground_im(self) -> np.ndarray:
        ''' Gets the background image with the ROI applied for this session. Necessitates calling Session.find_roi() first
        '''
        if self._roi_bground_im is not None:
            return self._roi_bground_im
        raise RuntimeError('You must first call Session.find_roi() to use this property!')


    def iterate(self, chunk_size: int=1000, chunk_overlap: int=0, streams: Iterable[Stream]=(Stream.DEPTH,)):
        ''' Iterate over all frames, returning `chunck_size` frames on each iteration with `chunk_overlap` overlap

            Parameters:
                chunk_size (int): Number of frames to return on each iteration
                chunk_overlap (int): Number of frames each iteration should overlap with the previous iteration
                streams (Iterable[Stream]): Streams from which to return data
        '''
        return SessionFramesIterator(self, chunk_size, chunk_overlap, streams)


    def sample(self, num_samples: int, chunk_size: int=1000, streams: Iterable[Stream]=(Stream.DEPTH,)):
        ''' Randomally sample `num_samples` frames, returning `chunck_size` frames on each iteration.

            Parameters:
                num_samples (int): Total number of frames to sample
                chunk_size (int): Number of frames to return on each iteration
                streams (Iterable[Stream]): Streams from which to return data
        '''
        return SessionFramesSampler(self, num_samples, chunk_size=chunk_size, chunk_overlap=0, streams=streams)


    def index(self, frame_idxs: Sequence[int], chunk_size: int=1000, streams: Iterable[Stream]=(Stream.DEPTH,)):
        ''' Fetch specific frames, given by `frame_idxs`, returning `chunck_size` frames on each iteration.

            Parameters:
                frame_idxs (Sequence[int]): Frame indices that should be fetched
                chunk_size (int): Number of frames to return on each iteration
                streams (Iterable[Stream]): Streams from which to return data
        '''
        return SessionFramesIndexer(self, frame_idxs, chunk_size=chunk_size, chunk_overlap=0, streams=streams)


    def __str__(self) -> str:
        return f"{self.session_path} ({self.nframes} frames, [{self.first_frame_idx}:{self.last_frame_idx}])"


    def __repr__(self) -> str:
        return f'{self.__class__.__name__}("{self.session_path}", frame_trim=({self.frame_trim[0]}, {self.frame_trim[1]}))'


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


class SessionFramesIterator(object):
    ''' Iterator that iterates over Session frames in order
    '''
    def __init__(self, session: Session, chunk_size: int, chunk_overlap: int, streams: Iterable[Stream]):
        ''' Iterator that iterates over Session frames

        Parameters:
            session (Session): Session over which this iterator should iterate
            chunk_size (int): each iteration should produce `chunk_size` frames
            chunk_overlap (int): iterations should overlap but this number of frames
        '''
        self.session = session
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.batches = list(self.generate_samples())
        self.current = 0
        self._streams = list(set(streams))
        self.filters: List[_FilterItem] = []
        self.ts_map = TimestampMapper()
        self._drivers = build_drivers(self.session, self._streams)
        for stream in streams:
            self.ts_map.add_timestamps(stream, session.load_timestamps(stream))

    @property
    def streams(self) -> int:
        ''' Get the streams that this iterator will generate
        '''
        return self._streams

    @property
    def nframes(self) -> int:
        ''' Get the total number of frames this iterator will produce (batches * chunk_size)
        '''
        return sum([len(batch) for batch in self.batches])

    @property
    def nbatches(self) -> int:
        ''' Get the total number of batches produced by this iterator
            Same as calling len(iterator)
        '''
        return len(self.batches)

    def attach_filter(self, stream: Union[Stream, Iterable[Stream]], filterer: Callable[[np.ndarray], np.ndarray]):
        ''' Attach a filter to this frames iterator. A filter is simply a callable accepting a
            numpy array of data, performs some operation upon it, and returns and output array.
            Multiple filters can be attached, and they are called in the order in which they were
            attachd. You may also specify the streams upon which the filter should apply.

        Parameters:
        filterer (Callable[[np.ndarray], np.ndarray]): callable used to filter data
        streams (Iterable[Stream]): streams for which this filter should apply
        '''
        streams: List[Stream]
        if isinstance(stream, Stream):
            streams = [stream]
        else:
            streams = list(stream)

        self.filters.append({
            'filter': filterer,
            'streams': streams
        })

    def __apply_filters(self, data: np.ndarray, stream: Stream) -> np.ndarray:
        for filt in self.filters:
            if stream in filt['streams']:
                data = filt['filter'](data)
        return data

    def generate_samples(self):
        ''' Generate the sequence of batches of frames indices

        Default is to linear read. Override this if you want to change behaviour

        Returns
        list<(range|list<int>)>
        '''
        return gen_batch_sequence(self.session.nframes, self.chunk_size, self.chunk_overlap, self.session.first_frame_idx)

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        return self

    def __next__(self):
        if self.current >= len(self):
            raise StopIteration
        else:
            frame_range = self.batches[self.current]
        self.current += 1

        frame_idxs = list(frame_range)

        out_data = [frame_idxs]

        for stream in self._streams:
            if stream == Stream.DEPTH:
                data = load_movie_data(self.session.depth_file, frame_idxs, tar_object=self.session.tar)
            elif stream == Stream.RGB:
                # rgb_idxs = self.ts_map.map_index(Stream.RGB, Stream.Depth, frame_idxs)
                data = load_movie_data(self.session.rgb_file, frame_idxs, pixel_format='rgb24', tar_object=self.session.tar)
            else:
                raise ValueError(f'Unsupported stream "{stream}"')

            data = self.__apply_filters(data, stream)
            out_data.append(data)

        return tuple(out_data)


class SessionFramesSampler(SessionFramesIterator):
    ''' Iterator which randomly samples `num_frames` indices
    '''
    def __init__(self, session: Session, num_samples: int, chunk_size: int, chunk_overlap: int, streams: Iterable[Stream]):
        self.num_samples = int(num_samples)
        super().__init__(session, chunk_size, chunk_overlap, streams)


    def generate_samples(self):
        '''Generate a sequence with overlap
        '''
        offset = self.session.first_frame_idx
        seq = range(offset, self.session.nframes)
        seq = np.random.choice(seq, self.num_samples, replace=False)
        for i in range(offset, len(seq)-self.chunk_overlap, self.chunk_size-self.chunk_overlap):
            yield seq[i:i+self.chunk_size]


class SessionFramesIndexer(SessionFramesIterator):
    ''' Iterator which iterates from a fixed sequence of indices
    '''
    def __init__(self, session: Session, frame_idxs: Sequence[int], chunk_size: int, chunk_overlap: int, streams: Iterable[Stream]):
        self.frame_idxs = frame_idxs
        super().__init__(session, chunk_size, chunk_overlap, streams)


    def generate_samples(self):
        '''Generate a sequence with overlap
        '''
        offset = self.session.first_frame_idx
        for i in range(offset, len(self.frame_idxs)-self.chunk_overlap, self.chunk_size-self.chunk_overlap):
            yield self.frame_idxs[i:i+self.chunk_size]




def build_drivers(session: Session, streams: Iterable[Stream]):
    ''' Build drivers '''
    drivers = []
    for stream in streams:
        if stream == Stream.DEPTH:
            drivers.append(DepthReader(depth_file=session.depth_file, tar_object=session.tar))

        elif stream == Stream.RGB:
            drivers.append(ColorReader(rgb_file=session.rgb_file, tar_object=session.tar))

        else:
            raise ValueError(f'Unsupported stream "{stream}"')

    return drivers
