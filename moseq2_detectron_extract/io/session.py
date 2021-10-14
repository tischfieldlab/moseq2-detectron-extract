from enum import Enum
import os
import tarfile
from typing import Iterable, Sequence, Tuple, Union
from moseq2_detectron_extract.io.image import read_image, write_image
from moseq2_detectron_extract.io.util import (gen_batch_sequence, load_metadata,
                                         load_timestamps)
from moseq2_detectron_extract.io.video import get_movie_info, load_movie_data
from moseq2_detectron_extract.io.proc import (apply_roi, get_bground_im_file,
                                           get_roi, select_strel)
import numpy as np
import tqdm

class Stream(str, Enum):
    Depth = 'depth'
    RGB = 'rgb'


class Session(object):

    def __init__(self, path: str, frame_trim: Tuple[int, int]=(0, 0)):
        self.__init_session(path)
        self.__trim_frames(frame_trim)


    def __init_session(self, input_file: str):
        self.dirname = os.path.dirname(input_file)

        if input_file.endswith('.tar.gz') or input_file.endswith('.tgz'):
            with tqdm.tqdm(total=1, leave=False, desc='Scanning tarball {} (this will take a minute)'.format(input_file)) as pbar:
                #compute NEW psuedo-dirname now, `input_file` gets overwritten below with depth.dat tarinfo...
                self.dirname = os.path.join(self.dirname, os.path.basename(input_file).replace('.tar.gz', '').replace('.tgz', ''))

                self.tar = tarfile.open(input_file, 'r:gz')
                self.tar_members = self.tar.getmembers()
                self.tar_names = [_.name for _ in self.tar_members]
                self.depth_file = self.tar_members[self.tar_names.index('depth.dat')]
                self.rgb_file = self.tar_members[self.tar_names.index('rgb.mp4')]
                self.session_id = os.path.basename(input_file).split('.')[0]
                pbar.update(1)
        else:
            self.tar = None
            self.tar_members = None
            self.depth_file = input_file
            self.rgb_file = os.path.join(self.dirname, 'rgb.mp4')
            self.session_id = os.path.basename(self.dirname)

        self.depth_metadata = get_movie_info(self.depth_file, tar_object=self.tar)
        self.rgb_metadata = get_movie_info(self.rgb_file, tar_object=self.tar)
    #end init_session()

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
    #end trim_frames()

    @property
    def is_compressed(self):
        return self.tar is not None

    def load_metadata(self):
        if self.tar is not None:
            metadata_path = self.tar.extractfile(self.tar_members[self.tar_names.index('metadata.json')])
        else:
            metadata_path = os.path.join(self.dirname, 'metadata.json')
        return load_metadata(metadata_path)
    #end load_metadata()

    def load_timestamps(self, stream: Stream):
        timestamp_path = None
        correction_factor = 1.0
        ts_search = []

        if stream == Stream.Depth:
            ts_search.append(('depth_ts.txt', 1.0))
            ts_search.append(('timestamps.csv', 1000.0))
        elif stream == Stream.RGB:
            ts_search.append(('rgb_ts.txt', 1.0))
        else:
            raise ValueError(f"unknown stream {stream}")

        if self.tar is not None:
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

        timestamps = load_timestamps(timestamp_path, col=0)

        if timestamps is not None:
            timestamps = timestamps[self.first_frame_idx:self.last_frame_idx]

        timestamps *= correction_factor

        return timestamps
    #end load_timestamps()

    def find_roi(self, bg_roi_dilate: Tuple[int, int]=(10,10), bg_roi_shape='ellipse', bg_roi_index: int=0, bg_roi_weights=(1, .1, 1),
                 bg_roi_depth_range: Tuple[int, int]=(650, 750), bg_roi_gradient_filter: bool=False, bg_roi_gradient_threshold: int=3000,
                 bg_roi_gradient_kernel: int=7, bg_roi_fill_holes: bool=True, use_plane_bground: bool=False, verbose: bool=False, cache_dir: Union[None, str]=None):

        if cache_dir and os.path.exists(os.path.join(cache_dir, 'bground.tiff')):
            if verbose:
                print('Loading background...')
            bground_im = read_image(os.path.join(cache_dir, 'bground.tiff'), scale=True)
        else:
            if verbose:
                print('Getting background...')
            bground_im = get_bground_im_file(self.depth_file, tar_object=self.tar)

            if cache_dir and not use_plane_bground:
                write_image(os.path.join(cache_dir, 'bground.tiff'), bground_im, scale=True)

        if cache_dir:
            first_frame = load_movie_data(self.depth_file, 0, tar_object=self.tar)
            write_image(os.path.join(cache_dir, 'first_frame.tiff'), first_frame[0], scale=True, scale_factor=bg_roi_depth_range)

        strel_dilate = select_strel(bg_roi_shape, bg_roi_dilate)

        roi_filename = 'roi_{:02d}.tiff'.format(bg_roi_index)

        if cache_dir and os.path.exists(os.path.join(cache_dir, roi_filename)):
            if verbose:
                print('Loading ROI...')
            roi = read_image(os.path.join(cache_dir, roi_filename), scale=True) > 0
        else:
            if verbose:
                print('Getting roi...')
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
                    print('Using plane fit for background...')
                xx, yy = np.meshgrid(np.arange(bground_im.shape[1]), np.arange(bground_im.shape[0]))
                coords = np.vstack((xx.ravel(), yy.ravel()))
                plane_im = (np.dot(coords.T, plane[:2]) + plane[3]) / -plane[2]
                plane_im = plane_im.reshape(bground_im.shape)
                if cache_dir:
                    write_image(os.path.join(cache_dir, 'bground.tiff'), plane_im, scale=True)
                bground_im = plane_im

            roi = rois[bg_roi_index]
            if cache_dir:
                write_image(os.path.join(cache_dir, roi_filename), roi, scale=True, dtype='uint8')

        true_depth = np.median(bground_im[roi > 0])
        if verbose:
            print('Detected true depth: {}'.format(true_depth))

        return bground_im, roi, true_depth
    #end find_roi()

    def iterate(self, chunk_size: int=1000, chunk_overlap: int=0, streams: Iterable[Stream]=(Stream.Depth,)):
        ''' Iterate over all frames, returning `chunck_size` frames on each iteration with `chunk_overlap` overlap

            Parameters:
                chunk_size (int): Number of frames to return on each iteration
                chunk_overlap (int): Number of frames each iteration should overlap with the previous iteration
                streams (Iterable[Stream]): Streams from which to return data
        '''
        return SessionFramesIterator(self, chunk_size, chunk_overlap, streams)

    def sample(self, num_samples: int, chunk_size: int=1000, streams: Iterable[Stream]=(Stream.Depth,)):
        ''' Randomally sample `num_samples` frames, returning `chunck_size` frames on each iteration.

            Parameters:
                num_samples (int): Total number of frames to sample
                chunk_size (int): Number of frames to return on each iteration
                streams (Iterable[Stream]): Streams from which to return data
        '''
        return SessionFramesSampler(self, num_samples, chunk_size=chunk_size, chunk_overlap=0, streams=streams)

    def index(self, frame_idxs: Sequence[int], chunk_size: int=1000, streams: Iterable[Stream]=(Stream.Depth,)):
        ''' Fetch specific frames, given by `frame_idxs`, returning `chunck_size` frames on each iteration.

            Parameters:
                frame_idxs (Sequence[int]): Frame indicies that should be fetched
                chunk_size (int): Number of frames to return on each iteration
                streams (Iterable[Stream]): Streams from which to return data
        '''
        return SessionFramesIndexer(self, frame_idxs, chunk_size=chunk_size, chunk_overlap=0, streams=streams)

    def __str__(self) -> str:
        return "{} ({} frames, [{}:{}])".format(self.depth_file, self.nframes, self.first_frame_idx, self.last_frame_idx)

    def __repr__(self) -> str:
        return '{}("{}", frame_trim=({}, {}))'.format(self.__class__.__name__, self.depth_file, *self.frame_trim)
#end class Session


class SessionFramesIterator(object):
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
        self.streams = list(set(streams))
        self.ts_map = TimestampMapper()
        for stream in streams:
            self.ts_map.add_timestamps(stream, session.load_timestamps(stream))

    def generate_samples(self):
        ''' Generate the sequence of batches of frames indicies

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

        for stream in self.streams:
            if stream == Stream.Depth:
                out_data.append(load_movie_data(self.session.depth_file, frame_idxs, tar_object=self.session.tar))
            elif stream == Stream.RGB:
                # rgb_idxs = self.ts_map.map_index(Stream.RGB, Stream.Depth, frame_idxs)
                out_data.append(load_movie_data(self.session.rgb_file, frame_idxs, pixel_format='rgb24', tar_object=self.session.tar))

        return tuple(out_data)


class SessionFramesSampler(SessionFramesIterator):
    def __init__(self, session: Session, num_samples: int, chunk_size: int, chunk_overlap: int, streams: Iterable[Stream]):
        self.num_samples = int(num_samples)
        super().__init__(session, chunk_size, chunk_overlap, streams)

    def generate_samples(self):
        """Generate a sequence with overlap
        """
        offset = self.session.first_frame_idx
        seq = range(offset, self.session.nframes)
        seq = np.random.choice(seq, self.num_samples, replace=False)
        for i in range(offset, len(seq)-self.chunk_overlap, self.chunk_size-self.chunk_overlap):
            yield seq[i:i+self.chunk_size]


class SessionFramesIndexer(SessionFramesIterator):
    def __init__(self, session: Session, frame_idxs: Sequence[int], chunk_size: int, chunk_overlap: int, streams: Iterable[Stream]):
        self.frame_idxs = frame_idxs
        super().__init__(session, chunk_size, chunk_overlap, streams)

    def generate_samples(self):
        """Generate a sequence with overlap
        """
        offset = self.session.first_frame_idx
        for i in range(offset, len(self.frame_idxs)-self.chunk_overlap, self.chunk_size-self.chunk_overlap):
            yield self.frame_idxs[i:i+self.chunk_size]



class TimestampMapper():
    def __init__(self) -> None:
        self.timestamp_map = {}

    def add_timestamps(self, name: str, timestamps: np.array):
        self.timestamp_map[name] = np.asarray(timestamps)

    def map_index(self, query: str, reference: str, index: Union[int, Sequence[int]]):
        if isinstance(index, int):
            index = [index]
        
        out = []
        for idx in index:
            reference_time = self.timestamp_map[reference][idx]
            out.append(self.nearest(self.timestamp_map[query], reference_time))
        return out

    def map_time(self, query: str, reference: str, index: Union[int, Sequence[int]]):
        if isinstance(index, int):
            index = [index]

        out = []
        for idx in index:
            reference_time = self.timestamp_map[reference][idx]
            query_idx = self.nearest(self.timestamp_map[query], reference_time)
            out.append(self.timestamp_map[query][query_idx])
        return out

    def nearest(self, data, value):
        return np.abs(data - value).argmin()
