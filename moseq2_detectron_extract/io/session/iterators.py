

from typing import Callable, Iterable, List, Sequence, Union

import numpy as np
from moseq2_detectron_extract.io.video import load_movie_data

from .filters import _FilterItem
from .session import Session, Stream
from .time_map import TimestampMapper


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
        self.streams = list(set(streams))
        self.filters: List[_FilterItem] = []
        self.ts_map = TimestampMapper()
        for stream in streams:
            self.ts_map.add_timestamps(stream, session.load_timestamps(stream))

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

        for stream in self.streams:
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
