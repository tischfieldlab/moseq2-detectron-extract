from typing import Sequence, Union

import numpy as np


class TimestampMapper():
    ''' Map timestamps between various data streams
    '''
    def __init__(self) -> None:
        self.timestamp_map: dict = {}


    def add_timestamps(self, name: str, timestamps: np.ndarray):
        ''' Add timestampes to this mapper
        '''
        self.timestamp_map[name] = np.asarray(timestamps)


    def map_index(self, query: str, reference: str, index: Union[int, Sequence[int]]):
        ''' map a query index to a timestamp
        '''
        if isinstance(index, int):
            index = [index]

        out = []
        for idx in index:
            reference_time = self.timestamp_map[reference][idx]
            out.append(self.nearest(self.timestamp_map[query], reference_time))
        return out


    def map_time(self, query: str, reference: str, index: Union[int, Sequence[int]]):
        ''' map a query timestamp to an index
        '''
        if isinstance(index, int):
            index = [index]

        out = []
        for idx in index:
            reference_time = self.timestamp_map[reference][idx]
            query_idx = self.nearest(self.timestamp_map[query], reference_time)
            out.append(self.timestamp_map[query][query_idx])
        return out


    def nearest(self, data, value):
        ''' get a value from data which is nearest to value
        '''
        return np.abs(data - value).argmin()
