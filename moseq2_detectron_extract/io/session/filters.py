from functools import partial

import numpy as np
from moseq2_detectron_extract.proc.fast_hha_encode import HHAEncoder
from moseq2_detectron_extract.proc.proc import prep_raw_frames




class CommonFilters():
    ''' Class containing several common methods for filtering moseq data
    '''

    @staticmethod
    def duplicate():
        ''' Return a function which returns a tuple which containing `data` and a copy of `data`
        '''
        def _duplicate(data):
            ''' Return a tuple  containing `data` and a copy of `data`'''
            return (data, np.copy(data))
        return _duplicate

    @staticmethod
    def prep_raw_frames(bground_im=None, roi=None, vmin=None, vmax=None):
        ''' Clasic preparation of raw frames, see moseq2_detectron_extract.proc.proc.prep_raw_frames
        '''
        return partial(prep_raw_frames,
                        bground_im=bground_im,
                        roi=roi,
                        vmin=vmin,
                        vmax=vmax)

    @staticmethod
    def depth_encode(crop_bground_im, roi, mode='mda'):
        ''' Return a function which will encode depth frames to the specified mode
        '''
        encoder = HHAEncoder(crop_bground_im, mode=mode)
        def _encode(data):
            data = prep_raw_frames(data, bground_im=None, roi=roi, vmin=None, vmax=None)
            for i in range(data.shape[0]):
                data[i] = encoder.encode(data[i])
        return _encode
