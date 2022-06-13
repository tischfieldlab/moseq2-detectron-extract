from functools import partial
from typing import Tuple

import numpy as np
import numpy.typing as npt
from detectron2.data.transforms import (Augmentation, BlendTransform,
                                        NoOpTransform)
from FyeldGenerator import generate_field
from moseq2_detectron_extract.model.augmentations.occlude_transform import MaxBlendTransform, ThresholdBlendTransform
from moseq2_detectron_extract.model.augmentations.util import RangeType, validate_range_arg


class RandomFieldNoiseAugmentation(Augmentation):
    ''' Augmentation to apply Gaussian Random Field type noise to an image
    '''
    def __init__(self, mu: float=0, std_limit: RangeType=(5.0, 100.0), power: RangeType=(1.0, 4.0), intensity_max: RangeType=(5.0, 65.0),
                 always_apply: bool=False, p: float=0.5):
        ''' Apply Gaussian Random Field type noise to an image

        Parameters:
        mu (float): mean of the noise
        std_limit (RangeType): std dev range for noise. If std_limit is a single number, the range will be (0, std_limit).
        power (RangeType): exponent for the power spectrum
        intensity_max (RangeType): rescale the intensity of generate particles to be less than this value
        always_apply (bool): True to always apply the transform
        p (float): probability of applying the transform.
        '''
        super().__init__()
        self._init(locals())
        self.mu = mu
        self.std_limit = validate_range_arg('std_limit', std_limit)
        self.power = validate_range_arg('power', power)
        self.intensity_max = validate_range_arg('intensity_max', intensity_max)
        self.always_apply = always_apply
        self.p_application = p
        self.eps = np.finfo(np.float64).eps

    def pkgen(self, n: float):
        ''' Helper that generates power-law power spectrum
        '''
        def pk(k):
            return np.power(k + self.eps, -n)
        return pk

    def distrib(self, shape, mu=0.0, scale=1.0) -> complex:
        ''' Draw samples from a normal distribution
        '''
        a = np.random.normal(loc=mu, scale=scale, size=shape)
        b = np.random.normal(loc=mu, scale=scale, size=shape)
        return a + 1j * b

    def get_field(self, shape: Tuple[int, int]=(512, 512)) -> np.ndarray:
        ''' Get the gaussian random field
        '''
        # select random values for some parameters
        dist = partial(self.distrib, mu=self.mu, scale=self._rand_range(*self.std_limit))
        power = self.pkgen(self._rand_range(*self.power))

        field = generate_field(dist, power, shape)

        # seems that field shape can be off by one in axis 1, so resize without warping (padding)
        if field.shape != shape:
            field2 = np.zeros(shape)
            field2[0:field.shape[0], 0:field.shape[1]] = field
            field = field2

        return field

    def rescale_intensity(self, image: np.ndarray, vmin: float=0, vmax: float=255) -> np.ndarray:
        ''' Rescale image intensity by linear stretching to `vmin` and `vmax`

        Parameters:
        image (np.ndarray): image data to rescale intensity
        vmin (float): minimum value of output data
        vmax (flaot): maximum value of output data

        Returns:
        np.ndarray: image data rescaled to `vmin` and `vmax`
        '''
        dtype = image.dtype
        dmin = image.min()
        dmax = image.max()
        return ((image - dmin) * ((vmax - vmin) / (dmax - dmin)) + vmin).astype(dtype)

    def get_transform(self, image: np.ndarray=None):
        ''' Get the transform
        '''
        if (self._rand_range() < self.p_application) or self.always_apply:
            field = self.get_field(shape=image.shape[:2])

            field = np.abs(field)
            field = self.rescale_intensity(field, vmin=0, vmax=int(self._rand_range(*self.intensity_max)))

            field = field.astype(image.dtype)

            if len(image.shape) == 3:
                field = np.expand_dims(field, -1)

            #return BlendTransform(src_image=field, src_weight=1, dst_weight=1)
            #return MaxBlendTransform(src_image=field)
            return ThresholdBlendTransform(src_image=field, threshold=10)
        else:
            return NoOpTransform()
