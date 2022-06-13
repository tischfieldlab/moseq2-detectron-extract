
import random
from typing import Tuple

import numpy as np
import numpy.typing as npt

from detectron2.data.transforms import (Augmentation, BlendTransform,
                                        NoOpTransform)
from moseq2_detectron_extract.model.augmentations.random_field_noise import RandomFieldNoiseAugmentation
from moseq2_detectron_extract.model.augmentations.util import RangeType, create_doughnut_mask, validate_range_arg


class DoughnutWhiteNoiseAugmentation(Augmentation):
    ''' Augmentation to add doughnut shaped noise to an image
    '''
    def __init__(self, mu: float=0, var_limit: RangeType=(10.0, 50.0), thickness: RangeType=(0, 30),
                 always_apply: bool=False, p: float=0.5):
        ''' Apply Doughnut-shaped random noise to an image

        Parameters:
        mu (float): mean of the noise
        var_limit (RangeType): variance range for noise. If var_limit is a single number, the range will be (0, var_limit).
        thickness (RangeType): the thickness of the doughnut ring
        always_apply (bool): True to always apply the transform
        p (float): probability of applying the transform.
        '''
        super().__init__()
        self._init(locals())
        self.mu = mu
        self.thickness = validate_range_arg('thickness', thickness)
        self.var_limit = validate_range_arg('var_limit', var_limit)
        self.always_apply = always_apply
        self.p_application = p


    def get_transform(self, image: np.ndarray=None):
        ''' Get the transform
        '''
        if (self._rand_range() < self.p_application) or self.always_apply:
            # select random values for some parameters
            thickness = self._rand_range(*self.thickness)
            var = self._rand_range(*self.var_limit)
            sigma = var ** 0.5
            random_state = np.random.RandomState(random.randint(0, 2 ** 32 - 1))

            donut = create_doughnut_mask(*image.shape[:2], thickness=thickness)

            im = np.zeros(shape=image.shape[:2], dtype=float)
            im[donut] = random_state.normal(self.mu, sigma, size=np.count_nonzero(donut))
            if len(image.shape) == 3:
                im = np.expand_dims(im, -1)
            return BlendTransform(im, src_weight=1, dst_weight=1)
        else:
            return NoOpTransform()


class DoughnutGRFNoiseAugmentation(RandomFieldNoiseAugmentation):
    ''' Augmentation to add doughnut shaped noise to an image
    '''

    def __init__(self, mu: float=0.0, std_limit: RangeType=(75.0, 100.0), power: RangeType=(1.5, 2.5), thickness: RangeType=(0, 30),
                 intensity_max: RangeType=(30., 100.0), always_apply: bool=False, p: float=0.5):
        ''' Apply Doughnut-shaped random noise to an image

        Parameters:
        mu (float): mean of the noise
        std_limit (RangeType): std dev range for noise. If std_limit is a single number, the range will be (0, std_limit).
        power (RangeType): exponent for the power spectrum
        thickness (RangeType): the thickness of the doughnut ring
        intensity_max (RangeType): rescale the intensity of generate particles to be less than this value
        always_apply (bool): True to always apply the transform
        p (float): probability of applying the transform.
        '''
        super().__init__(mu=mu, std_limit=std_limit, power=power, intensity_max=intensity_max, always_apply=always_apply, p=p)
        self._init(locals())

        self.thickness = validate_range_arg('thickness', thickness)


    def generate_doughnut_noise(self, size: Tuple[int, int]=(512, 512), dtype: npt.DTypeLike='uint8'):
        # select random values for some parameters
        thickness = self._rand_range(*self.thickness)
        intensity_max = int(self._rand_range(*self.intensity_max))

        donut = create_doughnut_mask(*size, thickness=thickness)
        field = self.get_field(size)
        field[~donut] = 0
        field = np.abs(field)

        # rescale intensity
        self.rescale_intensity(field, vmax=intensity_max)

        return field


    def get_transform(self, image: np.ndarray=None):
        ''' Get the transform
        '''
        if (self._rand_range() < self.p_application) or self.always_apply:
            noise_field = self.generate_doughnut_noise(size=image.shape[:2], dtype=image.dtype)

            if len(image.shape) == 3:
                noise_field = np.expand_dims(noise_field, -1)

            return BlendTransform(noise_field, src_weight=1, dst_weight=1)
        else:
            return NoOpTransform()
