
import random

import numpy as np
from detectron2.data.transforms import (Augmentation, BlendTransform,
                                        NoOpTransform)
from moseq2_detectron_extract.model.augmentations.util import RangeType, create_doughnut_mask, validate_range_arg


class DoughnutNoiseAugmentation(Augmentation):
    ''' Augmentation to add doughnut shaped noise to an image
    '''
    def __init__(self, mu: float=0, var_limit: RangeType=(10.0, 50.0), thickness: RangeType=(0, 30), weight: float=0.5,
                 always_apply: bool=False, p: float=0.5):
        ''' Apply Doughnut-shaped random noise to an image

        Parameters:
        mu (float): mean of the noise
        var_limit (RangeType): variance range for noise. If var_limit is a single number, the range will be (0, var_limit).
        thickness (RangeType): the thickness of the doughnut ring
        weight (float): Weight of the underlying blend transformation
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

    def validate_range_arg(self, param_name: str, value):
        ''' validate user supplied range arguments
        '''
        if isinstance(value, (tuple, list)):
            if value[0] < 0:
                raise ValueError(f"Lower {param_name} should be non negative.")
            if value[1] < 0:
                raise ValueError(f"Upper {param_name} should be non negative.")
            return value
        elif isinstance(value, (int, float)):
            if value < 0:
                raise ValueError(f"{param_name} should be non negative.")

            return (0, value)
        else:
            raise TypeError(
                f"Expected {param_name} type to be one of (int, float, tuple[int|float], list[int|float]), got {type(value)}"
            )

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
