import random
from typing import Tuple

import elasticdeform
import numpy as np
from detectron2.data.transforms import BlendTransform, NoOpTransform
from moseq2_detectron_extract.model.augmentations.random_field_noise import \
    RandomFieldNoiseAugmentation
from moseq2_detectron_extract.model.augmentations.util import (
    RangeType, create_circular_mask)


class ParticleNoiseAugmentation(RandomFieldNoiseAugmentation):
    def __init__(self, mu: float=0, std_limit: RangeType=(10.0, 50.0), power: RangeType=(1.0, 3.0), radius: RangeType=(5, 50), points: RangeType=(3, 20), weight: float=0.5,
                 always_apply: bool=False, p: float=0.5):
        ''' Apply Gaussian Random Field type noise to an image

        Parameters:
        mu (float): mean of the noise
        std_limit (RangeType): std dev range for noise. If std_limit is a single number, the range will be (0, std_limit).
        power (RangeType): exponent for the power spectrum
        radius (RangeType): radius of the produced particles (prior to elastic deformation)
        points (RangeType): number of points to use during elastic deformation
        weight (float): Weight of the underlying blend transformation
        always_apply (bool): True to always apply the transform
        p (float): probability of applying the transform.
        '''
        super().__init__(mu=mu, std_limit=std_limit, power=power, weight=weight, always_apply=always_apply, p=p)
        self._init(locals())

        # params for particle generation
        self.radius = self.validate_range_arg('radius', radius)
        self.points = self.validate_range_arg('points', points)
        self.weight = weight
        self.always_apply = always_apply
        self.p_application = p
        self.eps = np.finfo(np.float64).eps


    def generate_particle(self, size: Tuple[int, int]=(512, 512)):

        radius = random.uniform(self.radius[0], self.radius[1])
        points = int(random.uniform(self.points[0], self.points[1]))
        center = (int(random.uniform(0, size[0])), int(random.uniform(0, size[1])))

        gen = RandomFieldNoiseAugmentation(mu=100, std_limit=(100.0, 100.0), power=(3.0, 3.0), always_apply=True)
        particle = gen.get_field(size)

        mask = create_circular_mask(*size, center=center, radius=radius)
        particle[~mask] = 0
        particle = elasticdeform.deform_random_grid(particle, sigma=radius // 2, points=points)
        particle[particle < 0] *= -1.0

        return particle

    def get_transform(self, image: np.ndarray=None):
        ''' Get the transform
        '''
        if (random.random() < self.p_application) or self.always_apply:
            field = self.get_field(image.shape[:2])
            field = np.abs(field)

            if len(image.shape) == 3:
                field = np.expand_dims(field, -1)

            return BlendTransform(src_image=field, src_weight=1, dst_weight=1)
        else:
            return NoOpTransform()
