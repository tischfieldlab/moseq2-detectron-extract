from typing import Tuple

import elasticdeform
import numpy as np
import numpy.typing as npt
from detectron2.data.transforms import BlendTransform, NoOpTransform
from moseq2_detectron_extract.model.augmentations.random_field_noise import \
    RandomFieldNoiseAugmentation
from moseq2_detectron_extract.model.augmentations.util import (
    RangeType, create_circular_mask, validate_range_arg)


class ParticleNoiseAugmentation(RandomFieldNoiseAugmentation):
    ''' Augmentation policy for generating particle noise
    '''
    def __init__(self, mu: float=0.0, std_limit: RangeType=(75.0, 100.0), power: RangeType=(2.5, 4.0), radius: RangeType=(3, 20),
                 points: RangeType=(3, 20), n_particles: RangeType=(1,4), intensity_max: RangeType=(30., 100.0), weight: float=1.0,
                 always_apply: bool=False, p: float=0.5):
        ''' Apply Gaussian Random Field type noise to an image

        Parameters:
        mu (float): mean of the noise
        std_limit (RangeType): std dev range for noise. If std_limit is a single number, the range will be (0, std_limit).
        power (RangeType): exponent for the power spectrum
        radius (RangeType): radius of the produced particles (prior to elastic deformation)
        points (RangeType): number of points to use during elastic deformation
        intensity_max (RangeType): rescale the intensity of generate particles to be less than this value
        weight (float): Weight of the underlying blend transformation
        always_apply (bool): True to always apply the transform
        p (float): probability of applying the transform.
        '''
        super().__init__(mu=mu, std_limit=std_limit, power=power, weight=weight, always_apply=always_apply, p=p)
        self._init(locals())

        # params for particle generation
        self.radius = validate_range_arg('radius', radius)
        self.points = validate_range_arg('points', points)
        self.n_particles = validate_range_arg('n_particles', n_particles)
        self.intensity_max = validate_range_arg('intensity_max', intensity_max)


    def generate_particle(self, size: Tuple[int, int]=(512, 512), dtype: npt.DTypeLike='uint8'):
        ''' Generate a particle '''
        radius = self._rand_range(*self.radius)
        points = int(self._rand_range(*self.points))
        center = (int(self._rand_range(size[0])), int(self._rand_range(0, size[1])))
        intensity_max = int(self._rand_range(*self.intensity_max))

        particle = self.get_field(size)

        mask = create_circular_mask(*size, center=center, radius=radius)
        particle[~mask] = 0
        particle = elasticdeform.deform_random_grid(particle, sigma=radius // 2, points=points)
        particle = np.abs(particle)

        # rescale intensity
        vmin = particle.min()
        vmax = particle.max()
        dmin = 0
        dmax = intensity_max
        particle = ((particle - vmin) * ((dmax - dmin) / (vmax - vmin)) + dmin).astype(dtype)

        return particle

    def get_transform(self, image: np.ndarray=None):
        ''' Get the transform
        '''
        if (self._rand_range() < self.p_application) or self.always_apply:
            n_particles = int(self._rand_range(*self.n_particles))
            field = np.zeros(image.shape[:2], dtype=image.dtype)
            for _ in range(n_particles):
                field += self.generate_particle(size=image.shape[:2], dtype=image.dtype)

            if len(image.shape) == 3:
                field = np.expand_dims(field, -1)

            return BlendTransform(src_image=field, src_weight=1, dst_weight=1)
        else:
            return NoOpTransform()
