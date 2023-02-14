from typing import Tuple

import elasticdeform
import numpy as np
import numpy.typing as npt
from detectron2.data.transforms import NoOpTransform, BlendTransform
from moseq2_detectron_extract.model.augmentations.occlude_transform import MaxBlendTransform
from moseq2_detectron_extract.model.augmentations.random_field_noise import \
    RandomFieldNoiseAugmentation
from moseq2_detectron_extract.model.augmentations.util import (
    RangeType, create_circular_mask, validate_range_arg)


class ParticleNoiseAugmentation(RandomFieldNoiseAugmentation):
    ''' Augmentation policy for generating particle noise
    '''
    def __init__(self, mu: float=0.0, std_limit: RangeType=(75.0, 100.0), power: RangeType=(2.5, 4.0), radius: RangeType=(3, 20),
                 points: RangeType=(3, 20), n_particles: RangeType=(1,4), intensity_max: RangeType=(30., 250.0),
                 always_apply: bool=False, p: float=0.5):
        ''' Apply Gaussian Random Field type noise to an image

        Parameters:
        mu (float): mean of the noise
        std_limit (RangeType): std dev range for noise. If std_limit is a single number, the range will be (0, std_limit).
        power (RangeType): exponent for the power spectrum
        radius (RangeType): radius of the produced particles (prior to elastic deformation)
        points (RangeType): number of points to use during elastic deformation
        intensity_max (RangeType): rescale the intensity of generate particles to be less than this value
        always_apply (bool): True to always apply the transform
        p (float): probability of applying the transform.
        '''
        super().__init__(mu=mu, std_limit=std_limit, power=power, intensity_max=intensity_max, always_apply=always_apply, p=p)
        self._init(locals())

        # params for particle generation
        self.radius = validate_range_arg('radius', radius)
        self.points = validate_range_arg('points', points)
        self.n_particles = validate_range_arg('n_particles', n_particles)


    def generate_particle(self, size: Tuple[int, int]=(512, 512), dtype: npt.DTypeLike='uint8'):
        ''' Generate a particle '''
        radius = self._rand_range(*self.radius)
        points = int(self._rand_range(*self.points))
        center = (int(self._rand_range(size[0])), int(self._rand_range(0, size[1])))

        particle = self.get_field(size)

        mask = create_circular_mask(*size, center=center, radius=radius)
        particle[~mask] = 0
        particle = elasticdeform.deform_random_grid(particle, sigma=radius // 2, points=points)
        particle = np.abs(particle)

        particle = np.abs(particle)
        particle = self.rescale_intensity(particle, vmin=0, vmax=int(self._rand_range(*self.intensity_max)))
        particle = particle.astype(dtype)

        return particle

    def get_transform(self, image: np.ndarray):
        ''' Get the transform
        '''
        if (self._rand_range() < self.p_application) or self.always_apply:
            n_particles = int(self._rand_range(*self.n_particles))
            field = np.zeros(image.shape[:2], dtype=image.dtype)
            for _ in range(n_particles):
                field += self.generate_particle(size=(image.shape[0], image.shape[1]), dtype=image.dtype)

            if len(image.shape) == 3:
                field = np.expand_dims(field, -1)

            #return MaxBlendTransform(src_image=field)
            return BlendTransform(src_image=field, src_weight=1, dst_weight=1)
        else:
            return NoOpTransform()
