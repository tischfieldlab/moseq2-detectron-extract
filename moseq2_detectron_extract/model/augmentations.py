from typing import List, Tuple, Union
from detectron2.data.transforms import Augmentation, Transform, BlendTransform, NoOpTransform
import random
import numpy as np

RangeType = Union[int, float, Tuple[int, int], Tuple[float, float], List[int], List[float]]

class DoughnutNoiseAugmentation(Augmentation):
    def __init__(self, mu: float=0, var_limit: RangeType=(10.0, 50.0), thickness: RangeType=(0, 30), weight: float=0.5, always_apply: bool=False, p: float=0.5):
        ''' Apply Doughnut-shaped random noise to an image

        Parameters:
        mu (float): mean of the noise
        var_limit (RangeType): variance range for noise. If var_limit is a single number, the range will be (0, var_limit).
        thickness (RangeType): the thickness of the doughnut ring
        weight (float): Weight of the underlying blend transformation
        always_apply (bool): True to always apply the transform
        p (float): probability of applying the transform.
        '''
        self.mu = mu
        self.thickness = self.validate_range_arg('thickness', thickness)
        self.var_limit = self.validate_range_arg('var_limit', var_limit)
        self.weight = weight
        self.always_apply = always_apply
        self.p_application = p

    def validate_range_arg(self, param_name: str, value):
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
                "Expected var_limit type to be one of (int, float, tuple[int|float], list[int|float]), got {}".format(type(value))
            )

    def get_transform(self, image):
        if (random.random() < self.p_application) or self.always_apply:
            # select random values for some parameters
            thickness = random.uniform(self.thickness[0], self.thickness[1])
            var = random.uniform(self.var_limit[0], self.var_limit[1])
            sigma = var ** 0.5
            random_state = np.random.RandomState(random.randint(0, 2 ** 32 - 1))

            xx, yy = np.mgrid[:image.shape[0], :image.shape[1]]
            r_outer = np.max(image.shape[:2]) // 2
            r_inner = r_outer - thickness
            cx = image.shape[0] // 2
            cy = image.shape[1] // 2
            circle = (xx - cx) ** 2 + (yy - cy) ** 2
            donut = np.logical_and(circle < r_outer ** 2, circle > r_inner ** 2)

            im = np.zeros(shape=image.shape[:2], dtype=float)
            im[donut] = random_state.normal(self.mu, sigma, size=np.count_nonzero(donut))
            if len(im.shape) == 3:
                np.expand_dims(im, -1)
            return BlendTransform(im, self.weight, 1-self.weight)
        else:
            return NoOpTransform()


#https://github.com/facebookresearch/detectron2/pull/3306
class AlbumentationsTransform(Transform):
    def __init__(self, aug, param):
        self.aug = aug
        self.param = param

    def apply_image(self, img):
        try:
            td_params = self.aug.get_params_dependent_on_targets({"image": img})
        except:
            td_params = {}
        return self.aug.apply(img, **self.param, **td_params)

    def apply_coords(self, coords):
        return coords


class Albumentations(Augmentation):
    """
    Wrap an augmentor form the albumentations library: https://github.com/albu/albumentations.
    Coordinate augmentation is not supported by the library.
    Example:
    .. code-block:: python
        import detectron2.data.transforms.external as  A
        import albumentations as AB
        ## Resize 
        #augs1 = A.Albumentations(AB.SmallestMaxSize(max_size=1024, interpolation=1, always_apply=False, p=1))
        #augs1 = A.Albumentations(AB.RandomScale(scale_limit=0.8, interpolation=1, always_apply=False, p=0.5))
        ## Rotate 
        augs1 = A.Albumentations(AB.RandomRotate90(p=1))
        transform_1 = augs1(input)
        image_transformed_1 = input.image
        cv2_imshow(image_transformed_1)
    """

    def __init__(self, augmentor):
        """
        Args:
            augmentor (albumentations.BasicTransform):
        """
        super(Albumentations, self).__init__()
        self._aug = augmentor

    def get_transform(self, img):
        return AlbumentationsTransform(self._aug, self._aug.get_params())
