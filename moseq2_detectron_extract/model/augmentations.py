import inspect
import pprint
import random
from functools import partial
from typing import List, Tuple, Union

import numpy as np
from albumentations.core.transforms_interface import (BasicTransform,
                                                      DualTransform)
from detectron2.data.transforms import (Augmentation, BlendTransform,
                                        CropTransform, FixedSizeCrop,
                                        NoOpTransform, PadTransform,
                                        ResizeTransform, Transform,
                                        TransformList)
from FyeldGenerator import generate_field
from PIL import Image

RangeType = Union[int, float, Tuple[int, int], Tuple[float, float], List[int], List[float]]


class ScaleAugmentation(Augmentation):
    """
    Takes target size as input and randomly scales the given target size between `min_scale`
    and `max_scale`. It then scales the input image such that it fits inside the scaled target
    box, keeping the aspect ratio constant.
    This implements the resize part of the Google's 'resize_and_crop' data augmentation:
    https://github.com/tensorflow/tpu/blob/master/models/official/detection/utils/input_utils.py#L127
    """

    def __init__(
        self,
        min_scale: float,
        max_scale: float,
        target_height: int,
        target_width: int,
        interp: int = Image.BILINEAR,
    ):
        """
        Args:
            min_scale: minimum image scale range.
            max_scale: maximum image scale range.
            target_height: target image height.
            target_width: target image width.
            interp: image interpolation method.
        """
        super().__init__()
        self._init(locals())

    def _get_resize(self, image: np.ndarray, scale: float) -> TransformList:
        input_size = image.shape[:2]

        # Compute new target size given a scale.
        target_size = (self.target_height, self.target_width)
        target_scale_size = np.multiply(target_size, scale)

        # Compute actual rescaling applied to input image and output size.
        output_scale = np.minimum(
            target_scale_size[0] / input_size[0], target_scale_size[1] / input_size[1]
        )
        output_size = np.round(np.multiply(input_size, output_scale)).astype(int)
        # print(output_scale)

        return TransformList([
            ResizeTransform(input_size[0], input_size[1], output_size[0], output_size[1], self.interp),
            BlendTransform(src_image=0, src_weight=1 - output_scale, dst_weight=output_scale),
        ])

    def get_transform(self, image: np.ndarray) -> Transform:
        random_scale = np.random.uniform(self.min_scale, self.max_scale)
        return self._get_resize(image, random_scale)


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
        super().__init__()
        self._init(locals())
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
                f"Expected {param_name} type to be one of (int, float, tuple[int|float], list[int|float]), got {type(value)}"
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
            if len(image.shape) == 3:
                im = np.expand_dims(im, -1)
            return BlendTransform(im, src_weight=1, dst_weight=1)
        else:
            return NoOpTransform()


class RandomFieldNoiseAugmentation(Augmentation):
    def __init__(self, mu: float=0, std_limit: RangeType=(10.0, 50.0), power: RangeType=(1.0, 3.0), weight: float=0.5, always_apply: bool=False, p: float=0.5):
        ''' Apply Gaussian Random Field type noise to an image

        Parameters:
        mu (float): mean of the noise
        std_limit (RangeType): std dev range for noise. If std_limit is a single number, the range will be (0, std_limit).
        power (RangeType): exponent for the power spectrum
        weight (float): Weight of the underlying blend transformation
        always_apply (bool): True to always apply the transform
        p (float): probability of applying the transform.
        '''
        super().__init__()
        self._init(locals())
        self.mu = mu
        self.std_limit = self.validate_range_arg('std_limit', std_limit)
        self.power = self.validate_range_arg('power', power)
        self.weight = weight
        self.always_apply = always_apply
        self.p_application = p
        self.eps = np.finfo(np.float64).eps

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
                f"Expected {param_name} type to be one of (int, float, tuple[int|float], list[int|float]), got {type(value)}"
            )

    def Pkgen(self, n):
        # Helper that generates power-law power spectrum
        def Pk(k):
            return np.power(k + self.eps, -n)
        return Pk

    def distrib(self, shape, mu=0.0, scale=1.0):
        # Draw samples from a normal distribution
        a = np.random.normal(loc=mu, scale=scale, size=shape)
        b = np.random.normal(loc=mu, scale=scale, size=shape)
        return a + 1j * b

    def get_field(self, shape):
        # select random values for some parameters
        dist = partial(self.distrib, mu=self.mu, scale=random.uniform(self.std_limit[0], self.std_limit[1]))
        power = self.Pkgen(random.uniform(self.power[0], self.power[1]))

        field = generate_field(dist, power, shape)

        # seems that field shape can be off by one in axis 1, so resize without warping (padding)
        if field.shape != shape:
            f2 = np.zeros(shape)
            f2[0:field.shape[0], 0:field.shape[1]] = field
            field = f2

        return field

    def get_transform(self, image):
        if (random.random() < self.p_application) or self.always_apply:
            field = self.get_field(image.shape[:2])
            field = np.abs(field)

            if len(image.shape) == 3:
                field = np.expand_dims(field, -1)

            return BlendTransform(src_image=field, src_weight=1, dst_weight=1)
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
    '''
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
    '''

    def __init__(self, augmentor: BasicTransform):
        '''
        Args:
            augmentor (albumentations.BasicTransform):
        '''
        super().__init__()
        self._aug = augmentor

    def get_transform(self, img):
        return AlbumentationsTransform(self._aug, self._aug.get_params())

    def __repr__(self):
        """
        Produce something like:
        "MyAugmentation(field1={self.field1}, field2={self.field2})"
        """
        try:
            sig = inspect.signature(self._aug.__init__)
            outer_classname = type(self).__name__
            classname = type(self._aug).__name__
            argstr = []
            for name, param in sig.parameters.items():
                assert (
                    param.kind != param.VAR_POSITIONAL and param.kind != param.VAR_KEYWORD
                ), "The default __repr__ doesn't support *args or **kwargs"
                assert hasattr(self._aug, name), (
                    "Attribute {} not found! "
                    "Default __repr__ only works if attributes match the constructor.".format(name)
                )
                attr = getattr(self._aug, name)
                default = param.default
                if default is attr:
                    continue
                attr_str = pprint.pformat(attr)
                if "\n" in attr_str:
                    # don't show it if pformat decides to use >1 lines
                    attr_str = "..."
                argstr.append("{}={}".format(name, attr_str))
            return "{}({}({}))".format(outer_classname, classname, ", ".join(argstr))
        except AssertionError:
            return super().__repr__()

    __str__ = __repr__
