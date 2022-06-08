import inspect
import pprint
from typing import List, Tuple, Union

from albumentations.core.transforms_interface import BasicTransform
from detectron2.data.transforms import Augmentation, Transform
import numpy as np


RangeType = Union[int, float, Tuple[int, int], Tuple[float, float], List[int], List[float]]


def create_circular_mask(height: int, width: int, center: Tuple[int, int]=None, radius: int=None) -> np.ndarray:
    ''' Generate a mask of size `height` and `width` with a positive circle shape with `radius` and position `center`

    Parameters:
    height (int): height of the generated mask
    width (int): width of the generated mask
    center (Tuple[int, int]): center of the circular positive region. If `None`, the circle is centered in the generated mask
    radius (int): radius of the circular positive region. If `None`, the circle has the largest radius which allows the circle to fit inside the generated mask

    Returns:
    np.ndarray: mask of size `height` and `width`, containing a circular positive region centered on `center` and with radius `radius`
    '''

    if center is None: # use the middle of the image
        center = (int(width/2), int(height/2))

    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], width-center[0], height-center[1])

    Y, X = np.ogrid[:height, :width]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask


class AlbumentationsTransform(Transform):
    ''' Transform wrapper for transforms implemented by Albumentations
        see: https://github.com/facebookresearch/detectron2/pull/3306
    '''
    def __init__(self, aug, param):
        self.aug = aug
        self.param = param

    def apply_image(self, img):
        try:
            td_params = self.aug.get_params_dependent_on_targets({"image": img})
        except Exception: # pylint: disable=broad-except
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

    def get_transform(self):
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
                    f"Attribute {name} not found! Default __repr__ only works if attributes match the constructor."
                )
                attr = getattr(self._aug, name)
                default = param.default
                if default is attr:
                    continue
                attr_str = pprint.pformat(attr)
                if "\n" in attr_str:
                    # don't show it if pformat decides to use >1 lines
                    attr_str = "..."
                argstr.append(f"{name}={attr_str}")
            return f'{outer_classname}({classname}({", ".join(argstr)}))'
        except AssertionError:
            return super().__repr__()

    __str__ = __repr__
