import inspect
import pprint
from typing import Any, List, Tuple, Union

from albumentations.core.transforms_interface import BasicTransform
from detectron2.data.transforms import Augmentation, NoOpTransform, Transform

RangeType = Union[int, float, Tuple[int, int], Tuple[float, float], List[int], List[float]]

def validate_range_arg(param_name: str, value: Any) -> Union[Tuple[int, int], Tuple[float, float]]:
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
        and: https://github.com/facebookresearch/detectron2/issues/3054
    '''
    def __init__(self, aug: BasicTransform, params):
        self.aug = aug
        self.params = params

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        try:
            return self.aug.apply_to_keypoints(coords, **self.params)
        except AttributeError:
            return coords

    def apply_image(self, image):
        return self.aug.apply(image, **self.params)

    def apply_box(self, box: np.ndarray) -> np.ndarray:
        try:
            return np.array(self.aug.apply_to_bboxes(box.tolist(), **self.params))
        except AttributeError:
            return box

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        try:
            return self.aug.apply_to_mask(segmentation, **self.params)
        except AttributeError:
            return segmentation


class Albumentations(Augmentation):
    '''
    Wrap an augmentor form the albumentations library: https://github.com/albu/albumentations.
    Image, Bounding Box, keypoints, and Segmentation are supported.
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

    def get_transform(self, image):
        do = self._aug.always_apply or self._rand_range() < self._aug.p
        if do:
            params = self._prepare_param(image)
            return AlbumentationsTransform(self._aug, params)
        else:
            return NoOpTransform()

    def _prepare_param(self, image):
        params = self._aug.get_params()
        if self._aug.targets_as_params:
            targets_as_params = {"image": image}
            params_dependent_on_targets = self._aug.get_params_dependent_on_targets(targets_as_params)
            params.update(params_dependent_on_targets)
        params = self._aug.update_params(params, **{"image": image})
        return params

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
