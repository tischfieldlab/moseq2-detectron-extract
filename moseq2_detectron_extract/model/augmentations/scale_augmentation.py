import numpy as np
from detectron2.data.transforms import (Augmentation, BlendTransform,
                                        ResizeTransform, Transform,
                                        TransformList)
from PIL import Image


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
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.target_height = target_height
        self.target_width = target_width
        self.interp = interp
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
        random_scale = self._rand_range(self.min_scale, self.max_scale)
        return self._get_resize(image, random_scale)
