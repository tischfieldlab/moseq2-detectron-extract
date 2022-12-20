from abc import ABCMeta

import numpy as np
from detectron2.data import transforms as T


class BaseImageBlendTransform(T.Transform, metaclass=ABCMeta):
    """Base class for a blending transform for images"""
    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        """
        Apply no transform on the coordinates.
        """
        return coords

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        """
        Apply no transform on the full-image segmentation.
        """
        return segmentation

    def inverse(self) -> T.Transform:
        """
        The inverse is a no-op.
        """
        return T.NoOpTransform()


class MaxBlendTransform(BaseImageBlendTransform):
    """
    Transforms pixel values taking the max.
    """
    def __init__(self, src_image: np.ndarray):
        """
        Blends the input image (dst_image) with the src_image using formula:
        ``src_weight * src_image + dst_weight * dst_image``

        Args:
            src_image (ndarray): Input image is blended with this image.
                The two images must have the same shape, range, channel order
                and dtype.
        """
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        """
        Apply blend transform on the image(s).

        Args:
            img (ndarray): of shape NxHxWxC, or HxWxC or HxW. The array can be
                of type uint8 in range [0, 255], or floating point in range
                [0, 1] or [0, 255].
            interp (str): keep this option for consistency, perform blend would not
                require interpolation.
        Returns:
            ndarray: blended image(s).
        """
        return np.where(img > self.src_image, img, self.src_image)


class ThresholdBlendTransform(BaseImageBlendTransform):
    """
    Transforms pixel values.
    """
    def __init__(self, src_image: np.ndarray, threshold: float):
        """
        Blends the input image (dst_image) with the src_image using formula:
        ``src_weight * src_image + dst_weight * dst_image``

        Args:
            src_image (ndarray): Input image is blended with this image.
                The two images must have the same shape, range, channel order
                and dtype.
        """
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        """
        Apply a thresholded blend between two images

        Args:
            img (ndarray): of shape NxHxWxC, or HxWxC or HxW. The array can be
                of type uint8 in range [0, 255], or floating point in range
                [0, 1] or [0, 255].
            interp (str): keep this option for consistency, perform blend would not
                require interpolation.
        Returns:
            ndarray: blended image(s).
        """
        return np.where(img > self.threshold, img, self.src_image)
