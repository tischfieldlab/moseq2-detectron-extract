from detectron2.data.transforms import Augmentation, Transform
from albumentations.augmentations.transforms import GaussNoise

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
