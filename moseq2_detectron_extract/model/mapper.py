import copy

import detectron2.data.detection_utils as utils
import detectron2.data.transforms as T
import numpy as np
import torch
from detectron2.data.dataset_mapper import DatasetMapper
from moseq2_detectron_extract.io.image import read_image
from moseq2_detectron_extract.proc.proc import colorize_video


class MoseqDatasetMapper(DatasetMapper):
    ''' Custom dataset mapper for moseq data
    '''


    def __call__(self, dataset_dict: dict):
        '''
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        '''
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below

        # USER: Write your own image loading if it's not from a file
        image = read_image(dataset_dict["file_name"], dtype='uint8')
        # image = image[:,:,0] # grayscale, first channel only, but keep the dimention
        utils.check_image_size(dataset_dict, image)


        aug_input = T.AugInput(image)
        transforms = self.augmentations(aug_input)
        image = aug_input.image

        if len(image.shape) == 2:
            # seems the augmentations can cause the last axis to drop
            # when only a single channel. Lets pop the data into a third dimention!
            image = image[:,:,None]

        #print(image.shape)
        image = np.expand_dims(image[:, :, 0], axis=0)
        #print(image.shape)
        image = colorize_video(image, vmin=0, vmax=255, cmap='jet')[0]
        #print(image.shape)

        image_shape = image.shape[:2]  # h, w
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.use_instance_mask:
                    anno.pop("segmentation", None)
                if not self.use_keypoint:
                    anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(
                    obj, transforms, image_shape, keypoint_hflip_indices=self.keypoint_hflip_indices
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(
                annos, image_shape, mask_format=self.instance_mask_format
            )

            # After transforms such as cropping are applied, the bounding box may no longer
            # tightly bound the object. As an example, imagine a triangle object
            # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
            # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
            # the intersection of original bounding box and the cropping box.
            if self.recompute_boxes:
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict["instances"] = utils.filter_empty_instances(instances)
        return dataset_dict
