import random

import cv2
import matplotlib.pyplot as plt
from detectron2.data.catalog import MetadataCatalog
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.utils.visualizer import ColorMode, Visualizer
import numpy as np
import skimage
from skimage.util.dtype import img_as_bool


def visualize_annotations(annotations, metadata, num=5):
    fig, axs = plt.subplots(1, num, figsize=(20*num,20))
    for d, ax in zip(random.sample(annotations, num), axs):
        im = cv2.imread(d["file_name"])
        v = Visualizer(im[:, :, ::-1],
                    metadata=metadata, 
                    scale=2, 
                    instance_mode=ColorMode.IMAGE   # remove the colors of unsegmented pixels. This option is only available for segmentation models
        )
        out = v.draw_dataset_dict(d)
        ax.imshow(out.get_image())
    return fig, axs

def visualize_inference(frame, instances, min_height, max_height, scale=2.0):
    im = frame[:,:,None].copy().astype('uint8')
    im = (im-min_height)/(max_height-min_height)
    im[im < min_height] = 0
    im[im > max_height] = max_height
    im = im * 255
    return draw_instances(im, instances, scale=scale)


def draw_instances(frame, instances, scale=2.0):
    v = Visualizer(
            convert_image_to_rgb(frame, "L"),
            metadata=MetadataCatalog.get("moseq_train"),
            scale=scale,
            instance_mode=ColorMode.SEGMENTATION   # remove the colors of unsegmented pixels. This option is only available for segmentation models
    )
    out = v.draw_instance_predictions(instances)
    return out.get_image()

def draw_instances_fast(frame, instances, dataset_name='moseq_train', scale=2.0):
    im = convert_image_to_rgb(frame, "L")
    im = cv2.resize(im, (int(im.shape[1] * scale), int(im.shape[0] * scale)))

    for i in range(len(instances)):
        # draw mask
        mask = instances.pred_masks[i].cpu().numpy()
        mask = img_as_bool(skimage.transform.resize(mask, (im.shape[0], im.shape[1])))
        mask_overlay = np.zeros_like(im)
        mask_overlay[mask] = np.array([255, 0, 0])
        mask_alpha = 0.3
        im = cv2.addWeighted(mask_overlay, mask_alpha, im, 1-mask_alpha, 0, im)

        # draw box
        box = instances.pred_boxes.tensor.to('cpu').numpy()[i]
        box *= scale
        im = cv2.rectangle(im, tuple(box[0:2].astype(int)), tuple(box[2:4].astype(int)), (0,255,0))


        # keypoints
        kpts = instances.pred_keypoints[i, :, :].cpu().numpy()
        kpts *= scale
        
        # draw keypoint connections
        kn = MetadataCatalog.get(dataset_name).keypoint_names
        kcr = MetadataCatalog.get(dataset_name).keypoint_connection_rules
        for rule in kcr:
            ki1 = kn.index(rule[0])
            ki2 = kn.index(rule[1])
            cv2.line(im, tuple(kpts[ki1, :2].astype(int)), tuple(kpts[ki2, :2].astype(int)), rule[2], 2)
        
        # draw keypoints
        for ki in range(kpts.shape[0]):
            im = cv2.circle(im, tuple(kpts[ki, :2].astype(int)), 3, (0,0,255), -1)

        
        

    return im

