import random

import cv2
import matplotlib.pyplot as plt
from detectron2.data.catalog import MetadataCatalog
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.utils.visualizer import ColorMode, Visualizer


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
