import matplotlib.pyplot as plt
import random
from moseq2_detectron_extract.io.image import read_image
from detectron2.utils.visualizer import Visualizer, ColorMode
import cv2

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
