import random
from typing import Iterable, Sequence, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage
from detectron2.data.catalog import MetadataCatalog
from detectron2.data.detection_utils import convert_image_to_rgb, read_image
from detectron2.structures import Instances
from detectron2.utils.visualizer import ColorMode, Visualizer
from skimage.util.dtype import img_as_bool

from moseq2_detectron_extract.io.annot import KeypointConnections


def visualize_annotations(annotations, metadata, num=5):
    fig, axs = plt.subplots(1, num, figsize=(20*num,20))
    for d, ax in zip(random.sample(annotations, num), axs):
        scale_factor = d["rescale_intensity"] if "rescale_intensity" in d else None
        im = read_image(d["file_name"], scale_factor=scale_factor, dtype='uint8')
        v = Visualizer(im[:, :, ::-1],
                    metadata=metadata,
                    scale=2,
                    instance_mode=ColorMode.IMAGE   # remove the colors of unsegmented pixels. This option is only available for segmentation models
        )
        out = v.draw_dataset_dict(d)
        ax.imshow(out.get_image())
    return fig, axs


def visualize_inference(frame: np.ndarray, instances: Instances, min_height: float, max_height: float, scale: float=2.0) -> np.ndarray:
    im = frame[:,:,None].copy().astype('uint8')
    im = (im-min_height)/(max_height-min_height)
    im[im < min_height] = 0
    im[im > max_height] = max_height
    im = im * 255
    return draw_instances(im, instances, scale=scale)


def draw_instances(frame: np.ndarray, instances: Instances, scale: float=2.0) -> np.ndarray:
    ''' Draw instances using Detectron2 Visualizer class. This is slow, so for speed, use `draw_instances_fast()`
    '''
    v = Visualizer(
            convert_image_to_rgb(frame, "L"),
            metadata=MetadataCatalog.get("moseq_train"),
            scale=scale,
            instance_mode=ColorMode.SEGMENTATION   # remove the colors of unsegmented pixels. This option is only available for segmentation models
    )
    out = v.draw_instance_predictions(instances)
    return out.get_image()


def scale_depth_frames(frames: np.ndarray, scale: float=2.0) -> np.ndarray:
    ''' Scale/resize single channel (grayscale) frames

    Parameters:
    frames (np.ndarray): frames to scale, of shape (height, width) or (nframes, height, width)
    scale (float): scale factor to resize image by

    Returns:
    np.ndarray, scaled/resized image
    '''
    if len(frames.shape) == 2:
        # single frame
        return cv2.resize(frames, (int(frames.shape[1] * scale), int(frames.shape[0] * scale)))

    else:
        #batch of frames
        width, height = (int(frames.shape[2] * scale), int(frames.shape[1] * scale))
        out = np.zeros_like(frames, shape=(frames.shape[0], width, height))
        for i in range(frames.shape[0]):
            out[i] = cv2.resize(frames[i], (width, height))
        return out


def scale_color_frames(frames: np.ndarray, scale: float=2.0) -> np.ndarray:
    ''' Scale/resize color (RGB) frames

    Parameters:
    frames (np.ndarray): frames to scale, of shape (height, width, 3) or (nframes, height, width, 3)
    scale (float): scale factor to resize image by

    Returns:
    np.ndarray, scaled/resized image
    '''
    if len(frames.shape) == 3:
        # single frame
        return cv2.resize(frames, (int(frames.shape[1] * scale), int(frames.shape[0] * scale)))

    else:
        #batch of frames
        width, height = (int(frames.shape[2] * scale), int(frames.shape[1] * scale))
        out = np.zeros_like(frames, shape=(frames.shape[0], width, height, 3))
        for i in range(frames.shape[0]):
            out[i] = cv2.resize(frames[i], (width, height))
        return out


def draw_instances_fast(frame: np.ndarray, instances: Instances, keypoint_names: Sequence[str],
    keypoint_connection_rules: KeypointConnections, roi_contour: Iterable[np.ndarray]=None, scale: float=2.0) -> np.ndarray:
    ''' Draw `instances` on `frame`

    Parameters:
    frame (np.ndarray): grayscale frame to draw instances on, of shape (height, width)
    instances (Instances): Detectron2 instances
    keypoint_names (Sequence[str]): names of the keypoints to draw
    keypoint_connection_rules (KeypointConnections): rules describing how keypoints are connected
    roi_contour (Iterable[np.ndarray]): Contours of the roi, found via `cv2.findContours()`
    scale (float): size scale factor to scale frame and instances by

    Returns:
    np.ndarray, frame with instances drawn
    '''
    im = convert_image_to_rgb(frame, "L")

    if roi_contour is not None:
        im = draw_contour(im, roi_contour)

    im = cv2.resize(im, (int(im.shape[1] * scale), int(im.shape[0] * scale)))

    for i in range(len(instances)):
        # draw mask
        mask = instances.pred_masks[i].cpu().numpy()
        im = draw_mask(im, mask)

        # draw box
        box = instances.pred_boxes.tensor.to('cpu').numpy()[i]
        box *= scale
        im = cv2.rectangle(im, tuple(box[0:2].astype(int)), tuple(box[2:4].astype(int)), (0,255,0))

        # keypoints
        kpts = instances.pred_keypoints[i, :, :].cpu().numpy()
        im = draw_keypoints(im, kpts, keypoint_names, keypoint_connection_rules, scale=scale)

    return im


def draw_mask(im: np.ndarray, mask: np.ndarray, alpha: float=0.3, color: Tuple[int, int, int]=(255,0,0)) -> np.ndarray:
    ''' Draw a mask on an image

    Parameters:
    im (np.ndarray): image to draw mask upon, of shape (height, width, 3)
    mask (np.ndarray): mask to draw
    alpha (float): alpha leve to use when drawing mask. 0=Transparent, 1=Opaque
    color (Tuple[int, int, int]): color to draw the mask, tuple of ints in range [0-255] in RGB order

    Returns:
    np.ndarray, image with mask drawn
    '''
    mask = img_as_bool(skimage.transform.resize(mask, (im.shape[0], im.shape[1]), preserve_range=True))
    mask_overlay = np.zeros_like(im)
    mask_overlay[mask] = np.array(color)
    im = cv2.addWeighted(mask_overlay, alpha, im, 1-alpha, 0, im)
    return im


def draw_keypoints(im: np.ndarray, keypoints: np.ndarray, keypoint_names: Sequence[str], keypoint_connection_rules: KeypointConnections,
    origin: Tuple[int, int]=(0,0), scale: float=1.0) -> np.ndarray:
    ''' Draw keypoints on an image

    Parameters:
    im (np.ndarray): image to draw keypoints upon, of shape (height, width, 3)
    keypoints (np.ndarray): Keypoints to draw
    keypoint_names (Sequence[str]): names of the keypoints to draw
    keypoint_connection_rules (KeypointConnections): rules describing how keypoints are connected
    origin (Tuple[int, int]): origin of the coordinate system keypoints are in
    scale (float): size scale factor to scale keypoints by

    Returns:
    np.ndarray, image with keypoints drawn
    '''
    keypoints = np.copy(keypoints) * scale

    # draw keypoint connections
    for rule in keypoint_connection_rules:
        ki1 = keypoint_names.index(rule[0])
        ki2 = keypoint_names.index(rule[1])
        x1, y1 = keypoints[ki1, :2].astype(int)
        x2, y2 = keypoints[ki2, :2].astype(int)
        cv2.line(im, (x1 + origin[0], y1 + origin[1]), (x2 + origin[0], y2 + origin[1]), rule[2], 2, cv2.LINE_AA)

    # draw keypoints
    for ki in range(keypoints.shape[0]):
        x = keypoints[ki, 0].astype(int) + origin[0]
        y = keypoints[ki, 1].astype(int) + origin[1]
        im = cv2.circle(im, (x,y), 3, (0,0,255), -1, cv2.LINE_AA)

    return im


def draw_contour(im: np.ndarray, contour: Iterable[np.ndarray], color: Tuple[int, int, int]=(0,255,0), thickness: float=1) -> np.ndarray:
    ''' Draw contours on a image

    Parameters:
    im (np.ndarray): image to draw upon
    contour (Iterable[np.ndarray]): Contours to draw, found via `cv2.findContours()`
    color (Tuple[int, int, int]): color to draw the contour, tuple of ints in range [0-255] in RGB order
    thickness (float): thickness of the contour line

    Returns:
    np.ndarray, image with contours drawn
    '''
    return cv2.drawContours(im, contour, -1, color, thickness, cv2.LINE_AA)

