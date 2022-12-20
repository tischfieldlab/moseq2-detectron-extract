from concurrent.futures import ProcessPoolExecutor
import itertools
import random
from typing import Iterable, Sequence, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage.transform
from detectron2.data.catalog import MetadataCatalog, Metadata
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.structures import Instances
from detectron2.utils.visualizer import ColorMode, Visualizer
from skimage.util.dtype import img_as_bool
from tqdm import tqdm
import h5py

from moseq2_detectron_extract.io.annot import KeypointConnections, DataItem
from moseq2_detectron_extract.io.image import read_image
from moseq2_detectron_extract.io.util import gen_batch_sequence
from moseq2_detectron_extract.io.video import PreviewVideoWriter
from moseq2_detectron_extract.proc.keypoints import load_keypoint_data_from_h5
from moseq2_detectron_extract.proc.proc import colorize_video, reverse_crop_and_rotate_frame, scale_raw_frames, stack_videos
from moseq2_detectron_extract.proc.roi import get_bbox_size, get_roi_contour


def visualize_annotations(annotations: Sequence[DataItem], metadata, num: int=5):
    ''' Visualize annotatated segmentation masks and keypoints

    Parameters:
    annotations (Sequence[DataItem]): Annotated data to visualize
    metadata: dataset metadata
    num (int): Number of items to visualize

    Returns:
    fig, axs: matplotlib Figure and Axes contating visualized annotations
    '''
    fig, axs = plt.subplots(1, num, figsize=(20*num,20))
    for d, ax in zip(random.sample(annotations, num), axs):
        scale_factor = d["rescale_intensity"] if "rescale_intensity" in d else None
        im = read_image(d["file_name"], scale_factor=scale_factor, dtype='uint8')
        viz = Visualizer(im[:, :, ::-1],
                    metadata=metadata,
                    scale=2,
                    instance_mode=ColorMode.IMAGE   # remove the colors of unsegmented pixels. This option is only available for segmentation models
        )
        out = viz.draw_dataset_dict(d)
        ax.imshow(out.get_image())
    return fig, axs


def visualize_inference(frame: np.ndarray, instances: Instances, min_height: float, max_height: float, scale: float=2.0) -> np.ndarray:
    ''' Visualize Instances from model inference

    Parameters:
    frame (np.ndarray): Array containing image data, of shape (x, y)
    instances (Instances): Instances to visualize
    min_height (float): Minimum height used in frame scaling
    max_height (float): Maximum height used in frame scaling
    scale (float): geometric scaling to apply to image and visualization

    Returns:
    np.ndarray: array containing image and instance visualization
    '''
    im = frame[:,:,None].copy().astype('uint8')
    im = (im-min_height)/(max_height-min_height)
    im[im < min_height] = 0
    im[im > max_height] = max_height
    im = im * 255
    return draw_instances(im, instances, scale=scale)


def draw_instances(frame: np.ndarray, instances: Instances, scale: float=2.0, dataset_name='moseq_train') -> np.ndarray:
    ''' Draw instances using Detectron2 Visualizer class. This is slow, so for speed, use `draw_instances_fast()`
    '''
    viz = Visualizer(
            convert_image_to_rgb(frame, "L"),
            metadata=MetadataCatalog.get(dataset_name),
            scale=scale,
            instance_mode=ColorMode.SEGMENTATION   # remove the colors of unsegmented pixels. This option is only available for segmentation models
    )
    out = viz.draw_instance_predictions(instances)
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
        out = np.zeros_like(frames, shape=(frames.shape[0], width, height)) # pylint: disable=unexpected-keyword-arg
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
        out = np.zeros_like(frames, shape=(frames.shape[0], width, height, 3))  # pylint: disable=unexpected-keyword-arg
        for i in range(frames.shape[0]):
            out[i] = cv2.resize(frames[i], (width, height))
        return out


def draw_instances_fast(frame: np.ndarray, instances: Instances, keypoint_names: Sequence[str],
    keypoint_connection_rules: KeypointConnections, keypoint_colors: Sequence[Tuple[int, int, int]],
    roi_contour: Iterable[np.ndarray]=None, scale: float=2.0, radius: int=3, thickness: int=2) -> np.ndarray:
    ''' Draw `instances` on `frame`

    Parameters:
    frame (np.ndarray): grayscale frame to draw instances on, of shape (height, width, 1)
    instances (Instances): Detectron2 instances
    keypoint_names (Sequence[str]): names of the keypoints to draw
    keypoint_connection_rules (KeypointConnections): rules describing how keypoints are connected
    keypoint_colors (Sequence[Tuple[int, int, int]]): Colors to use for drawing keypoints, each a tuple of ints in the range 0-255 specifying color in RGB
    roi_contour (Iterable[np.ndarray]): Contours of the roi, found via `cv2.findContours()`
    scale (float): size scale factor to scale frame and instances by
    radius (int): radius of the circles drawn for keypoints
    thickness (int): thickness of the lines drawn for keypoint connections

    Returns:
    np.ndarray, frame with instances drawn
    '''
    return draw_instances_data_fast(frame,
                                    instances.pred_keypoints.cpu().numpy(),
                                    instances.pred_masks.cpu().numpy(),
                                    instances.pred_boxes.tensor.to('cpu').numpy(),
                                    keypoint_names=keypoint_names,
                                    keypoint_connection_rules=keypoint_connection_rules,
                                    keypoint_colors=keypoint_colors,
                                    roi_contour=roi_contour,
                                    scale=scale,
                                    radius=radius,
                                    thickness=thickness)



def draw_instances_data_fast(frame: np.ndarray, keypoints: np.ndarray, masks: np.ndarray, boxes: np.ndarray, keypoint_names: Sequence[str],
    keypoint_connection_rules: KeypointConnections, keypoint_colors: Sequence[Tuple[int, int, int]],
    roi_contour: Iterable[np.ndarray]=None, scale: float=2.0, radius: int=3, thickness: int=2) -> np.ndarray:
    ''' Draw `instances` on `frame`

    Parameters:
    frame (np.ndarray): grayscale frame to draw instances on, of shape (height, width, 1)
    keypoints (np.ndarray): keypoint data, of shape (ninstances, nkeypoints, 3 [x, y, s])
    masks (np.ndarray): mask data, of shape (ninstances, height, width)
    boxes (np.ndarray): bounding box data, of shape (ninstances, 4)
    keypoint_names (Sequence[str]): names of the keypoints to draw
    keypoint_connection_rules (KeypointConnections): rules describing how keypoints are connected
    keypoint_colors (Sequence[Tuple[int, int, int]]): Colors to use for drawing keypoints, each a tuple of ints in the range 0-255 specifying color in RGB
    roi_contour (Iterable[np.ndarray]): Contours of the roi, found via `cv2.findContours()`
    scale (float): size scale factor to scale frame and instances by
    radius (int): radius of the circles drawn for keypoints
    thickness (int): thickness of the lines drawn for keypoint connections

    Returns:
    np.ndarray, frame with instances drawn
    '''
    im = convert_image_to_rgb(frame, "L")

    if roi_contour is not None:
        im = draw_contour(im, roi_contour)

    im = cv2.resize(im, (int(im.shape[1] * scale), int(im.shape[0] * scale)))

    if masks is None:
        masks = []

    if keypoints is None:
        keypoints = []

    if boxes is None:
        boxes = []

    for mask, kpts, box in itertools.zip_longest(masks, keypoints, boxes, fillvalue=None):
        # draw mask
        if mask is not None:
            im = draw_mask(im, mask)

        # draw box
        if box is not None:
            box *= scale
            im = cv2.rectangle(im, tuple(box[0:2].astype(int)), tuple(box[2:4].astype(int)), (0,255,0))

        # keypoints
        if kpts is not None:
            im = draw_keypoints(im, kpts, keypoint_names, keypoint_connection_rules, keypoint_colors, scale=scale, radius=radius, thickness=thickness)

    return im


def draw_mask(im: np.ndarray, mask: np.ndarray, alpha: float=0.3, color: Tuple[int, int, int]=(0,0,255)) -> np.ndarray:
    ''' Draw a mask on an image

    Parameters:
    im (np.ndarray): image to draw mask upon, of shape (height, width, 3)
    mask (np.ndarray): mask to draw
    alpha (float): alpha level to use when drawing mask. 0=Transparent, 1=Opaque
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
    keypoint_colors: Sequence[Tuple[int, int, int]], origin: Tuple[int, int]=(0,0), scale: float=1.0, radius: int=3, thickness: int=2) -> np.ndarray:
    ''' Draw keypoints on an image

    Parameters:
    im (np.ndarray): image to draw keypoints upon, of shape (height, width, 3)
    keypoints (np.ndarray): Keypoints to draw
    keypoint_names (Sequence[str]): names of the keypoints to draw
    keypoint_connection_rules (KeypointConnections): rules describing how keypoints are connected
    keypoint_colors (Sequence[Tuple[int, int, int]]): Colors to use for drawing keypoints, each a tuple of ints in the range 0-255 specifying color in RGB
    origin (Tuple[int, int]): origin of the coordinate system keypoints are in
    scale (float): size scale factor to scale keypoints by
    radius (int): radius of the circles drawn for keypoints
    thickness (int): thickness of the lines drawn for keypoint connections

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
        cv2.line(im, (x1 + origin[0], y1 + origin[1]), (x2 + origin[0], y2 + origin[1]), rule[2], thickness, cv2.LINE_AA)

    # draw keypoints
    for ki in range(keypoints.shape[0]):
        x = keypoints[ki, 0].astype(int) + origin[0]
        y = keypoints[ki, 1].astype(int) + origin[1]
        im = cv2.circle(im, (x,y), radius, keypoint_colors[ki], -1, cv2.LINE_AA)
        im = cv2.circle(im, (x,y), radius, (0,0,0), 1, cv2.LINE_AA)

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


# def preview_video_from_h5(h5_file: str, dest: str, dset_name: str = 'moseq', vmin=0, vmax=100, fps=30, batch_size=10, start=None, stop=None):
#     with(h5py.File(h5_file, 'r')) as h5:
#         total_frames = h5['/frames'].shape[0]
#         roi = h5['/metadata/extraction/roi'][()]
#         roi_size = get_bbox_size(roi)

#         dset_meta = MetadataCatalog.get(dset_name)
#         clean_frames_view = CleanedFramesView(scale=1.5, dset_meta=dset_meta)
#         rot_kpt_view = RotatedKeypointsView(scale=1.5, dset_meta=dset_meta)
#         arena_view = ArenaView(roi, scale=2.0, vmin=vmin, vmax=vmax, dset_meta=dset_meta)

#         video_pipe = PreviewVideoWriter(dest, fps=fps, vmin=vmin, vmax=vmax)

#         batches = list(gen_batch_sequence(h5['/frames'].shape[0], batch_size, 0, 0))

#         with tqdm(desc='Generating Frames', total=total_frames) as pbar:
#             for batch, batch_idxs in enumerate(batches):
#                 batch_idxs = list(batch_idxs)

#                 # load data from h5 file
#                 masks = h5['/frames_mask'][batch_idxs, ...]
#                 clean_frames = h5['/frames'][batch_idxs, ...]
#                 centroids = np.stack((
#                     h5['/scalars/centroid_x_px'][batch_idxs],
#                     h5['/scalars/centroid_y_px'][batch_idxs]
#                 ), axis=1)
#                 angles = h5['/scalars/angle'][batch_idxs]
#                 rot_keypoints = load_keypoint_data_from_h5(h5, coord_system='rotated', units='px')[batch_idxs]
#                 ref_keypoints = load_keypoint_data_from_h5(h5, coord_system='reference', units='px')[batch_idxs]

#                 # rotate frames and masks to origional coordinates and angles
#                 raw_frames = np.zeros((clean_frames.shape[0], roi_size[1], roi_size[0]), dtype='uint8')
#                 raw_masks = np.zeros((clean_frames.shape[0], roi_size[1], roi_size[0]), dtype='bool')
#                 for i in range(clean_frames.shape[0]):
#                     raw_frames[i, ...] = reverse_crop_and_rotate_frame(clean_frames[i], roi_size, centroids[i], angles[i])
#                     raw_masks[i, ...] = reverse_crop_and_rotate_frame(masks[i].astype('uint8'), roi_size, centroids[i], angles[i]).astype('bool')

#                 # generate movie chunks with instance data
#                 field_video = arena_view.generate_frames(raw_frames=raw_frames, keypoints=ref_keypoints[:, None, ...], masks=raw_masks[:,None,...], boxes=None)
#                 rc_kpts_video = rot_kpt_view.generate_frames(masks=masks, keypoints=rot_keypoints)
#                 cln_depth_video = clean_frames_view.generate_frames(clean_frames=clean_frames, masks=masks)

#                 # stack and write frames
#                 proc_stack = stack_videos([cln_depth_video, rc_kpts_video], orientation='vertical')
#                 out_video_combined = stack_videos([proc_stack, field_video], orientation='horizontal')
#                 video_pipe.write_frames(batch_idxs, out_video_combined)
#                 pbar.update(n=len(batch_idxs))

#         video_pipe.close()


class H5ResultPreviewVideoGenerator():
    ''' Generates a "result preview video" from an extracted h5 result file
    '''
    def __init__(self, h5_file: str, dset_name: str = 'moseq', vmin: float = 0., vmax: float = 100., fps: int = 30,
                 batch_size: int = 100, start: int = None, stop: int = None) -> None:
        self.h5_file = h5_file
        self.dset_name = dset_name
        self.vmin = vmin
        self.vmax = vmax
        self.fps = fps
        self.batch_size = batch_size
        self.start = start
        self.stop = stop

        # These members will be set after a call to `self._initialize()`
        self.total_frames: int
        self.batches: list
        self.roi: np.ndarray
        self.roi_size: Tuple[int, int]
        self.clean_frames_view: CleanedFramesView
        self.rot_kpt_view: RotatedKeypointsView
        self.arena_view: ArenaView

        # these are paths to the datasets inside the h5 file
        self.frames_path = '/frames'
        self.mask_path = '/frames_mask'
        self.roi_path = '/metadata/extraction/roi'
        self.centroid_x_path = '/scalars/centroid_x_px'
        self.centroid_y_path = '/scalars/centroid_y_px'
        self.angle_path = '/scalars/angle'

    def _initialize(self) -> None:
        ''' Fetch some initial information from the h5 file and load it into this instance
        '''
        with(h5py.File(self.h5_file, 'r')) as h5:
            if self.start is None:
                self.start = 0

            if self.stop is None:
                self.stop = h5[self.frames_path].shape[0]

            self.total_frames = h5[self.frames_path][self.start:self.stop].shape[0]
            self.batches = list(gen_batch_sequence(self.total_frames, self.batch_size, 0, self.start))

            self.roi = h5[self.roi_path][()]
            self.roi_size = get_bbox_size(self.roi)

            dset_meta = MetadataCatalog.get(self.dset_name)
            self.clean_frames_view = CleanedFramesView(scale=1.5, dset_meta=dset_meta)
            self.rot_kpt_view = RotatedKeypointsView(scale=1.5, dset_meta=dset_meta)
            self.arena_view = ArenaView(self.roi, scale=2.0, vmin=self.vmin, vmax=self.vmax, dset_meta=dset_meta)

    def _read_chunk(self, batch_idxs: Iterable[int]) -> dict:
        ''' read a chunk from the h5 file, defined by batch_idxs, and return data as a dict
        '''
        batch_idxs = list(batch_idxs)

        with(h5py.File(self.h5_file, 'r')) as h5:
            # load data from h5 file
            masks = h5[self.mask_path][batch_idxs, ...]
            clean_frames = h5[self.frames_path][batch_idxs, ...]
            centroids = np.stack((
                h5[self.centroid_x_path][batch_idxs],
                h5[self.centroid_y_path][batch_idxs]
            ), axis=1)
            angles = h5[self.angle_path][batch_idxs]
            rot_keypoints = load_keypoint_data_from_h5(h5, coord_system='rotated', units='px')[batch_idxs]
            ref_keypoints = load_keypoint_data_from_h5(h5, coord_system='reference', units='px')[batch_idxs]

            # rotate frames and masks to origional coordinates and angles
            raw_frames = np.zeros((clean_frames.shape[0], self.roi_size[1], self.roi_size[0]), dtype='uint8')
            raw_masks = np.zeros((clean_frames.shape[0], self.roi_size[1], self.roi_size[0]), dtype='bool')
            for i in range(clean_frames.shape[0]):
                raw_frames[i] = reverse_crop_and_rotate_frame(clean_frames[i], self.roi_size, centroids[i], angles[i])
                raw_masks[i] = reverse_crop_and_rotate_frame(masks[i].astype('uint8'), self.roi_size, centroids[i], angles[i]).astype('bool')

        return {
            'batch_idxs': batch_idxs,
            'masks': masks,
            'clean_frames': clean_frames,
            'centroids': centroids,
            'angles': angles,
            'rot_keypoints': rot_keypoints,
            'ref_keypoints': ref_keypoints[:, None, ...],
            'raw_frames': raw_frames,
            'raw_masks': raw_masks[:, None, ...]
        }

    def _write_chunk(self, video_pipe: PreviewVideoWriter, data: dict) -> None:
        ''' Compile a chunk, `data`, into final frames, and send to the video writer, `video_pipe`, to be written to disk
        '''
        # generate movie chunks with instance data
        field_video = self.arena_view.generate_frames(raw_frames=data['raw_frames'], keypoints=data['ref_keypoints'], masks=data['raw_masks'], boxes=None)
        rc_kpts_video = self.rot_kpt_view.generate_frames(masks=data['masks'], keypoints=data['rot_keypoints'])
        cln_depth_video = self.clean_frames_view.generate_frames(clean_frames=data['clean_frames'], masks=data['masks'])

        # stack and write frames
        proc_stack = stack_videos([cln_depth_video, rc_kpts_video], orientation='vertical')
        out_video_combined = stack_videos([proc_stack, field_video], orientation='horizontal')
        video_pipe.write_frames(data['batch_idxs'], out_video_combined)

    def generate(self, dest: str) -> None:
        ''' Commence generation of the video

        Parameters:
        dest (str): destination for the video file
        '''
        self._initialize()
        video_pipe = PreviewVideoWriter(dest, fps=self.fps, vmin=self.vmin, vmax=self.vmax)
        pool = ProcessPoolExecutor(max_workers=2)

        with tqdm(desc='Generating Frames', total=self.total_frames) as pbar:

            for data in pool.map(self._read_chunk, self.batches):
                #print(data['batch_idxs'])
                self._write_chunk(video_pipe, data)
                pbar.update(n=len(data['batch_idxs']))

        # TODO: We really should be calling shutdown, but this seems to reliably cause an exception
        # should be fixed on python=3.9, but for now lets just pretend we shutdown
        # see: https://github.com/python/cpython/issues/83285
        # pool.shutdown(wait=True)

        video_pipe.close()






class BaseView():
    ''' Base class for a view
    '''
    def __init__(self, dset_meta: Metadata, scale: float = 1,) -> None:
        self.is_setup = False
        self.scale = scale
        self.dset_meta = dset_meta


class ArenaView(BaseView):
    ''' A view showing the arena depth image plus any instance annotations
    '''
    def __init__(self, roi: np.ndarray, dset_meta: Metadata, scale: float = 2, vmin: float = 0.0, vmax: float = 100.0) -> None:
        super().__init__(scale=scale, dset_meta=dset_meta)
        self.vmin = vmin
        self.vmax = vmax
        self.contour = get_roi_contour(roi, crop=True)


    def generate_frames(self, raw_frames: np.ndarray, keypoints: np.ndarray = None, masks: np.ndarray = None, boxes: np.ndarray = None):
        ''' Generate frames for this view
        '''
        rfs = raw_frames.shape
        video = np.zeros((rfs[0], int(rfs[1]*self.scale), int(rfs[2]*self.scale), 3), dtype='uint8')

        for i in range(rfs[0]):
            scaled_frames = scale_raw_frames(raw_frames[i,:,:,None].copy(), vmin=self.vmin, vmax=self.vmax)
            video[i,:,:,:] = draw_instances_data_fast(
                                scaled_frames,
                                keypoints=keypoints[i] if keypoints is not None else None,
                                masks=masks[i] if masks is not None else None,
                                boxes=boxes[i] if boxes is not None else None,
                                roi_contour=self.contour,
                                scale=self.scale,
                                keypoint_names=self.dset_meta.keypoint_names,
                                keypoint_connection_rules=self.dset_meta.keypoint_connection_rules,
                                keypoint_colors=self.dset_meta.keypoint_colors,
                                thickness=1)
        return video



class RotatedKeypointsView(BaseView):
    ''' A view showing cropped and rotated masks and keypoints
    '''
    def __init__(self, dset_meta: Metadata, scale: float = 1.5) -> None:
        super().__init__(scale=scale, dset_meta=dset_meta)

    def generate_frames(self, masks: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
        ''' Generate frames for this view
        '''
        width = int(masks.shape[2] * 1.5)
        height = int(masks.shape[1] * 1.5)
        video = np.zeros((masks.shape[0], height, width, 3), dtype='uint8')
        origin = (int(width // 2), int(height // 2))

        for i in range(masks.shape[0]):
            video[i,:,:,:] = draw_mask(video[i,:,:,:], masks[i], alpha=0.7)
            video[i,:,:,:] = draw_keypoints(video[i,:,:,:],
                                            keypoints[i],
                                            origin=origin,
                                            keypoint_names=self.dset_meta.keypoint_names,
                                            keypoint_connection_rules=self.dset_meta.keypoint_connection_rules,
                                            keypoint_colors=self.dset_meta.keypoint_colors,
                                            scale=1.5,
                                            radius=3,
                                            thickness=1)

        return video

class CleanedFramesView(BaseView):
    ''' A view showing cleaned, cropped, and rotated depth frames
    '''
    def __init__(self, dset_meta: Metadata, scale: float = 1.5) -> None:
        super().__init__(scale=scale, dset_meta=dset_meta)

    def generate_frames(self, clean_frames: np.ndarray, masks: np.ndarray) -> np.ndarray:
        ''' Generate frames for this view
        '''
        cleaned_depth = colorize_video(scale_depth_frames(clean_frames * masks, scale=1.5))
        return cleaned_depth
