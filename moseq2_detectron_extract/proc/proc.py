from typing import Dict, List, Literal, Optional, Sequence, Tuple
import warnings

import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy
import tqdm
from bottleneck import move_median

from moseq2_detectron_extract.io.util import find_unused_file_path
from moseq2_detectron_extract.proc.kalman import (KalmanTracker,
                                                  angle_difference)
from moseq2_detectron_extract.proc.keypoints import rotate_points_batch
from moseq2_detectron_extract.proc.roi import apply_roi


def stack_videos(videos: Sequence[np.ndarray], orientation: Literal['horizontal', 'vertical', 'diagional']) -> np.ndarray:
    ''' Stack videos according to orientation to create one big video

    Parameters:
    videos (Iterable[np.ndarray]): Iterable of videos to stack of shape (nframes, height, width, channels).
        All videos must match dimentions in axis 0 and axis 3
    orientation (Literal['horizontal', 'vertical', 'diagional']): orientation of stacking.

    Retruns:
    stacked composite video
    '''
    dtype = reduce_dtypes(videos)
    nframes = reduce_axis_size(videos, 0)
    channels = reduce_axis_size(videos, 3)
    heights = [v.shape[1] for v in videos]
    widths = [v.shape[2] for v in videos]

    if orientation == 'horizontal':
        height = max(heights)
        width = sum(widths)
    elif orientation == 'vertical':
        height = sum(heights)
        width = max(widths)
    elif orientation == 'diagional':
        height = sum(heights)
        width = sum(widths)
    else:
        raise ValueError(f'Unknown orientation "{orientation}". Expected one of ["horizontal", "vertical"].')

    output_movie = np.zeros((nframes, height, width, channels), dtype)
    for i, video in enumerate(videos):
        if orientation == 'horizontal':
            offset = sum([w for wi, w in enumerate(widths) if wi < i])
            output_movie[:, :heights[i], offset:offset+widths[i], :] = video
        elif orientation == 'vertical':
            offset = sum([h for hi, h in enumerate(heights) if hi < i])
            output_movie[:, offset:offset+heights[i], :widths[i], :] = video
        elif orientation == 'diagional':
            offsetw = sum([w for wi, w in enumerate(widths) if wi < i])
            offseth = sum([h for hi, h in enumerate(heights) if hi < i])
            output_movie[:, offseth:offseth+heights[i], offsetw:offsetw+widths[i], :] = video

    return output_movie


def reduce_axis_size(data: Sequence[np.ndarray], axis: int) -> int:
    ''' Reduce an iterable of numpy.ndarrays to a single scalar for a given axis
    Will raise an exception if no items are passed or the arrays are not the same size on the given axis

    Parameters:
    data (Iterable[np.ndarray]): Arrays to inspect
    axis (int): axis to be inspected

    Returns:
    int - the shared shape of `axis` in all arrays in `data`
    '''
    if len(data) <= 0:
        raise ValueError('Need a list with at least one array!')

    sizes = {d.shape[axis] for d in data}
    if len(sizes) == 1:
        return int(sizes.pop())
    else:
        raise ValueError(f'Arrays should be equal sized on axis{axis}!')


def reduce_dtypes(data: Sequence[np.ndarray]) -> npt.DTypeLike:
    ''' Reduce an iterable of numpy.ndarrays to a dtype
    Will raise an exception if no items are passed or the all arrays do not share a dtype

    Parameters:
    data (Iterable[np.ndarray]): Arrays to inspect

    Returns:
    npt.DTypeLike - the shared dtype of all arrays in `data`
    '''
    if len(data) <= 0:
        raise ValueError('Need a list with at least one array!')

    dtypes = {d.dtype for d in data}
    if len(dtypes) == 1:
        return dtypes.pop()
    else:
        raise ValueError('Arrays should have same dtype!')


def colorize_video(frames: np.ndarray, vmin: float=0, vmax: float=100, cmap: str='jet') -> np.ndarray:
    ''' Colorize single channel video data

    Parameters:
    frames (np.ndarray): frames to be colorized, assumed shape (nframes, height, width)
    vmin (float): minimum data value corresponding to cmap min
    vmax (float): maximum data value corresponding to cmap max
    cmap (str): colormap to use for converting to color

    Returns:
    np.ndarray containing colorized frames of shape (nframe, height, width, 3)
    '''
    use_cmap = plt.get_cmap(cmap)

    disp_img = frames.copy().astype('float32')
    disp_img = (disp_img-vmin)/(vmax-vmin)
    disp_img[disp_img < 0] = 0
    disp_img[disp_img > 1] = 1
    disp_img = use_cmap(disp_img)[...,:3]*255

    return disp_img.astype('uint8')


def prep_raw_frames(frames: np.ndarray, bground_im: Optional[np.ndarray]=None, roi: Optional[np.ndarray]=None, vmin: Optional[float]=None, vmax: Optional[float]=None,
                    dtype: npt.DTypeLike='uint8', fix_invalid_pixels=True) -> np.ndarray:
    ''' Prepare raw `frames` by:
            1) subtracting background based on `bground_im`
            2) applying a region of interest (crop and mask according to `roi`)
            3) clamping values in the image to `vmin` and `vmax`
                a) values less than `vmin` are set to zero
                b) values greater than `vmax` are set to `vmax`
            All operations are optional.

    Parameters:
    frames (np.ndarray): frames to process, of shape (nframes, height, width).
    bground_im (np.ndarray): background image to subtract from `frames`, of shape (height, width). If None, the operation is skipped.
    roi (np.ndarray): mask image specifying the region of interest, of shape (height, width), used to crop and mask `frames`. If None, the operation is skipped.
    vmin (float): minimum value allowed in `frames`, values less than this parameter will be set to this value. If None, the operation is skipped.
    vmax (float): maximum value allowed in `frames`, values greater than this parameter will be set to this value. If None, the operation is skipped.
    dtype (npt.DTypeLike): dtype of the returned frames (default='uint8')

    Returns:
    Processed frames of shape (nframes, roi_height, roi_width)
    '''
    if fix_invalid_pixels:
        mask = find_invalid_pixels(frames)

    if bground_im is not None:
        frames = bground_im - frames

    if roi is not None:
        frames = apply_roi(frames, roi)
        if fix_invalid_pixels:
            mask = apply_roi(mask, roi)

    if vmin is not None:
        frames[frames < vmin] = 0

    if vmax is not None:
        frames[frames > vmax] = vmax

    frames = frames.astype(dtype)

    if fix_invalid_pixels:
        frames = fill_invalid_pixels(frames, mask)

    return frames


def find_invalid_pixels(frames: np.ndarray) -> np.ndarray:
    ''' Find invalid pixels in `frames` and return their locations as a mask

    Parameters:
    frames (np.ndarray): frames to search for invalid pixels, of shape (nframes, height, width)

    Returns:
    Mask with same shape as `frames`. Zeros indicate valid pixels and ones indicate invalid pixels
    '''
    mask = np.zeros_like(frames, dtype='uint8')
    mask[frames == 0] = 1 # values of zero indicate bad pixels in Kinect v2
    return mask


def fill_invalid_pixels(frames: np.ndarray, invalid_mask: np.ndarray) -> np.ndarray:
    ''' Fill invalid pixels in `frames`

    We use openCV inpaint method, choosing the algorithm cv2.INPAINT_NS. cv.INPAINT_TELEA was also
    considered; both seem to have similar results but cv2.INPAINT_NS is slightly faster with lower
    variance of execution time, so we go with that one.

    Parameters:
    frames (np.ndarray): frames to fill, of shape (nframes, height, width)
    invalid_mask (np.ndarray): mask where ones indicating invalid pixels to be filled, of same shape as `frames`

    Returns:
    frames with invalid pixels filled.
    '''
    assert frames.shape == invalid_mask.shape

    if frames.dtype == np.int16:
        frames = frames.astype(np.uint16)

    for i in range(frames.shape[0]):
        frames[i] = cv2.inpaint(frames[i], invalid_mask[i], 3, cv2.INPAINT_NS)
    return frames



def scale_raw_frames(frames: np.ndarray, vmin: float, vmax: float, dtype: npt.DTypeLike='uint8') -> np.ndarray:
    ''' Linear scale `frames` to the range afforded by `dtype`

    Parameters:
    frames (np.ndarray): data to intensity scale
    vmin (float): data minimum value
    vmax (float): data maximum value
    dtype (npt.DTypeLike): Data type to return as

    Returns:
    np.ndarray contatining intensity scaled frames data
    '''
    real_dtype = np.dtype(dtype)
    if np.issubdtype(real_dtype, np.integer):
        dmin = float(np.iinfo(real_dtype).min)
        dmax = float(np.iinfo(real_dtype).max)
    else:
        dmin = float(np.finfo(real_dtype.name).min)
        dmax = float(np.finfo(real_dtype.name).max)

    return ((frames - vmin) * ((dmax - dmin) / (vmax - vmin)) + dmin).astype(dtype)


def get_frame_features(frames: np.ndarray, frame_threshold: float=10, mask: np.ndarray=np.array([]), mask_threshold: float=-30,
    use_cc: bool=False, progress_bar: bool=True) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    ''' Use image moments to compute features of the largest object in the frame

    Parameters:
    frames (np.ndarray): frames of shape (nframes, height, width)
    frame_threshold (float): threshold in mm separating floor from mouse
    mask (np.ndarray): mask to consider in feature extraction
    mask_threshold (float): masking threshold to use for connected components detection
    use_cc (bool): True to use connected components to augment masks
    progress_bar (bool): True to show a progress bar for the operation

    Returns:
    features, masks (Tuple[Dict[str, np.ndarray], np.ndarray]):

    Features is a dictionary with simple image features
    Masks are image masks of shape (nframes, height, width)
    '''

    nframes = frames.shape[0]

    if isinstance(mask, np.ndarray) and mask.size > 0:
        has_mask = True
    else:
        has_mask = False
        mask = np.zeros((frames.shape), 'uint8')

    features: dict = {
        'centroid': np.empty((nframes, 2)),
        'orientation': np.empty((nframes,)),
        'axis_length': np.empty((nframes, 2)),
    }

    for k in features.keys():
        features[k][:] = np.nan

    features['contour'] = []

    for i in tqdm.tqdm(range(nframes), disable=not progress_bar, desc='Computing moments'):

        frame_mask = frames[i, ...] > frame_threshold

        if use_cc:
            cc_mask = get_largest_cc((frames[[i], ...] > mask_threshold).astype('uint8')).squeeze()
            frame_mask = np.logical_and(cc_mask, frame_mask)

        if has_mask:
            frame_mask = np.logical_and(frame_mask, mask[i, ...])
        else:
            mask[i, ...] = frame_mask

        cnts, _ = cv2.findContours(
            frame_mask.astype('uint8'),
            cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        tmp = np.array([cv2.contourArea(x) for x in cnts])

        if tmp.size == 0:
            continue

        mouse_cnt = tmp.argmax()

        for key, value in im_moment_features(cnts[mouse_cnt]).items():
            features[key][i] = value
        features['contour'].append(cnts[mouse_cnt])

    return features, mask


def crop_and_rotate_frame(frame: np.ndarray, center: Tuple[float, float], angle: float, crop_size: Tuple[int, int]=(80, 80)) -> np.ndarray:
    ''' Rotate a single frame around `center` by `angle` degrees and crop to `crop_size` centered on `center`

    Parameters:
    frame (np.ndarray): Frame to crop and rotate, of shape (x, y)
    center (Tuple[float, float]): center coordinates of operation, (x, y)
    angle (float): angle by which to rotate, in degrees
    crop_size (Tuple[int, int]): size of the output image, (x, y)

    Returns:
    np.ndarray: copped and rotated frames
    '''
    if np.isnan(angle) or np.any(np.isnan(center)):
        return np.zeros_like(frame, shape=crop_size) # pylint: disable=unexpected-keyword-arg

    if np.any(center < 0):
        warnings.warn(f"Encountered center < 0 ({center[0]}, {center[1]}).")
        return np.zeros_like(frame, shape=crop_size) # pylint: disable=unexpected-keyword-arg

    try:
        xmin = int(center[0] - crop_size[0] // 2) + crop_size[0]
        xmax = int(center[0] + crop_size[0] // 2) + crop_size[0]
        ymin = int(center[1] - crop_size[1] // 2) + crop_size[1]
        ymax = int(center[1] + crop_size[1] // 2) + crop_size[1]

        border = (crop_size[1], crop_size[1], crop_size[0], crop_size[0])
        rot_mat = cv2.getRotationMatrix2D((crop_size[0] // 2, crop_size[1] // 2), angle, 1)
        use_frame = cv2.copyMakeBorder(frame, *border, cv2.BORDER_CONSTANT, 0)
        return cv2.warpAffine(use_frame[ymin:ymax, xmin:xmax], rot_mat, (crop_size[0], crop_size[1]))
    except Exception as e:
        return np.zeros_like(frame, shape=crop_size) # pylint: disable=unexpected-keyword-arg
        # raise ValueError("Here is a snapshot of data:\n"
        #                 f"center: [{center[0]}, {center[1]}]\n"
        #                 f"crop_size: [{crop_size[0]}, {crop_size[1]}]\n"
        #                 f"angle: {angle}\n"
        #                 f"extents: [[{xmin}, {xmax}], [{ymin}, {ymax}]]") from e


def reverse_crop_and_rotate_frame(frame: np.ndarray, dest_size: Tuple[int, int], center: Tuple[float, float], angle: float):
    ''' Attempts to reverse the process of `crop_and_rotate_frame()`.

    Parameters:
    frame (np.ndarray): Cropped and rotated frame to reverse crop/rotate process, of shape (y, x)
    dest_size (Tuple[int, int]): size of the output image, (width, height)
    center (Tuple[float, float]): center coordinates in real space (not cropped/rotated), (x, y)
    angle (float): angle by which to rotate, in degrees

    Returns:
    np.ndarray: reverse copped and rotated frames
    '''
    if np.isnan(angle) or np.any(np.isnan(center)):
        return np.zeros_like(frame, shape=(dest_size[1], dest_size[0])) # pylint: disable=unexpected-keyword-arg

    frame = frame.copy()
    src_shape = frame.shape
    src_center = (src_shape[1] // 2, src_shape[0] // 2)

    rot_mat = cv2.getRotationMatrix2D(src_center, -angle, 1)
    frame = cv2.warpAffine(frame, rot_mat, (dest_size[0], dest_size[1]))

    translate_mat = np.array([
        [1, 0, center[0] - src_center[0]],
        [0, 1, center[1] - src_center[1]]
    ], dtype=float)
    frame = cv2.warpAffine(frame, translate_mat, (dest_size[0], dest_size[1]))

    return frame


def crop_and_rotate_frames(frames: np.ndarray, features: Dict[str, np.ndarray], crop_size: Tuple[int, int]=(80, 80),
                           progress_bar: bool=True):
    ''' Rotate a `frames` and crop to `crop_size` given features (containing orientation and centroid keys)

    Parameters:
    frames (np.ndarray): Frames to crop and rotate, of shape (nframes, x, y)
    features (Dict[str, np.ndarray]): dict of features, containing keys orientation and centroid
    crop_size (Tuple[int, int]): size of the output image, (x, y)
    progress_bar (bool): True to show a progress bar for the operation

    Returns:
    np.ndarray: copped and rotated frames
    '''
    nframes = frames.shape[0]
    cropped_frames = np.zeros((nframes, crop_size[0], crop_size[1]), frames.dtype)
    win = (crop_size[0] // 2, crop_size[1] // 2 + 1)
    border = (crop_size[1], crop_size[1], crop_size[0], crop_size[0])

    for i in tqdm.tqdm(range(frames.shape[0]), disable=not progress_bar, desc='Rotating'):

        if np.any(np.isnan(features['centroid'][i, :])):
            continue

        # use_frame = np.pad(frames[i, ...], (crop_size, crop_size), 'constant', constant_values=0)
        use_frame = cv2.copyMakeBorder(frames[i, ...], *border, cv2.BORDER_CONSTANT, 0)

        rr = np.arange(features['centroid'][i, 1]-win[0],
                       features['centroid'][i, 1]+win[1]).astype('int16')
        cc = np.arange(features['centroid'][i, 0]-win[0],
                       features['centroid'][i, 0]+win[1]).astype('int16')

        rr = rr+crop_size[0]
        cc = cc+crop_size[1]

        if (np.any(rr >= use_frame.shape[0]) or np.any(rr < 1)
                or np.any(cc >= use_frame.shape[1]) or np.any(cc < 1)):
            continue

        rot_mat = cv2.getRotationMatrix2D((crop_size[0] // 2, crop_size[1] // 2),
                                          -np.rad2deg(features['orientation'][i]), 1)
        cropped_frames[i, :, :] = cv2.warpAffine(use_frame[rr[0]:rr[-1], cc[0]:cc[-1]],
                                                 rot_mat, (crop_size[0], crop_size[1]))

    return cropped_frames


def feature_hampel_filter(features, centroid_hampel_span=None, centroid_hampel_sig=3,
                          angle_hampel_span=None, angle_hampel_sig=3):
    ''' Apply a hampel filter to features
    '''

    if centroid_hampel_span is not None and centroid_hampel_span > 0:
        padded_centroids = np.pad(features['centroid'],
                                  (((centroid_hampel_span // 2, centroid_hampel_span // 2)),
                                   (0, 0)),
                                  'constant', constant_values = np.nan)
        for i in range(1):
            vws = strided_app(padded_centroids[:, i], centroid_hampel_span, 1)
            med = np.nanmedian(vws, axis=1)
            mad = np.nanmedian(np.abs(vws - med[:, None]), axis=1)
            vals = np.abs(features['centroid'][:, i] - med)
            fill_idx = np.where(vals > med + centroid_hampel_sig * mad)[0]
            features['centroid'][fill_idx, i] = med[fill_idx]


    if angle_hampel_span is not None and angle_hampel_span > 0:
        padded_orientation = np.pad(features['orientation'],
                                    (angle_hampel_span // 2, angle_hampel_span // 2),
                                    'constant', constant_values = np.nan)
        vws = strided_app(padded_orientation, angle_hampel_span, 1)
        med = np.nanmedian(vws, axis=1)
        mad = np.nanmedian(np.abs(vws - med[:, None]), axis=1)
        vals = np.abs(features['orientation'] - med)
        fill_idx = np.where(vals > med + angle_hampel_sig * mad)[0]
        features['orientation'][fill_idx] = med[fill_idx]

    return features


def hampel_filter(data, span, sigma=3):
    ''' Apply a hampel filter
    '''
    if len(data.shape) == 1:
        padded_data = np.pad(data, (span // 2, span // 2), 'constant', constant_values=np.nan)
        vws = broadcasting_app(padded_data, span, 1)
        med = np.nanmedian(vws, axis=1)
        mad = np.nanmedian(np.abs(vws - med[:, None]), axis=1)
        vals = np.abs(data - med)
        fill_idx = np.where(vals > med + sigma * mad)[0]
        data[fill_idx] = med[fill_idx]

    elif len(data.shape) == 2:
        padded_data = np.pad(data, ((span // 2, span // 2), (0,)*data.shape[1]), 'constant', constant_values=np.nan)
        for i in range(data.shape[1]):
            vws = strided_app(padded_data[:, i], span, 1)
            med = np.nanmedian(vws, axis=1)
            mad = np.nanmedian(np.abs(vws - med[:, None]), axis=1)
            vals = np.abs(data[:, i] - med)
            fill_idx = np.where(vals > med + sigma * mad)[0]
            data[fill_idx, i] = med[fill_idx]
    else:
        raise ValueError(f"cannot accept data with {len(data.shape)} dimentions!")

    return data


def clean_frames(frames: np.ndarray, prefilter_space=(3,), prefilter_time=None,
                 strel_tail=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
                 iters_tail=None, frame_dtype='uint8',
                 strel_min=cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
                 iters_min=None, progress_bar=True) -> np.ndarray:
    ''' Simple filtering, median filter and morphological opening

    Parameters:
    frames (3d np array): frames x r x c
    strel (opencv structuring element): strel for morph opening
    iters_tail (int): number of iterations to run opening

    Returns:
    filtered_frames (3d np array): frame x r x c
    '''
    # seeing enormous speed gains w/ opencv
    filtered_frames = frames.copy().astype(frame_dtype)

    for i in tqdm.tqdm(range(frames.shape[0]),
                       disable=not progress_bar, desc='Cleaning frames'):

        if iters_min is not None and iters_min > 0:
            filtered_frames[i, ...] = cv2.erode(filtered_frames[i, ...], strel_min, iters_min)

        if prefilter_space is not None and np.all(np.array(prefilter_space) > 0):
            for pfs in prefilter_space:
                filtered_frames[i, ...] = cv2.medianBlur(filtered_frames[i, ...], pfs)

        if iters_tail is not None and iters_tail > 0:
            filtered_frames[i, ...] = cv2.morphologyEx(filtered_frames[i, ...], cv2.MORPH_OPEN, strel_tail, iters_tail)

    if prefilter_time is not None and np.all(np.array(prefilter_time) > 0) and np.all(np.array(prefilter_time) <= filtered_frames.shape[0]):
        for pft in prefilter_time:
            filtered_frames = scipy.signal.medfilt(filtered_frames, [pft, 1, 1])

    return filtered_frames


def im_moment_features(image: np.ndarray) -> dict:
    ''' Use the method of moments and centralized moments to get image properties

    Parameters:
    image (2d numpy array): depth image

    Returns:
    Features (dictionary): returns a dictionary with orientation,
    centroid, and ellipse axis length
    '''

    tmp = cv2.moments(image)
    num = 2*tmp['mu11']
    den = tmp['mu20']-tmp['mu02']

    common = np.sqrt(4*np.square(tmp['mu11'])+np.square(den))

    if tmp['m00'] == 0:
        features = {
            'orientation': np.nan,
            'centroid': np.nan,
            'axis_length': [np.nan, np.nan]
        }
    else:
        features = {
            'orientation': -.5*np.arctan2(num, den),
            'centroid': [tmp['m10']/tmp['m00'], tmp['m01']/tmp['m00']],
            'axis_length': [2*np.sqrt(2)*np.sqrt((tmp['mu20']+tmp['mu02']+common)/tmp['m00']),
                            2*np.sqrt(2)*np.sqrt((tmp['mu20']+tmp['mu02']-common)/tmp['m00'])]
        }

    return features


def get_largest_cc(frames, progress_bar=False):
    ''' Returns largest connected component blob in image
    Parameters:
    frame (3d numpy array): frames x r x c, uncropped mouse
    progress_bar (bool): display progress bar

    Returns:
    flips (3d bool array):  frames x r x c, true where blob was found
    '''
    foreground_obj = np.zeros((frames.shape), 'bool')

    for i in tqdm.tqdm(range(frames.shape[0]), disable=not progress_bar, desc='CC'):
        _, output, stats, _ =\
            cv2.connectedComponentsWithStats(frames[i, ...], connectivity=4)
        szs = stats[:, -1]
        foreground_obj[i, ...] = output == szs[1:].argmax()+1

    return foreground_obj



def strided_app(a, L, S):
    '''Make a strided app
    see: https://stackoverflow.com/questions/40084931/taking-subarrays-from-numpy-array-with-given-stride-stepsize/40085052#40085052
    dang this is fast!

    Parameters:
    a: array
    L: Window len
    S: Stride len/stepsize
    '''
    nrows = ((a.size-L)//S)+1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a, shape=(nrows, L), strides=(S*n, n))


def broadcasting_app(a, L, S ):
    '''Make a broadcasting app

    Parameters:
    a: array
    L: Window len
    S: Stride len/stepsize
    '''
    nrows = ((a.size-L)//S)+1
    return a[S*np.arange(nrows)[:,None] + np.arange(L)]


def filter_angles(angles: np.ndarray, window: int=3, tolerance: float=60) -> np.ndarray:
    ''' Correct angle measurements that are approximatly 180 degrees off

    We slide a moving median of size `window` across angles and compute the deviation from the window
    median. If the absolute difference between the window median and the real value is approximatly
    180 degrees (180 - tolerance < a < 180 + tolerance), we consider it flipped and add 180 degrees
    to the value (correctly signed).

    Parameters:
    angles (np.ndarray): angles to inspect, of shape (nframes,)
    window (int): window size to use when inspecting angles
    tolerance (float): tolerance of the angle deviance relative to 180 degrees

    Returns:
    np.ndarray containing corrected angle values
    '''
    out = np.copy(angles)
    window = min(window, out.shape[0])
    windows = move_median(angles, window=window, min_count=1)
    diff = out - windows
    absdiff = np.abs(diff)
    flips = ((absdiff>(180-tolerance)) & (absdiff<(180+tolerance)))
    signs = np.sign(diff[flips])
    out[flips] = out[flips] + (-180 * signs)
    return out


def iterative_filter_angles(angles: np.ndarray, window: int=3, tolerance: float=60, max_iters: int=1000) -> Tuple[np.ndarray, np.ndarray]:
    ''' Iteratively filter angles until filtering stabilizes or `max_iters` is reached

    Parameters:
    angles (np.ndarray): angles to inspect, of shape (nframes,)
    window (int): window size to use when inspecting angles
    tolerance (float): tolerance of the angle deviance relative to 180 degrees
    max_iters (int): maximum number of iterations allowed

    Returns:
    (angles, flips) - angles are corrected angles. flips is bool np.ndarray with True values indicating a flipped index
    '''
    last = np.copy(angles)
    iterations = 0
    while True:
        if iterations > max_iters:
            break

        iterations += 1
        curr = filter_angles(last, window=window, tolerance=tolerance)

        if np.allclose(curr, last):
            # logging.debug(f'Converged after {iterations} iterations')
            break

        last = curr
    flips = np.isclose(np.abs(curr - angles), 180)
    return curr, flips


def mask_and_keypoints_from_model_output(model_outputs: List[dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ''' Convert Detectron2 model output to arrays of data

    https://detectron2.readthedocs.io/en/latest/tutorials/models.html#model-output-format

    Parameters:
    model_outputs (List[dict]): model outputs to parse

    Returns:
    Tuple of (masks, keypoints, num_instances).
    masks - array of shape (nframes, 1, height, width)
    keypoints - array of shape (nframes, 1, nkeypoints, 3 [x, y, s])
    num_instances - array of shape (nframes,)
    '''
    # initialize output arrays
    first = model_outputs[0]["instances"]
    masks = np.zeros((len(model_outputs), 1, *first.pred_masks.shape[1:]), dtype='uint8')
    # masks.fill(np.nan) # NaNs cannot be used in uint8 arrays :(
    keypoints = np.empty((len(model_outputs), 1, first.pred_keypoints.shape[1], 3), dtype=float)
    keypoints.fill(np.nan)
    num_instances = np.zeros((len(model_outputs)), dtype=int)

    # fill output with data
    for i, output in enumerate(model_outputs):
        num_instances[i] = len(output["instances"])
        if len(output["instances"]) > 0:
            keypoints[i, 0] = output["instances"].pred_keypoints[0, :, :].cpu().numpy()
            masks[i, 0] = output["instances"].pred_masks[0, :, :].cpu().numpy()
    return masks, keypoints, num_instances


def clamp_angles_deg(angles: np.ndarray) -> np.ndarray:
    ''' Clamp angles so they are always in the range of [0, 360)
    '''
    return np.where(angles < 0, 360 + angles, angles) % 360


def clamp_angles_rad(angles: np.ndarray) -> np.ndarray:
    ''' Clamp angles so they are always in the range of [0, 2 * pi)
    '''
    return np.where(angles < 0, (2 * np.pi) + angles, angles) % (2 * np.pi)


def instances_to_features(model_outputs: List[dict], raw_frames: np.ndarray, point_tracker: KalmanTracker, angle_tracker: KalmanTracker, debug: bool = True):
    ''' Detect additional features and perform feature postprocessing

    Parameters:
    model_outputs (List[dict]): Model output data in Detectron2 output format
    raw_frames (np.ndarray): raw depth frames of shape (nframes, height, width)

    Returns:
    Dict[str, Any] - dict containing features and cleaned data
    '''
    DEBUG = debug

    # frames x instances x keypoints x 3
    d2_masks, allocentric_keypoints, num_instances = mask_and_keypoints_from_model_output(model_outputs)

    # Clean frames and get features
    cleaned_frames = clean_frames(raw_frames, iters_tail=3, progress_bar=False)
    features, masks = get_frame_features(cleaned_frames, mask=d2_masks[:, 0, :, :], use_cc=True, frame_threshold=3, progress_bar=False)


    lengths = np.max(features['axis_length'], axis=1)
    aspects = np.min(features['axis_length'], axis=1) / np.max(features['axis_length'], axis=1)
    angles = features['orientation']
    angles = -np.rad2deg(angles)  # convert angles from radians to degrees
    angles = clamp_angles_deg(angles)  # enforce angles are in the range [0, 360)

    if DEBUG:
        orig_angles = np.copy(angles)
        flip_info = []

    if point_tracker is not None and angle_tracker is not None:
        # STRATEGY (tracking-based):
        # 1) improve point estimations by smoothing centroids and keypoints using tracker
        # 2) use smoothed keypoints to flip angles
        # 3) peak at next angle prediction, see if it makes any sense, potentially augment
        # 4) filter and update angle tracking

        if not point_tracker.is_initialized:
            # we need to initialize the point tracker with some data, drop any non-finate values (not supported by kalman em estimation)
            point_tracker.initialize([
                features['centroid'],
                allocentric_keypoints[:, 0, :, :2]
            ])


        # apply kalman smoothing to the centroids and keypoints
        s_centroids, s_kpts = point_tracker.smooth_update([
            features['centroid'],
            allocentric_keypoints[:, 0, :, :2]
        ])
        features['centroid'] = s_centroids
        allocentric_keypoints[:, 0, :7, :2] = s_kpts[:, :7, :] # assign back but skip tailtip (inference is better than tracked since it moves so fast!)

        # apply flips to angles given smoothed keypoint information
        flips, flip_confs = flips_from_keypoints(allocentric_keypoints[:, 0, ...], features['centroid'], angles, lengths)
        angles[flips] = clamp_angles_deg(angles[flips] + 180)  # IMPORTANT! Enforce angles remain within range [0, 360)
        if DEBUG:
            post_kp_flip_angles = angles.copy()

        rot_kpts = rotate_points_batch(np.copy(allocentric_keypoints[:, 0, :7, :2]), features['centroid'], angles)
        kpt_alignment_scores = compute_keypoint_alignment_scores(rot_kpts)
        kpt_rotations = estimate_keypoint_rotation(rot_kpts)


        if not angle_tracker.is_initialized:
            # we need to initialize the angle tracker with some data
            angle_tracker.initialize([angles])


        # Iterate through angles and apply our heuristic
        for i in range(angles.shape[0]):
            # Get the next predicted state from the tracker
            # we are at time t, the tracker is at time t-1, sample 1 into the future (i.e. now)
            p_next_angle, = angle_tracker.sample(1)
            rel_angle_dist = angle_difference(p_next_angle, angles[[i]])[0]

            # Potentially intervene here
            if kpt_alignment_scores[i] < 0.4:
                # if we cannot rely on the angle being properly flipped by the keypoint data
                angles[i] = p_next_angle
                intervention = 'low kp algn score, defer to sample'

            elif np.abs(rel_angle_dist) > 140:
                # if predicted angle is completely off from observed angle
                # Here we choose a threshold of 140 degrees, meaning the difference is 180 +/- 40 degrees,
                # or looks approximatly like a flip
                angles[i] = clamp_angles_deg(angles[i] + 180)
                flips[i] = ~flips[i]
                intervention = 'flip 180'

            #elif np.abs(rel_angle_dist) > 15:
            #    angles[i] = clamp_angles(angles[i] - rel_angle_dist)
            #    intervention = f'nudge by {rel_angle_dist}'

            else:
                intervention = None

            rel_angle_dist2 = angle_difference(p_next_angle, angles[[i]])[0]

            # finally update kalman filter with our (potentially) corrected current angle
            t_angle, = angle_tracker.filter_update([angles[[i]]])

            if DEBUG:
                flip_info.append({
                    'i': i,
                    'aspect': aspects[i],
                    'kpt_flip_opinion': flips[i],
                    'kpt_flip_conf': flip_confs[i],
                    'kpt_align_score': kpt_alignment_scores[i],
                    'kpt_rotation': kpt_rotations[i],
                    'angle_in': orig_angles[i],
                    'post_kp_flip_angle': post_kp_flip_angles[i],
                    'sample_angle': p_next_angle[0],
                    'filt_angle': t_angle[0],
                    'rel_angle_dist': rel_angle_dist,
                    'rel_angle_dist2': rel_angle_dist2,
                    'intervention': intervention,
                    'angle_out':angles[i]
                })

        features['orientation'] = np.array(angles)

        if DEBUG:
            # dump this data into a file in the cwd. let's not worry too much about it now,
            # but this is the easiset way to just get data out for inspection.
            flip_info_df = pd.DataFrame(flip_info)
            flip_info_df.to_csv(find_unused_file_path('flip_info.tsv'), sep='\t', index=False)
    else:
        # STRATEGY (NOT tracking-based):
        # 1) use keypoints to flip angles
        # 2) perform iterative angle filtering (look for jumps in the range of 180)

        # Get and apply flips using keypoint information
        flips, _ = flips_from_keypoints(allocentric_keypoints[:,0,...], features['centroid'], angles, lengths)
        angles[flips] += 180

        # apply iterative filter on angle values
        angles, filter_flips = iterative_filter_angles(angles)
        features['orientation'] = np.array(angles)
        flips = np.logical_xor(flips, filter_flips)

    return {
        'cleaned_frames': cleaned_frames,
        'masks': masks,
        'features': features,
        'flips': flips,
        'keypoints': allocentric_keypoints[:,0,...],
        'num_instances': num_instances
    }


def flips_from_keypoints(keypoints: np.ndarray, centroids: np.ndarray, angles: np.ndarray, length: float=80) -> Tuple[np.ndarray, np.ndarray]:
    ''' Estimate flips given keypoints, centroids, angles, and lengths

    Parameters:
    keypoints (np.ndarray): keypoint data, of shape (nframes, nkeypoints, 3 [x, y, s])
    centroids (np.ndarray): centroid data, of shape (nframes, 2 [x, y])
    angles (np.ndarray): angle data, of shape (nframes,)
    length (np.ndarray): length data, of shape (nframes,)
    '''
    front_keypoints = [0, 1, 2, 3]
    rear_keypoints = [4, 5, 6]

    # Rotate keypoints to reflect angles
    rotated_keypoints = rotate_points_batch(np.copy(keypoints), centroids, angles)

    # Strategy:
    # Compute the distance of each keypoint to the left and right edge of the bounding box
    # The groups of front and rear keypoints vote on which edge they are closer to (left=-1; right=1)
    # The votes are compared, and if indicate a flip is needed, add 180 degrees to the angle
    extent_x_min = centroids[:, 0] - (length / 2)
    extent_x_max = centroids[:, 0] + (length / 2)
    rot_keypoint_scores = np.zeros(rotated_keypoints.shape[:-1], dtype=float)
    left_dist = np.abs(extent_x_min[:, np.newaxis] - rotated_keypoints[:, :, 0])
    right_dist = np.abs(extent_x_max[:, np.newaxis] - rotated_keypoints[:, :, 0])
    rot_keypoint_scores = np.where(left_dist < right_dist, -1, 1)
    front_votes = np.mean(rot_keypoint_scores[:, front_keypoints], axis=1)
    rear_votes = np.mean(rot_keypoint_scores[:, rear_keypoints], axis=1)
    flips = np.where(front_votes < rear_votes, True, False)

    # compute a confidence score
    # essentially, the proportion of all keypoints which agree with the final call of Flip/NoFlip
    # since it is majority voting, scores will vary between [0.5, 1.0]
    expected = np.where(flips[:, None], np.array([-1, 1]), np.array([1, -1]))
    agree = np.count_nonzero(rot_keypoint_scores[:, front_keypoints] == expected[:, 0, None], axis=1) \
          + np.count_nonzero(rot_keypoint_scores[:, rear_keypoints] == expected[:, 1, None], axis=1)
    total = len(front_keypoints) + len(rear_keypoints)
    conf_scores = agree / total

    return flips, conf_scores


def estimate_keypoint_rotation(keypoints: np.ndarray) -> np.ndarray:
    '''Estimate the relative rotation between subsequent sets of keypoints

    Parameters:
    keypoints (np.ndarray): keypoints, of shape (nframes, nkeypoints, at least 2)

    Returns:
    estimated rotation of object between subsequent frames, given keypoints
    '''
    angles = np.arctan2(keypoints[..., 1], keypoints[..., 0])
    angles = clamp_angles_deg(np.rad2deg(angles))
    angles = np.diff(angles, axis=0, prepend=angles[0,None,...])
    angles = angles % 360
    to_min = angles > 180
    angles[to_min] = -(360 - angles[to_min])
    return np.median(angles, axis=1)


def calc_keypoint_keypoint_distance(keypoints: np.ndarray, metric: str = 'x') -> np.ndarray:
    '''Calculate a distance matrix for each keypoint against others

    Parameters:
    kaypoints (np.ndarray): array of keypoint data, of shape (nkeypoints, [at least 2: x, y]) or (nframes, nkeypoints, [at least 2: x, y])
    metric (str): one of {euclidean, x, y}, type of distance to calculate
    '''
    if len(keypoints.shape) == 3:
        dist = np.ndarray((keypoints.shape[0], keypoints.shape[1], keypoints.shape[1]), dtype=float)
    else:
        dist = np.ndarray((keypoints.shape[0], keypoints.shape[0]), dtype=float)

    for i in range(dist.shape[-1]):
        for j in range(dist.shape[-1]):
            if metric == 'euclidean':
                dist[:, i, j] = np.sqrt(((keypoints[:, i, 0] - keypoints[:, j, 0])**2) + ((keypoints[:, i, 1] - keypoints[:, j, 1])**2))

            elif metric == 'x':
                dist[..., i, j] = keypoints[..., i, 0] - keypoints[..., j, 0]

            elif metric == 'y':
                dist[..., i, j] = keypoints[..., i, 1] - keypoints[..., j, 1]

    return dist


def compute_keypoint_alignment_scores(keypoints, expected_alignment=None):
    '''Compute a score indicating how well keypoints match an expected alignment
    '''
    if expected_alignment is None:
        expected_alignment = get_expected_keypoint_alignment()

    # calculate keypoint-keypoint distances, and take only the signs
    distances = calc_keypoint_keypoint_distance(keypoints)
    distance_signs = np.sign(distances)

    # mask to zeros keypoint pairs for which we do not have strong expectations
    masked_distance_signs = np.where(expected_alignment == 0, 0, distance_signs)

    # compute number of keypoint pairs which agree with our expectations, subtracting those which we do not have strong expectations for
    axis = (1, 2) if len(keypoints.shape) == 3 else None
    num_expectations_met = np.count_nonzero(masked_distance_signs == expected_alignment, axis=axis) - np.count_nonzero(expected_alignment == 0)

    # generate scores by normalizing to the number of keypoint pairs for which we have a strong expectation
    scores = num_expectations_met / np.count_nonzero(expected_alignment)

    # finally return scores
    return scores


def get_expected_keypoint_alignment():
    '''Get a matrix with the default expected alignment
    '''
    # construct an array with the expected signs
    # positive values mean the row node expects to be to the EAST of the column node
    # negative values mean the row node expects to be to the WEST of the column node
    # zeros indicate we have no particular expectation of the east-west relation, i.e. they should be approximatly the same position on the EAST-WEST axis

    # 	Nose	LeftEar	RightEar	Neck	LeftHip	RightHip	TailBase
    # Nose	0	1	1	1	1	1	1
    # LeftEar	-1	0	0	1	1	1	1
    # RightEar	-1	0	0	1	1	1	1
    # Neck	-1	-1	-1	0	1	1	1
    # LeftHip	-1	-1	-1	-1	0	0	1
    # RightHip	-1	-1	-1	-1	0	0	1
    # TailBase	-1	-1	-1	-1	-1	-1	0

    return np.array([
        [ 0,  1,  1,  1,  1,  1,  1],
        [-1,  0,  0,  1,  1,  1,  1],
        [-1,  0,  0,  1,  1,  1,  1],
        [-1, -1, -1,  0,  1,  1,  1],
        [-1, -1, -1, -1,  0,  0,  1],
        [-1, -1, -1, -1,  0,  0,  1],
        [-1, -1, -1, -1, -1, -1,  0]
    ])


def interpolate_nan_values(data: np.ndarray) -> np.ndarray:
    ''' Interpolate NaN values in `data` using linear interpolation
    '''
    nans = np.isnan(data)
    x = lambda z: z.nonzero()[0]
    data[nans]= np.interp(x(nans), x(~nans), data[~nans])
    return data
