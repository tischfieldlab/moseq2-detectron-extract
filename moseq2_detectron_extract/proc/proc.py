from typing import Dict, Iterable, List, Literal, Sequence, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import scipy
import tqdm
from bottleneck import move_median
from moseq2_detectron_extract.proc.keypoints import rotate_points_batch
from moseq2_detectron_extract.proc.roi import apply_roi


def stack_videos(videos: Sequence[np.ndarray], orientation: Literal['horizontal', 'vertical', 'diagional']) -> np.ndarray:
    ''' Stack videos according to orientation to create one big video

    Parameters:
    videos (Iterable[np.ndarray]): Iterable of videos to stack of shape (nframes, height, width, channels). All videos must match dimentions in axis 0 and axis 3
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

    sizes = set([d.shape[axis] for d in data])
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

    dtypes = set([d.dtype for d in data])
    if len(dtypes) == 1:
        return dtypes.pop()
    else:
        raise ValueError(f'Arrays should have same dtype!')


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
    disp_img = use_cmap(disp_img)[:,:,:,:3]*255

    return disp_img.astype('uint8')


def prep_raw_frames(frames: np.ndarray, bground_im: np.ndarray=None, roi: np.ndarray=None, vmin: float=None, vmax: float=None, dtype: npt.DTypeLike='uint8'):
    ''' Prepare raw `frames` by:
            1) subtracting background based on `bground_im`
            2) applying a region of interest (crop and mask according to `roi`)
            3) clamping values in the image to `vmin` and `vmax`
                a) values less than `vmin` are set to zero
                b) values greater than `vmax` are set to `vmax`
            All operations are optional.
    '''
    if bground_im is not None:
        frames = bground_im - frames

    if roi is not None:
        frames = apply_roi(frames, roi)

    if vmin is not None:
        frames[frames < vmin] = 0

    if vmax is not None:
        frames[frames > vmax] = vmax

    return frames.astype(dtype)


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
    if np.issubdtype(np.dtype(dtype), np.integer):
        dmin = np.iinfo(dtype).min
        dmax = np.iinfo(dtype).max
    else:
        dmin = np.finfo(dtype).min
        dmax = np.finfo(dtype).max

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

    if type(mask) is np.ndarray and mask.size > 0:
        has_mask = True
    else:
        has_mask = False
        mask = np.zeros((frames.shape), 'uint8')

    features: dict = {
        'centroid': np.empty((nframes, 2)),
        'orientation': np.empty((nframes,)),
        'axis_length': np.empty((nframes, 2)),
    }

    for k, v in features.items():
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

        cnts, hierarchy = cv2.findContours(
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


def crop_and_rotate_frame(frame: np.ndarray, center: Tuple[float, float], angle: float, crop_size: Tuple[int, int]=(80, 80)):
    if np.isnan(angle) or np.any(np.isnan(center)):
        return np.zeros_like(frame, shape=crop_size)

    xmin = int(center[0] - crop_size[0] // 2) + crop_size[0]
    xmax = int(center[0] + crop_size[0] // 2) + crop_size[0]
    ymin = int(center[1] - crop_size[1] // 2) + crop_size[1]
    ymax = int(center[1] + crop_size[1] // 2) + crop_size[1]

    border = (crop_size[1], crop_size[1], crop_size[0], crop_size[0])
    rot_mat = cv2.getRotationMatrix2D((crop_size[0] // 2, crop_size[1] // 2), angle, 1)
    use_frame = cv2.copyMakeBorder(frame, *border, cv2.BORDER_CONSTANT, 0)
    return cv2.warpAffine(use_frame[ymin:ymax, xmin:xmax], rot_mat, (crop_size[0], crop_size[1]))


def crop_and_rotate_frames(frames: np.ndarray, features: dict, crop_size: Tuple[int, int]=(80, 80),
                           progress_bar: bool=True):

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
        raise ValueError("cannot accept data with {} dimentions!".format(len(data.shape)))

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
            for j in range(len(prefilter_space)):
                filtered_frames[i, ...] = cv2.medianBlur(filtered_frames[i, ...], prefilter_space[j])

        if iters_tail is not None and iters_tail > 0:
            filtered_frames[i, ...] = cv2.morphologyEx(filtered_frames[i, ...], cv2.MORPH_OPEN, strel_tail, iters_tail)

    if prefilter_time is not None and np.all(np.array(prefilter_time) > 0) and np.all(np.array(prefilter_time) <= filtered_frames.shape[0]):
        for j in range(len(prefilter_time)):
            filtered_frames = scipy.signal.medfilt(filtered_frames, [prefilter_time[j], 1, 1])

    return filtered_frames


def im_moment_features(IM: np.ndarray) -> dict:
    ''' Use the method of moments and centralized moments to get image properties

    Parameters:
    IM (2d numpy array): depth image

    Returns:
    Features (dictionary): returns a dictionary with orientation,
    centroid, and ellipse axis length
    '''

    tmp = cv2.moments(IM)
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
        nb_components, output, stats, centroids =\
            cv2.connectedComponentsWithStats(frames[i, ...], connectivity=4)
        szs = stats[:, -1]
        foreground_obj[i, ...] = output == szs[1:].argmax()+1

    return foreground_obj


# from https://stackoverflow.com/questions/40084931/taking-subarrays-from-numpy-array-with-given-stride-stepsize/40085052#40085052
# dang this is fast!
def strided_app(a, L, S):  # Window len = L, Stride len/stepsize = S
    nrows = ((a.size-L)//S)+1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a, shape=(nrows, L), strides=(S*n, n))


def broadcasting_app(a, L, S ):  # Window len = L, Stride len/stepsize = S
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


def iterative_filter_angles(data: np.ndarray, max_iters: int=1000) -> Tuple[np.ndarray, np.ndarray]:
    ''' Iteratively filter angles until filtering stabilizes or `max_iters` is reached

    Parameters:
    angles (np.ndarray): angles to inspect, of shape (nframes,)
    max_iters (int): maximum number of iterations allowed

    Returns:
    (angles, flips) - angles are corrected angles. flips is bool np.ndarray with True values indicating a flipped index
    '''
    last = np.copy(data)
    iterations = 0
    while True:
        if iterations > max_iters:
            break

        iterations += 1
        curr = filter_angles(last)

        if np.allclose(curr, last):
            # logging.debug(f'Converged after {iterations} iterations')
            break
        else:
            last = curr
    flips = np.isclose(np.abs(curr - data), 180)
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
    masks = np.empty((len(model_outputs), 1, *first.pred_masks.shape[1:]), dtype='uint8')
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


def instances_to_features(model_outputs: List[dict], raw_frames: np.ndarray):
    ''' Detect additional features and perform feature postprocessing

    Parameters:
    model_outputs (List[dict]): Model output data in Detectron2 output format
    raw_frames (np.ndarray): raw depth frames of shape (nframes, height, width)

    Returns:
    Dict[str, Any] - dict containing features and cleaned data
    '''
    front_keypoints = [0, 1, 2, 3]
    rear_keypoints = [4, 5, 6]

    # frames x instances x keypoints x 3
    d2_masks, allocentric_keypoints, num_instances = mask_and_keypoints_from_model_output(model_outputs)

    cleaned_frames = clean_frames(raw_frames, progress_bar=False, iters_tail=3)
    features, masks = get_frame_features(cleaned_frames, progress_bar=False, mask=d2_masks[:, 0, :, :], use_cc=True, frame_threshold=3)

    angles = features['orientation']
    lengths = np.max(features['axis_length'], axis=1)
    incl = ~np.isnan(angles)
    angles[incl] = np.unwrap(angles[incl] * 2) / 2
    angles = -np.rad2deg(angles)

    # Get and apply flips using keypoint information
    flips = flips_from_keypoints(allocentric_keypoints[:,0,...], features['centroid'], angles, lengths)
    angles[flips] += 180
    angles, filter_flips = iterative_filter_angles(angles)
    features['orientation'] = np.array(angles)
    flips = np.logical_xor(flips, filter_flips)

    return {
        'cleaned_frames': cleaned_frames,
        'masks': masks,
        'features': features,
        'flips': flips,
        'keypoints': allocentric_keypoints,
        'num_instances': num_instances
    }
