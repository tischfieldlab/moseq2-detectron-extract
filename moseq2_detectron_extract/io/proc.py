import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy
import skimage
import tqdm
from bottleneck import move_median
from moseq2_detectron_extract.io.video import (get_raw_info, get_video_info,
                                               read_frames, read_frames_raw)


def overlay_video(video1, video2):
    channels = video1.shape[-1]
    nframes, rows1, cols1 = video1.shape[:3]
    _, rows2, cols2 = video2.shape[:3]
    output_movie = np.zeros((nframes, (rows1 + rows2), (cols1 + cols2), channels), 'uint16')
    output_movie[:, :rows2, :cols2, :] = video2
    output_movie[:, rows2:, cols2:, :] = video1
    return output_movie


def colorize_video(frames, vmin=0, vmax=100, cmap='jet'):
    use_cmap = plt.get_cmap(cmap)

    disp_img = frames.copy().astype('float32')
    disp_img = (disp_img-vmin)/(vmax-vmin)
    disp_img[disp_img < 0] = 0
    disp_img[disp_img > 1] = 1
    disp_img = np.delete(use_cmap(disp_img), 3, 2)*255

    return disp_img


def get_bbox(roi):
    """
    Given a binary mask, return an array with the x and y boundaries
    """
    y, x = np.where(roi > 0)

    if len(y) == 0 or len(x) == 0:
        return None
    else:
        bbox = np.array([[y.min(), x.min()], [y.max(), x.max()]])
        return bbox

def get_bground_im(frames):
    """Returns background
    Args:
        frames (3d numpy array): frames x r x c, uncropped mouse

    Returns:
        bground (2d numpy array):  r x c, background image
    """
    bground = np.median(frames, 0)
    return bground


def get_bground_im_file(frames_file, frame_stride=500, med_scale=5, **kwargs):
    """Returns background from file
    Args:
        frames_file (path): path to data with frames
        frame_stride

    Returns:
        bground (2d numpy array):  r x c, background image
    """

    try:
        if frames_file.endswith('dat'):
            finfo = get_raw_info(frames_file)
        elif frames_file.endswith('avi'):
            finfo = get_video_info(frames_file)
    except AttributeError as e:
        finfo = get_raw_info(frames_file)

    frame_idx = np.arange(0, finfo['nframes'], frame_stride)
    frame_store = np.zeros((len(frame_idx), finfo['dims'][1], finfo['dims'][0]))

    for i, frame in enumerate(frame_idx):
        try:
            if frames_file.endswith('dat'):
                frs = read_frames_raw(frames_file, int(frame)).squeeze()
            elif frames_file.endswith('avi'):
                frs = read_frames(frames_file, [int(frame)]).squeeze()
        except AttributeError as e:
            frs = read_frames_raw(frames_file, int(frame), **kwargs).squeeze()

        frame_store[i, ...] = cv2.medianBlur(frs, med_scale)

    return get_bground_im(frame_store)

def select_strel(string='e', size=(10, 10)):
    if string[0].lower() == 'e':
        strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, size)
    elif string[0].lower() == 'r':
        strel = cv2.getStructuringElement(cv2.MORPH_RECT, size)
    else:
        strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, size)

    return strel


def get_roi(depth_image,
            strel_dilate=cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)),
            strel_erode=None,
            noise_tolerance=30,
            weights=(1, .1, 1),
            overlap_roi=None,
            gradient_filter=False,
            gradient_kernel=7,
            gradient_threshold=3000,
            fill_holes=True,
            **kwargs):
    """
    Get an ROI using RANSAC plane fitting and simple blob features
    """

    if gradient_filter:
        gradient_x = np.abs(cv2.Sobel(depth_image, cv2.CV_64F,
                                      1, 0, ksize=gradient_kernel))
        gradient_y = np.abs(cv2.Sobel(depth_image, cv2.CV_64F,
                                      0, 1, ksize=gradient_kernel))
        mask = np.logical_and(gradient_x < gradient_threshold, gradient_y < gradient_threshold)
    else:
        mask = None

    roi_plane, dists = plane_ransac(
        depth_image, noise_tolerance=noise_tolerance, mask=mask, **kwargs)
    dist_ims = dists.reshape(depth_image.shape)

    if gradient_filter:
        dist_ims[~mask] = np.inf

    bin_im = dist_ims < noise_tolerance

    # anything < noise_tolerance from the plane is part of it

    label_im = skimage.measure.label(bin_im)
    region_properties = skimage.measure.regionprops(label_im)

    areas = np.zeros((len(region_properties),))
    extents = np.zeros_like(areas)
    dists = np.zeros_like(extents)

    # get the max distance from the center, area and extent

    center = np.array(depth_image.shape)/2

    for i, props in enumerate(region_properties):
        areas[i] = props.area
        extents[i] = props.extent
        tmp_dists = np.sqrt(np.sum(np.square(props.coords-center), 1))
        dists[i] = tmp_dists.max()

    # rank features

    ranks = np.vstack((scipy.stats.rankdata(-areas, method='max'),
                       scipy.stats.rankdata(-extents, method='max'),
                       scipy.stats.rankdata(dists, method='max')))
    weight_array = np.array(weights, 'float32')
    shape_index = np.mean(np.multiply(ranks.astype('float32'), weight_array[:, np.newaxis]), 0).argsort()

    # expansion microscopy on the roi

    rois = []
    bboxes = []

    for shape in shape_index:
        roi = np.zeros_like(depth_image)
        roi[region_properties[shape].coords[:, 0],
            region_properties[shape].coords[:, 1]] = 1
        if strel_dilate is not None:
            roi = cv2.dilate(roi, strel_dilate, iterations=1)
        if strel_erode is not None:
            roi = cv2.erode(roi, strel_erode, iterations=1)
        if fill_holes:
            roi = scipy.ndimage.morphology.binary_fill_holes(roi)

        # roi=skimage.morphology.dilation(roi,dilate_element)
        rois.append(roi)
        bboxes.append(get_bbox(roi))

    if overlap_roi is not None:
        overlaps = np.zeros_like(areas)

        for i in range(len(rois)):
            overlaps[i] = np.sum(np.logical_and(overlap_roi, rois[i]))

        del_roi = np.argmax(overlaps)
        del rois[del_roi]
        del bboxes[del_roi]

    return rois, roi_plane, bboxes, label_im, ranks, shape_index


def plane_fit3(points):
    """Fit a plane to 3 points (min number of points for fitting a plane)
    Args:
        points (2d numpy array): each row is a group of points,
        columns correspond to x,y,z

    Returns:
        plane (1d numpy array): linear plane fit-->a*x+b*y+c*z+d

    """
    a = points[1, :]-points[0, :]
    b = points[2, :]-points[0, :]
    # cross prod
    normal = np.array([[a[1]*b[2]-a[2]*b[1]],
                       [a[2]*b[0]-a[0]*b[2]],
                       [a[0]*b[1]-a[1]*b[0]]])
    denom = np.sum(np.square(normal))
    if denom < np.spacing(1):
        plane = np.empty((4,))
        plane[:] = np.nan
    else:
        normal /= np.sqrt(denom)
        d = np.dot(-points[0, :], normal)
        plane = np.hstack((normal.flatten(), d))

    return plane


def plane_ransac(depth_image, depth_range=(650, 750), iters=1000,
                 noise_tolerance=30, in_ratio=.1, progress_bar=True,
                 mask=None):
    """Naive RANSAC implementation for plane fitting
    Args:
        depth_image (2d numpy array): hxw, background image to fit plane to
        depth_range (tuple): min/max depth (mm) to consider pixels for plane
        iters (int): number of RANSAC iterations
        noise_tolerance (float): dist. from plane to consider a point an inlier
        in_ratio (float): frac. of points required to consider a plane fit good

    Returns:
        best_plane (1d numpy array): plane fit to data
    """
    use_points = np.logical_and(
        depth_image > depth_range[0], depth_image < depth_range[1])

    if mask is not None:
        use_points = np.logical_and(use_points, mask)

    xx, yy = np.meshgrid(
        np.arange(depth_image.shape[1]), np.arange(depth_image.shape[0]))

    coords = np.vstack(
        (xx[use_points].ravel(), yy[use_points].ravel(),
         depth_image[use_points].ravel()))
    coords = coords.T

    best_dist = np.inf
    best_num = 0

    npoints = np.sum(use_points)

    for i in tqdm.tqdm(range(iters),
                       disable=not progress_bar, desc='Finding plane'):

        sel = coords[np.random.choice(coords.shape[0], 3, replace=True), :]
        tmp_plane = plane_fit3(sel)

        if np.all(np.isnan(tmp_plane)):
            continue

        dist = np.abs(np.dot(coords, tmp_plane[:3])+tmp_plane[3])
        inliers = dist < noise_tolerance
        ninliers = np.sum(inliers)

        if ((ninliers/npoints) > in_ratio
                and ninliers > best_num and np.mean(dist) < best_dist):

            best_dist = np.mean(dist)
            best_num = ninliers
            best_plane = tmp_plane

            # use all consensus samples to fit a better model

            # all_data=coords[inliers.flatten(),:]
            # mu=np.mean(all_data,0)
            # u,s,v=np.linalg.svd(all_data-mu)
            # v=v.conj().T
            # best_plane=np.hstack((v[:,-1],np.dot(-mu,v[:,-1])))

    # fit the plane to our x,y,z coordinates

    # if we have lots of datapoints this is unnecessarily expensive,
    # could rsample here too
    #
    # dist=np.abs(np.dot(coords,best_plane[:3])+best_plane[3])
    # inliers=dist<noise_tolerance
    #
    # all_data=coords[inliers.flatten(),:]
    # mu=np.mean(all_data,0)
    # u,s,v=np.linalg.svd(all_data-mu)
    # v=v.conj().T
    # best_plane=np.hstack((v[:,-1],np.dot(-mu,v[:,-1])))

    coords = np.vstack((xx.ravel(), yy.ravel(), depth_image.ravel())).T
    dist = np.abs(np.dot(coords, best_plane[:3])+best_plane[3])

    return best_plane, dist


def apply_roi(frames, roi):
    """
    Apply ROI to data, consider adding constraints (e.g. mod32==0)
    """
    # yeah so fancy indexing slows us down by 3-5x
    if len(frames.shape) == 3: # (N, W, H)
        frames = frames*roi
    
    bbox = get_bbox(roi)
    return frames[:, bbox[0, 0]:bbox[1, 0], bbox[0, 1]:bbox[1, 1]]



def get_frame_features(frames, frame_threshold=10, mask=np.array([]),
                       mask_threshold=-30, use_cc=False, progress_bar=True):
    """
    Use image moments to compute features of the largest object in the frame

    Args:
        frames (3d np array)
        frame_threshold (int): threshold in mm separating floor from mouse

    Returns:
        features (dict list): dictionary with simple image features

    """

    features = []
    nframes = frames.shape[0]

    if type(mask) is np.ndarray and mask.size > 0:
        has_mask = True
    else:
        has_mask = False
        mask = np.zeros((frames.shape), 'uint8')

    features = {
        'centroid': np.empty((nframes, 2)),
        'orientation': np.empty((nframes,)),
        'axis_length': np.empty((nframes, 2))
    }

    for k, v in features.items():
        features[k][:] = np.nan

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

    return features, mask



def crop_and_rotate_frame(frame, origin, angle, crop_size=(80, 80)):
    if np.isnan(angle) or np.any(np.isnan(origin)):
        return np.zeros_like(frame, shape=crop_size)

    xmin = int(origin[0] - crop_size[1] // 2) + crop_size[1]
    xmax = int(origin[0] + crop_size[1] // 2) + crop_size[1]
    ymin = int(origin[1] - crop_size[0] // 2) + crop_size[0]
    ymax = int(origin[1] + crop_size[0] // 2) + crop_size[0]

    border = (crop_size[1], crop_size[1], crop_size[0], crop_size[0])
    rot_mat = cv2.getRotationMatrix2D((crop_size[0] // 2, crop_size[1] // 2), angle, 1)
    use_frame = cv2.copyMakeBorder(frame, *border, cv2.BORDER_CONSTANT, 0)
    return cv2.warpAffine(use_frame[ymin:ymax, xmin:xmax], rot_mat, (crop_size[0], crop_size[1]))



def crop_and_rotate_frames(frames, features, crop_size=(80, 80),
                           progress_bar=True):

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

def hampel_filter_forloop(input_series, window_size, n_sigmas=3):
    # https://towardsdatascience.com/outlier-detection-with-hampel-filter-85ddf523c73d
    n = len(input_series)
    new_series = input_series.copy()
    k = 1.4826 # scale factor for Gaussian distribution
    
    indices = []
    
    # possibly use np.nanmedian 
    for i in range((window_size),(n - window_size)):
        x0 = np.nanmedian(input_series[(i - window_size):(i + window_size)])
        S0 = k * np.nanmedian(np.abs(input_series[(i - window_size):(i + window_size)] - x0))
        if (np.abs(input_series[i] - x0) > n_sigmas * S0):
            new_series[i] = x0
            indices.append(i)
    
    return new_series, indices

def clean_frames(frames, prefilter_space=(3,), prefilter_time=None,
                 strel_tail=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
                 iters_tail=None, frame_dtype='uint8',
                 strel_min=cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
                 iters_min=None, progress_bar=True):
    """
    Simple filtering, median filter and morphological opening

    Args:
        frames (3d np array): frames x r x c
        strel (opencv structuring element): strel for morph opening
        iters_tail (int): number of iterations to run opening

    Returns:
        filtered_frames (3d np array): frame x r x c

    """
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
            filtered_frames[i, ...] = cv2.morphologyEx(
                filtered_frames[i, ...], cv2.MORPH_OPEN, strel_tail, iters_tail)

    if prefilter_time is not None and np.all(np.array(prefilter_time) > 0):
        for j in range(len(prefilter_time)):
            filtered_frames = scipy.signal.medfilt(
                filtered_frames, [prefilter_time[j], 1, 1])

    return filtered_frames

def im_moment_features(IM):
    """
    Use the method of moments and centralized moments to get image properties

    Args:
        IM (2d numpy array): depth image

    Returns:
        Features (dictionary): returns a dictionary with orientation,
        centroid, and ellipse axis length

    """

    tmp = cv2.moments(IM)
    num = 2*tmp['mu11']
    den = tmp['mu20']-tmp['mu02']

    common = np.sqrt(4*np.square(tmp['mu11'])+np.square(den))

    if tmp['m00'] == 0:
        features = {
            'orientation': np.nan,
            'centroid': np.nan,
            'axis_length': [np.nan, np.nan]}
    else:
        features = {
            'orientation': -.5*np.arctan2(num, den),
            'centroid': [tmp['m10']/tmp['m00'], tmp['m01']/tmp['m00']],
            'axis_length': [2*np.sqrt(2)*np.sqrt((tmp['mu20']+tmp['mu02']+common)/tmp['m00']),
                            2*np.sqrt(2)*np.sqrt((tmp['mu20']+tmp['mu02']-common)/tmp['m00'])]
        }

    return features

def get_largest_cc(frames, progress_bar=False):
    """Returns largest connected component blob in image
    Args:
        frame (3d numpy array): frames x r x c, uncropped mouse
        progress_bar (bool): display progress bar

    Returns:
        flips (3d bool array):  frames x r x c, true where blob was found
    """
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

def weighted_centroid_points(xs, ys, ws):
    x = np.sum(xs * ws) / np.sum(ws)
    y = np.sum(ys * ws) / np.sum(ws)
    return (x, y)

def weighted_centroid_points2(points):
    x = np.sum(points[:, 0] * points[:, 2]) / np.sum(points[:, 2])
    y = np.sum(points[:, 1] * points[:, 2]) / np.sum(points[:, 2])
    return (x, y)

def sort_keypoints(keypoints, axis=0):
    ''' Sorts a keypoints array (N kyepoints x 2 [x, y, possibly score]) by x-position
    '''
    return keypoints[:, np.argsort(keypoints[axis])]

def rotate_points(points, origin=(0, 0), degrees=0):
    if points.shape[1] == 3:
        weights = points[:, 2]
        points = points[:, :2]
    else:
        weights = None
    
    angle = np.deg2rad(-degrees)
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    o = np.atleast_2d(origin)
    p = np.atleast_2d(points)
    rotated = np.squeeze((R @ (p.T-o.T) + o.T).T)

    if weights is not None:
        rotated = np.append(rotated, weights[..., None], 1)
    return rotated


def rotate_points2(points, centroids, angles):
    # expects (frames x instances x keypoints x 3 [x,, y, s])
    for i in range(points.shape[0]):
        points[i, 0] = rotate_points(points[i, 0], centroids[i], angles[i])
    return points



def filter_angles2(data):
    out = np.copy(data)
    diff = np.diff(data, prepend=data[0])
    absdiff = np.abs(diff)
    flips = ((absdiff>120) & (absdiff<240))
    signs = np.sign(diff[flips])
    out[flips] = out[flips] + (-180 * signs)
    return out

def iterative_filter_angles(data, max_iters=1000):
    last = np.copy(data)
    i=0
    while True:
        if i > max_iters:
            return curr

        curr = filter_angles2(last)

        if np.allclose(curr, last):
            return curr
        else:
            last = curr
            i += 1


def filter_angles3(angles, window=3):
    out = np.copy(angles)
    windows = move_median(angles, window=window, min_count=1)
    diff = out - windows
    absdiff = np.abs(diff)
    flips = ((absdiff>120) & (absdiff<240))
    signs = np.sign(diff[flips])
    out[flips] = out[flips] + (-180 * signs)
    #print(np.count_nonzero(flips))
    return out

def iterative_filter_angles3(data):
    last = np.copy(data)
    iterations = 0
    while True:
        iterations += 1
        curr = filter_angles3(last)
        
        if np.allclose(curr, last):
            print(f'Converged after {iterations} iterations')
            return curr
        else:
            last = curr


def mask_and_keypoints_from_model_output(model_outputs):
    keypoints = None
    masks = None
    for i, output in enumerate(model_outputs):
        if len(output["instances"]) > 0:
            kpts = output["instances"].pred_keypoints[0, :, :].cpu().numpy()
            if keypoints is None:
                keypoints = np.empty((len(model_outputs), 1, kpts.shape[0], 3), dtype=float)
                keypoints.fill(np.nan)
            keypoints[i, 0] = kpts

            msks = output["instances"].pred_masks[0, :, :].cpu().numpy()
            if masks is None:
                masks = np.empty((len(model_outputs), 1, *msks.shape), dtype='uint8')
            masks[i, 0] = msks
    return masks, keypoints



def instances_to_features(model_outputs, raw_frames):
    front_keypoints = [0, 1, 2, 3]
    rear_keypoints = [4, 5, 6]

    # frames x instances x keypoints x 3
    d2_masks, allosteric_keypoints = mask_and_keypoints_from_model_output(model_outputs)

    cleaned_frames = clean_frames(raw_frames, progress_bar=False, iters_tail=3, prefilter_time=(3,))
    features, masks = get_frame_features(cleaned_frames, progress_bar=False, mask=d2_masks[:, 0, :, :], use_cc=True)

    centroids = features['centroid']
    angles = features['orientation']
    lengths = np.max(features['axis_length'], axis=1)
    incl = ~np.isnan(angles)
    angles[incl] = np.unwrap(angles[incl] * 2) / 2
    angles = -np.rad2deg(angles)

    #rotate keypoints to reflect angles
    rotated_keypoints = rotate_points2(allosteric_keypoints, centroids, angles)




    # Strategy:
    # Compute the distance of each keypoint to the left and right edge of the bounding box
    # The groups of front and rear keypoints vote on which edge they are closer to (left=-1; right=1)
    # The votes are compared, and if indicate a flip is needed, add 180 degrees to the angle
    extent_x_min = centroids[:, 0] - (lengths / 2)
    extent_x_max = centroids[:, 0] + (lengths / 2)
    rot_keypoint_scores = np.zeros(rotated_keypoints.shape[:-1], dtype=float)
    left_dist = np.abs(extent_x_min[:, np.newaxis] - rotated_keypoints[:, 0, :, 0])
    right_dist = np.abs(extent_x_max[:, np.newaxis] - rotated_keypoints[:, 0, :, 0])
    rot_keypoint_scores = np.where(left_dist < right_dist, -1, 1)
    front_votes = np.mean(rot_keypoint_scores[:, front_keypoints], axis=1)
    rear_votes = np.mean(rot_keypoint_scores[:, rear_keypoints], axis=1)
    flips = np.where(front_votes < rear_votes, True, False)
    angles[flips] += 180
    for i, flip in enumerate(flips):
        if flip:
            rotated_keypoints[i, 0] = rotate_points(rotated_keypoints[i, 0], origin=centroids[i], degrees=180)
    #angles = iterative_filter_angles(angles)
    angles = iterative_filter_angles3(angles)

    return cleaned_frames, np.array(angles), np.array(centroids), masks, flips, allosteric_keypoints, rotated_keypoints


    # for kpi in range(rotated_keypoints.shape[0]):
    #     left_dist = np.abs(blob_min_x - rotated_keypoints[kpi, 0])
    #     right_dist = np.abs(blob_max_x - rotated_keypoints[kpi, 0])
    #     if left_dist < right_dist:
    #         rot_keypoint_scores[kpi] = -1
    #     else:
    #         rot_keypoint_scores[kpi] = 1
    # front_votes = np.mean(rot_keypoint_scores[front_keypoints])
    # rear_votes = np.mean(rot_keypoint_scores[rear_keypoints])
    


    # angles = []
    # centroids = []
    # flips = []
    # for i, output in enumerate(model_outputs):
    #     instances = output["instances"].to("cpu")
    #     if len(instances) > 0:

    #         # extract centroid
    #         mask = instances.pred_masks[0,...].numpy() * raw_frames[i,...]
    #         weighted_mask = mask * raw_frames[i,...]
    #         y_center, x_center = np.argwhere(weighted_mask > 0).sum(0)/np.count_nonzero(mask)
            


    #         try:
    #             keypoints = instances.pred_keypoints[0, all_keypoints, :].numpy()

    #             # Approach 1:
    #             #  - Fit a score-weighted 1d poly to all keypoints
    #             #  - Compute angle from slope
    #             # coords = sort_keypoints(keypoints)
    #             # slope, _ = np.polyfit(coords[:, 0], coords[:, 1], 1, w=coords[:, 2])
    #             # angle = np.rad2deg(math.atan(slope))


    #             # Approach 2:
    #             #  - Find score-weighted centroids of front and rear keypoints
    #             #  - Fit 1d poly to front and rear centroids
    #             #  - Compute angle from slope
    #             # front_pos = weighted_centroid_points2(keypoints[front_keypoints])
    #             # rear_pos = weighted_centroid_points2(keypoints[rear_keypoints])
    #             # coords = sort_keypoints(np.array([front_pos, rear_pos]))
    #             # slope, _ = np.polyfit(coords[:, 0], coords[:, 1], 1)
    #             # angle = np.rad2deg(math.atan(slope))


    #             # Approach 3:
    #             #  - Find score-weighted centroids of front and rear keypoints
    #             #  - Fit 1d poly to front, rear centroids, PLUS the whole animal centroid
    #             #  - Compute angle from slope
    #             # front_pos = weighted_centroid_points2(keypoints[front_keypoints])
    #             # rear_pos = weighted_centroid_points2(keypoints[rear_keypoints])
    #             # coords = sort_keypoints(np.array([front_pos, rear_pos, [x_center, y_center]]))
    #             # slope, _ = np.polyfit(coords[:, 0], coords[:, 1], 1)
    #             # angle = np.rad2deg(math.atan(slope))


    #             # Approach 4:
    #             #  - Use image coutours and moment features to determine angle and centroid
    #             angle = angles_from_features[i]
    #             x_center, y_center = centroids_from_features[i]
    #             length = np.max(lengths_from_features[i])


    #             # Sanity check our computed angle using the rotated keypoint coordinates
    #             #rotated_keypoints = rotate_points(keypoints[:], origin=[x_center, y_center], degrees=angle)
    #             #rot_keypoint_scores = np.zeros((rotated_keypoints.shape[0],))
    #             #for kpi in range(rotated_keypoints.shape[0]):
    #             #    if rotated_keypoints[kpi, 0] < x_center:
    #             #        rot_keypoint_scores[kpi] = -1
    #             #    else:
    #             #        rot_keypoint_scores[kpi] = 1
    #             #front_votes = rot_keypoint_scores[front_keypoints][np.argmax(rotated_keypoints[front_keypoints, 2])]
    #             #rear_votes = rot_keypoint_scores[rear_keypoints][np.argmax(rotated_keypoints[rear_keypoints, 2])]


    #             blob_max_x = x_center + (length / 2)
    #             blob_min_x = x_center - (length / 2)
    #             rotated_keypoints = rotate_points(keypoints[:], origin=[x_center, y_center], degrees=angle)
    #             rot_keypoint_scores = np.zeros((rotated_keypoints.shape[0],))
    #             for kpi in range(rotated_keypoints.shape[0]):
    #                 left_dist = np.abs(blob_min_x - rotated_keypoints[kpi, 0])
    #                 right_dist = np.abs(blob_max_x - rotated_keypoints[kpi, 0])
    #                 if left_dist < right_dist:
    #                     rot_keypoint_scores[kpi] = -1
    #                 else:
    #                     rot_keypoint_scores[kpi] = 1
    #             front_votes = np.mean(rot_keypoint_scores[front_keypoints]) #[np.argmax(rotated_keypoints[front_keypoints, 2])]
    #             rear_votes = np.mean(rot_keypoint_scores[rear_keypoints]) #[np.argmax(rotated_keypoints[rear_keypoints, 2])]






    #             #front_votes = stats.mode(rot_keypoint_scores[front_keypoints])
    #             #rear_votes = stats.mode(rot_keypoint_scores[rear_keypoints])
    #             #front_votes = np.median(rot_keypoint_scores[front_keypoints] * rotated_keypoints[front_keypoints, 2])
    #             #rear_votes = np.median(rot_keypoint_scores[rear_keypoints] * rotated_keypoints[rear_keypoints, 2])
    #             #rot_front_pos = weighted_centroid_points2(rotated_keypoints[front_keypoints])
    #             #rot_rear_pos = weighted_centroid_points2(rotated_keypoints[rear_keypoints])

    #             if front_votes < rear_votes:#rot_front_pos[0] < rot_rear_pos[0]:
    #                 # We need to flip the angle
    #                 angle = angle + 180
    #                 flips.append(True)
    #                 #tqdm.tqdm.write("Fixed angle {} in frame {}".format(angle, i))
    #             else:
    #                 flips.append(False)

    #             # norm angle to be positive between 0-360
    #             #angle = angle % 360
    #             #if angle < 0:
    #             #    angle += 360
                
    #         except:
    #             angle = 0
            
    #         angles.append(angle)
    #         centroids.append((x_center, y_center))

    #         # tqdm.tqdm.write("{}\t{}".format(angle, angle2))
    #     else:
    #         angles.append(0)
    #         centroids.append((0,0))
    # return np.array(angles), np.array(centroids), masks, flips
