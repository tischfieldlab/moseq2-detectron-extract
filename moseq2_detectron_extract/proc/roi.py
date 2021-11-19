from typing import Union

import cv2
import numpy as np
import scipy
import skimage
import tqdm
from moseq2_detectron_extract.io.video import (get_raw_info, get_video_info,
                                               read_frames, read_frames_raw)


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
    '''
    Get an ROI using RANSAC plane fitting and simple blob features
    '''

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
    '''Fit a plane to 3 points (min number of points for fitting a plane)

    Parameters:
    points (2d numpy array): each row is a group of points,
    columns correspond to x,y,z

    Returns:
    plane (1d numpy array): linear plane fit-->a*x+b*y+c*z+d

    '''
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


def plane_ransac(depth_image, depth_range=(650, 750), iters=1000, noise_tolerance=30, in_ratio=0.1, progress_bar=True, mask=None):
    ''' Naive RANSAC implementation for plane fitting

    Parameters:
    depth_image (2d numpy array): hxw, background image to fit plane to
    depth_range (tuple): min/max depth (mm) to consider pixels for plane
    iters (int): number of RANSAC iterations
    noise_tolerance (float): dist. from plane to consider a point an inlier
    in_ratio (float): frac. of points required to consider a plane fit good

    Returns:
    best_plane (1d numpy array): plane fit to data
    '''
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

    for i in tqdm.tqdm(range(iters), disable=not progress_bar, leave=False, desc='Finding plane'):

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


def apply_roi(frames: np.ndarray, roi: np.ndarray) -> np.ndarray:
    ''' Apply ROI to data:
        1) mask `frames` according to mask `roi`
        2) crop `frames` to the bounding box of `roi`

    consider adding constraints (e.g. mod32==0)

    Parameters:
    frames (np.ndarray): frame data of shape (nframes, rows, cols) to crop and mask
    roi (np.ndarray): 2D mask indicating region of interest

    Returns:
    3D np.ndarray containing masked and cropped frames
    '''
    # yeah so fancy indexing slows us down by 3-5x
    if len(frames.shape) == 3: # (N, W, H)
        frames = frames * roi

    bbox = get_bbox(roi)
    return frames[:, bbox[0, 0]:bbox[1, 0], bbox[0, 1]:bbox[1, 1]]


def get_bbox(roi: np.ndarray) -> Union[np.array, None]:
    ''' Given a binary mask, return an array with the x and y boundaries

    Parameters:
    roi (np.ndarray): 2D mask representing region of interest

    Returns:
    bounding box (np.array) of the form ((y_min, x_min), (y_max, x_max)).
    If no elements of the roi contain non-zero values, `None` will be returned.
    '''
    y, x = np.where(roi > 0)

    if len(y) == 0 or len(x) == 0:
        return None
    else:
        return np.array([[y.min(), x.min()], [y.max(), x.max()]])


def get_roi_contour(roi: np.ndarray, crop: bool=True) -> np.ndarray:
    ''' Get a contour describing the boundry of the roi.

    Parameters:
    roi (np.ndarray): region of interest mask
    crop (bool): if True, crop to the bounding box of the roi before finding contours

    Returns:
    contours, np.ndarray describing contour points
    '''
    if crop:
        bbox = get_bbox(roi)
        mask = roi[bbox[0, 0]:bbox[1, 0], bbox[0, 1]:bbox[1, 1]]
    else:
        mask = np.copy(roi)
    contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return contours


def get_bground_im(frames: np.ndarray) -> np.ndarray:
    ''' Returns background

    Parameters:
    frames (3d numpy array): frames x r x c, uncropped mouse

    Returns:
    bground (2d numpy array):  r x c, background image
    '''
    bground = np.median(frames, axis=0)
    return bground


def get_bground_im_file(frames_file: str, frame_stride: int=500, med_scale: int=5, **kwargs) -> np.ndarray:
    ''' Returns background from file

    Parameters:
    frames_file (path): path to data with frames
    frame_stride (int): steps between frames to consider for background
    med_scale (int): size of the median blur operation, must be odd and greater than 1
    **kwargs: kwargs passed to read_frames_raw()

    Returns:
    bground (2d numpy array):  shape (r x c), background image
    '''

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
