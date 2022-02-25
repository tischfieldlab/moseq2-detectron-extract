import json
import logging
import os
import random
from typing import (Callable, Dict, Iterable, List, MutableSequence, Optional,
                    Sequence, Tuple, Union)

import numpy as np
import pycocotools
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from moseq2_detectron_extract.io.image import read_image
from skimage.draw import polygon
from tqdm import tqdm
from typing_extensions import Literal, TypedDict


class Annotation(TypedDict):
    bbox: Sequence[float]
    bbox_mode: BoxMode
    category_id: int
    segmentation: Union[Sequence[Sequence[float]], dict]
    keypoints: Sequence[float]


class DataItemBase(TypedDict):
    file_name: str
    width: int
    height: int
    image_id: str
    rescale_intensity: float
    annotations: Sequence[Annotation]


class DataItem(DataItemBase, total=False):
    rotate: int


MaskFormat = Literal['polygon', 'bitmask']


default_keypoint_names = [
    'Nose',
    'Left Ear',
    'Right Ear',
    'Neck',
    'Left Hip',
    'Right Hip',
    'TailBase',
    'TailTip'
]

default_keypoint_colors = [
    (255, 255, 153), # Nose
    (166, 206, 227), # Left Ear
    ( 31, 120, 180), # Right Ear
    (255, 255, 153), # Neck
    (178, 223, 138), # Left Hip
    ( 51, 160,  44), # Right Hip
    (227,  26,  28), # TailBase
    (251, 154, 153)  # TailTip
]

KeypointConnections = Sequence[Tuple[str, str, Tuple[int, int, int]]]
default_keypoint_connection_rules: KeypointConnections = [
        ('Nose', 'Left Ear', (166, 206, 227)),
        ('Nose', 'Right Ear', (31,  120, 180)),
        ('Neck', 'Left Ear', (166, 206, 227)),
        ('Neck', 'Right Ear', (31,  120, 180)),
        ('Neck', 'Left Hip', (178, 223, 138)),
        ('Neck', 'Right Hip', (51,  160, 44 )),
        ('TailBase', 'Left Hip', (178, 223, 138)),
        ('TailBase', 'Right Hip', (51,  160, 44 )),
        ('TailBase', 'TailTip', (251, 154, 153)),
    ]


def get_dataset_statistics(dset: Sequence[DataItem]):
    ''' Calculate mean a standard deviation of images over a dataset.
    
    Parameters:
    dest (Sequence[DataItem]): annotations to compute statistics on

    Returns:
    Tuple(np.ndarray, np.ndarray), tuple of (mean, stdev), each an array indexed by channel
    '''
    nchannels = 1
    _count = 0
    _mean = np.zeros((nchannels,), dtype=float)
    _stdev = np.zeros((nchannels,), dtype=float)

    for d in tqdm(dset, desc='Computing Pixel Statistics', leave=False):
        scale_factor = d["rescale_intensity"] if "rescale_intensity" in d else None
        im = read_image(d["file_name"], scale_factor=scale_factor, dtype='uint8')[:, :, 0]

        _count += 1
        for c in range(nchannels):
            _mean[c] += im.mean()
            _stdev[c] += im.std()

    _mean = _mean / _count
    _stdev = _stdev / _count

    return (_mean, _stdev)


def get_dataset_im_size_range(dset: Sequence[DataItem]) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    ''' Calculate min/max image width and height

    Parameters:
    dest (Sequence[DataItem]): annotations to compute statistics on

    Returns:
    ((min_width, max_width), (min_height, max_height))
    '''
    widths = [d['width'] for d in dset]
    heights = [d['height'] for d in dset]
    return (
        (np.min(widths), np.max(widths)),
        (np.min(heights), np.max(heights))
    )


def get_dataset_bbox_aspect_ratios(dset: Sequence[DataItem]) -> Dict[str, float]:
    ''' Calculate descriptive stats for bounding box aspect ratios

    Parameters:
    dest (Sequence[DataItem]): annotations to compute statistics on

    Returns:
    dict with keys [min, max, mean, median, stdev] and corresponding values
    '''
    aspect_ratios = []
    for d in dset:
        box = d['annotations'][0]['bbox']
        ax1 = box[2] - box[0]
        ax2 = box[3] - box[1]
        if ax1 < ax2:
            aspect_ratios.append(ax2 / ax1)
        else:
            aspect_ratios.append(ax1 / ax2)

    return {
        'min': np.min(aspect_ratios),
        'max': np.max(aspect_ratios),
        'mean': np.mean(aspect_ratios),
        'median': np.median(aspect_ratios),
        'stdev': np.std(aspect_ratios)
    }


def get_dataset_bbox_range(dset: Sequence[DataItem]):
    ''' Calculates bounding box min/mean/max with and height
    '''
    widths = []
    heights = []
    for d in dset:
        box = d['annotations'][0]['bbox']
        widths.append(box[2] - box[0])
        heights.append(box[3] - box[1])
    #logging.info("Width: {:.2f}/{:.2f}/{:.2f}".format(np.min(widths), np.mean(widths), np.max(widths)))
    #logging.info("Height: {:.2f}/{:.2f}/{:.2f}".format(np.min(heights), np.mean(heights), np.max(heights)))

    return {
        'width': {
            'min': np.min(widths),
            'max': np.max(widths),
            'mean': np.mean(widths),
            'median': np.median(widths),
            'stdev': np.std(widths)
        },
        'height': {
            'min': np.min(heights),
            'max': np.max(heights),
            'mean': np.mean(heights),
            'median': np.median(heights),
            'stdev': np.std(heights)
        }
    }


def split_test_train(annotations: MutableSequence[DataItem], split: float=0.90) -> Tuple[Callable[[], MutableSequence[DataItem]], Callable[[], MutableSequence[DataItem]]]:
    ''' Split annotations into training and testing datasets according to `split` ratio

    Parameters:
    annotations (MutableSequence[DataItem]): annotations to spit
    split (float): Fraction of `annotations` to be put into train set, the compliment will be put into test set

    Returns:
    Tuple[Callable[[], MutableSequence[DataItem]], Callable[[], MutableSequence[DataItem]]]
    index zero contains function that when called will return the training dataset
    index one contains function that when called will return the testing dataset
    '''
    random.shuffle(annotations)
    split_idx = int(len(annotations) * split)

    def train_annotations():
        return annotations[:split_idx]

    def test_annotations():
        return annotations[split_idx:]

    return (train_annotations, test_annotations)


def register_datasets(annotations: MutableSequence[DataItem], keypoint_names: Iterable[str], split: bool=True) -> None:
    ''' Register annotations with the Detectron2 DatasetCatalog

    Parameters:
    annotations (MutableSequence[DataItem]): annotations to register
    keypoint_names (Iterable[str]): names of the keypoints to register
    split (bool): If true, split annotations into training and test subsets, otherwise all annotations will be assigned to training
    '''
    if split:
        split_annot = split_test_train(annotations)
        for i, d in enumerate(['train', 'test']):
            DatasetCatalog.register("moseq_{}".format(d), split_annot[i])
            register_dataset_metadata("moseq_{}".format(d), keypoint_names)
    else:
        DatasetCatalog.register("moseq_train", annotations)
        register_dataset_metadata("moseq_train", keypoint_names)


def register_dataset_metadata(name: str, keypoint_names: Iterable[str]) -> None:
    ''' Register dataset metadata with the Detectron2 MetadataCatalog

    Parameters:
    name (str): name of the dataset, should correspond to a dataset registered with Detectron2 DatasetCatalog
    keypoint_names (Iterable[str]): names of the keypoints to register
    '''
    MetadataCatalog.get(name).thing_classes = ["mouse"]
    MetadataCatalog.get(name).thing_colors = [(0, 0, 255)]
    MetadataCatalog.get(name).keypoint_names = list(keypoint_names)
    MetadataCatalog.get(name).keypoint_flip_map = [] #[('Left Ear', 'Right Ear'), ('Left Hip', 'Right Hip')]
    MetadataCatalog.get(name).keypoint_connection_rules = default_keypoint_connection_rules
    MetadataCatalog.get(name).keypoint_colors = default_keypoint_colors


def poly_to_mask(poly: np.ndarray, out_shape: Tuple[int, int]) -> np.ndarray:
    ''' Convert a polygon mask into a bitmap mask

    Parameters:
    poly (np.ndarray): array of polygon coordinates of shape (ncoords, 2 [x, y])
    out_shape (Tuple[int, int]): Shape of the output mask

    Returns:
    np.ndarray of shape `out_shape` containing ones inside the polygon specified by `poly`, and zeros elsewhere
    '''
    mask = np.zeros((*out_shape, 1), dtype=np.uint8)
    rr,cc = polygon(poly[:,0], poly[:,1], out_shape)
    mask[cc, rr, 0] = 1
    return mask


def read_annotations(annot_file: str, keypoint_names: List[str]=None, mask_format: MaskFormat='polygon', rescale: float=1.0) -> Sequence[DataItem]:
    ''' Read annotations from json file output by labelstudio (coco-ish) format

    Parameters:
    annot_file (string): path to the annotation json file
    keypoint_names (List[str]): list of the keypoint names, in the order desired. If None, ignore keypoints
    mask_format (MaskFormat): 'polygon'|'bitmask', format of the masks to output
    rescale (float): instensity rescaling to apply (by dataset mapper) to image when loading

    Returns:
    Sequence[DataItem] annotations
    '''
    if keypoint_names is None:
        logging.warn("WARNING: Ignoring any keypoint information because `keypoint_names` is None.")

    with open(annot_file, 'r') as fp:
        data = json.load(fp)

        completions = []
        for entry in data:
            # depending version, we have seen keys `annotations` and `completions`
            if 'annotations' in entry:
                key = 'annotations'
            elif 'completions' in entry:
                key = 'completions'
            else:
                raise ValueError('Cannot find annotation data for entry!')

            entry_data = get_annotation_from_entry(entry, key=key, mask_format=mask_format, keypoint_names=keypoint_names)
            entry_data['rescale_intensity'] = rescale
            completions.append(entry_data)

        return completions


def get_polygon_data(entry: dict, mask_format: MaskFormat) -> dict:
    ''' Extract polygon data from an annotation entry
    '''
    poly = np.array(entry['value']['points'])
    poly[:,1] = (poly[:,1] * entry['original_width']) / 100
    poly[:,0] = (poly[:,0] * entry['original_height']) / 100

    if mask_format == 'polygon':
        seg = np.empty((poly.size,), dtype=poly.dtype)
        seg[0::2] = poly[:,0]
        seg[1::2] = poly[:,1]
        segmentation = [list(seg)]

    elif mask_format == 'bitmask':
        mask = poly_to_mask(poly, (entry['original_height'], entry['original_width']))
        segmentation = pycocotools.mask.encode(np.asfortranarray(mask))[0]

    else:
        raise RuntimeError("Got unsupported mask_format '{}'".format(mask_format))

    return {
        'category_id': 0,
        'bbox_mode': BoxMode.XYXY_ABS,
        'segmentation': segmentation,
        'bbox': [
            np.min(poly[:,0]),
            np.min(poly[:,1]),
            np.max(poly[:,0]),
            np.max(poly[:,1]),
        ],
    }


def get_keypoint_data(entry: dict) -> dict:
    ''' Extract keypoint data from an annotation entry
    '''
    return {
        entry['value']['keypointlabels'][0]: {
            'x': (entry['value']['x'] * entry['original_width']) / 100,
            'y': (entry['value']['y'] * entry['original_height']) / 100,
            'v': 2
        }
    }


def sort_keypoints(keypoint_order: List[str], keypoints: dict):
    ''' Sort `keypoints` to the order specified by `keypoint_order`
    '''
    annot_keypoints = []
    for kp in keypoint_order:
        if kp in keypoints:
            k = keypoints[kp]
            annot_keypoints.extend([k['x'], k['y'], k['v']])
        else:
            #logging.info('missing keypoint {} in {}'.format(kp, entry['id']))
            annot_keypoints.extend([0, 0, 0])
    return annot_keypoints


def get_image_path(entry: dict) -> str:
    ''' Extract image file path from an annotation entry
    '''
    if 'task_path' in entry:
        return entry['task_path']
    elif 'data' in entry and 'image' in entry['data']:
        return entry['data']['image']
    elif 'data' in entry and 'depth_image' in entry['data']:
        return entry['data']['depth_image']


def get_annotation_from_entry(entry: dict, key: str='annotations', mask_format: MaskFormat='polygon', keypoint_names: Optional[List[str]]=None) -> dict:
    annot = {}
    kpts = {}

    if len(entry[key]) > 1:
        logging.warn('WARNING: Task {}: Multiple annotations found, only taking the first'.format(entry['id']))

    for rslt in entry[key][0]['result']:

        if rslt['type'] == 'polygonlabels':
            if len(annot.keys()) > 0:
                logging.warn('WARNING: Task {}: Polygon has already been parsed, replacing value'.format(entry['id']))
            annot.update(get_polygon_data(rslt, mask_format=mask_format))

        elif rslt['type'] == 'keypointlabels':
            if 'points' in rslt['value']:
                #logging.info('Skipping unexpected points in keypoint', rslt)
                continue
            try:
                kdata = get_keypoint_data(rslt)
                kname = list(kdata.keys())[0]
                if kname in kpts:
                    logging.warn('WARNING: Task {}: Keypoint "{}" has already been parsed, replacing value'.format(entry['id'], kname))
                kpts.update(kdata)
            except:
                logging.info(rslt['value'])
                raise

    if keypoint_names is not None:
        annot['keypoints'] = sort_keypoints(keypoint_names, kpts)

    return {
        'file_name': get_image_path(entry),
        'width': rslt['original_width'],
        'height': rslt['original_height'],
        'image_id': entry['id'],
        'annotations': [annot],
    }


def replace_data_path_in_annotations(annotations: Sequence[DataItem], search: str, replace: str) -> Sequence[DataItem]:
    ''' Replace substrings in the filename of annotations.
    Useful when moving datasets to another computer

    Parameters:
    annotations (Sequence[DataItem]): annotations to validate
    search (str): substring to search for
    replace (str): replacement string

    Returns:
    Sequence[DataItem], annotations with data path modified according to `search` and `replace`
    '''
    for annot in annotations:
        annot['file_name'] = annot['file_name'].replace(search, replace)
    return annotations


def show_dataset_info(annotations: Sequence[DataItem]) -> None:
    ''' Print some basic information about a list of annotations.
        Includes the number of items, the size range of images, and the size range of bboxes

    Parameters:
    annotations (Sequence[DataItem]): annotations to validate
    '''
    logging.info("Number of Items: {}".format(len(annotations)))

    logging.info("Image size range:")
    im_sizes = get_dataset_im_size_range(annotations)
    logging.info(f" -> Width: {im_sizes[0][0]} - {im_sizes[0][1]} px")
    logging.info(f" -> Height: {im_sizes[1][0]} - {im_sizes[1][1]} px")

    logging.info("Instance Bounding Box Sizes:")
    bbox_sizes = get_dataset_bbox_range(annotations)
    bbox_ratios = get_dataset_bbox_aspect_ratios(annotations)
    logging.info(f" -> Width: {bbox_sizes['width']['min']:.2f} - {bbox_sizes['width']['max']:.2f}; mean {bbox_sizes['width']['mean']:.2f} ± {bbox_sizes['width']['stdev']:.2f} stdev")
    logging.info(f" -> Height: {bbox_sizes['height']['min']:.2f} - {bbox_sizes['height']['max']:.2f}; mean {bbox_sizes['height']['mean']:.2f} ± {bbox_sizes['height']['stdev']:.2f} stdev")
    logging.info(f" -> Ratio: {bbox_ratios['min']:.2f} - {bbox_ratios['max']:.2f}; mean {bbox_ratios['mean']:.2f} ± {bbox_ratios['stdev']:.2f} stdev")

    logging.info("Pixel Intensity Statistics:")
    im_means, im_stdevs = get_dataset_statistics(annotations)
    for ch in range(im_means.shape[0]):
        logging.info(f" -> Ch{ch}: mean {im_means[ch]:.2f} ± {im_stdevs[ch]:.2f} stdev")


def validate_annotations(annotations: Sequence[DataItem]) -> bool:
    ''' Validate annotations

    Checks performed:
     - The file path exists on the file system

    Parameters:
    annotations (Sequence[DataItem]): annotations to validate

    Returns:
    bool, True if all validation checks pass, otherwise raises exception
    '''
    for annot in annotations:
        if not os.path.isfile(annot['file_name']):
            raise FileNotFoundError(annot['file_name'])
    return True

