import json
import logging
import os
import random
from typing import (Callable, Dict, Iterable, List, Literal, MutableSequence,
                    Optional, Sequence, Tuple, TypedDict, Union, cast)

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
import numpy as np
import pycocotools
from skimage.draw import polygon
from tqdm import tqdm

from moseq2_detectron_extract.io.image import read_image


class SegmAnnotation(TypedDict):
    ''' Annotation data used for segmentation tasks
    '''
    bbox: Sequence[float]
    bbox_mode: BoxMode
    category_id: int
    segmentation: Union[Sequence[Sequence[float]], dict]


class KptSegmAnnotation(SegmAnnotation):
    ''' Annotation data used for koypoint detection tasks
    '''
    keypoints: Sequence[float]


class DataItem(TypedDict):
    ''' DataItem containing data for detectron2 training
    '''
    file_name: str
    width: int
    height: int
    image_id: str
    rescale_intensity: float
    annotations: Sequence[KptSegmAnnotation]


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


def load_annotations_helper(annot_files: Iterable[str], replace_paths: Iterable[Tuple[str, str]]=None,
                            mask_format: str='polygon', register: bool=True, split: bool=True, show_info: bool=True):
    ''' Utility "do-it-all function for the common task of loading and processing annotations

    Parameters:
    annot_files (Iterable[str]): file paths to read annotation information from
    replace_paths (Iterable[Tuple[str, str]]): search and replacement pairs for fixing filename paths in annotations
    mask_format (str): format that masks should be loaded as
    register (bool): if True, register loaded annotations with detectron2 dataset register
    split (bool): if True, and `register` is True, split the dataset into train/test
    show_info (bool): if True, show information about the dataset

    Returns:
    List[DataItem] - annotations
    '''
    logging.info('Loading annotations....')
    annotations: List[DataItem] = []
    for anot_f in annot_files:
        logging.info(f'Reading annotation file "{anot_f}"')
        annot = read_annotations(anot_f, default_keypoint_names, mask_format=mask_format)
        logging.info(f' -> Found {len(annot)} annotations')
        annotations.extend(annot)

    if replace_paths is not None:
        for search, replace in replace_paths:
            replace_data_path_in_annotations(annotations, search, replace)
    validate_annotations(annotations)

    if show_info:
        logging.info('Dataset information:')
        show_dataset_info(annotations)

    if register:
        register_datasets(annotations, split=split)

    return annotations


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
        image = read_image(d["file_name"], scale_factor=scale_factor, dtype='uint8')[:, :, 0]

        _count += 1
        for c in range(nchannels):
            _mean[c] += image.mean()
            _stdev[c] += image.std()

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

DatasetSplitter = Tuple[Callable[[], MutableSequence[DataItem]], Callable[[], MutableSequence[DataItem]]]
def split_test_train(annotations: MutableSequence[DataItem], split: float=0.90) -> DatasetSplitter:
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


def register_datasets(annotations: MutableSequence[DataItem], split: bool=True) -> None:
    ''' Register annotations with the Detectron2 DatasetCatalog

    Parameters:
    annotations (MutableSequence[DataItem]): annotations to register
    split (bool): If true, split annotations into training and test subsets, otherwise all annotations will be assigned to training
    '''
    if split:
        split_annot = split_test_train(annotations)
        for i, dset_type in enumerate(['train', 'test']):
            DatasetCatalog.register(f"moseq_{dset_type}", split_annot[i])
            register_dataset_metadata(f"moseq_{dset_type}")
    else:
        DatasetCatalog.register("moseq_train", lambda: annotations)
        register_dataset_metadata("moseq_train")


def register_dataset_metadata(name: str) -> None:
    ''' Register dataset metadata with the Detectron2 MetadataCatalog

    Parameters:
    name (str): name of the dataset, should correspond to a dataset registered with Detectron2 DatasetCatalog
    '''
    MetadataCatalog.get(name).thing_classes = ["mouse"]
    MetadataCatalog.get(name).thing_colors = [(0, 0, 255)]
    MetadataCatalog.get(name).keypoint_names = default_keypoint_names
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
        logging.warning("WARNING: Ignoring any keypoint information because `keypoint_names` is None.")

    with open(annot_file, 'r', encoding='utf-8') as in_file:
        data = json.load(in_file)

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


def get_polygon_data(entry: dict, mask_format: MaskFormat) -> SegmAnnotation:
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
        raise RuntimeError(f"Got unsupported mask_format '{mask_format}'")

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
    else:
        raise KeyError('Could not locate image path from entry!')


def get_annotation_from_entry(entry: dict, key: str='annotations', mask_format: MaskFormat='polygon',
                              keypoint_names: Optional[List[str]]=None) -> DataItem:
    ''' Fetch annotation data from a task entry
    '''
    annot: dict = {}
    kpts = {}
    original_width = None
    original_height = None


    if len(entry[key]) > 1:
        logging.warning(f"WARNING: Task {entry['id']}: Multiple annotations found, only taking the first")

    for rslt in entry[key][0]['result']:
        if 'original_width' in rslt and original_width is None:
            original_width = rslt['original_width']

        if 'original_height' in rslt and original_height is None:
            original_height = rslt['original_height']

        if rslt['type'] == 'polygonlabels':
            if len(annot.keys()) > 0:
                logging.warning(f"WARNING: Task {entry['id']}: Polygon has already been parsed, replacing value")
            annot.update(get_polygon_data(rslt, mask_format=mask_format))

        elif rslt['type'] == 'keypointlabels':
            if 'points' in rslt['value']:
                #logging.info('Skipping unexpected points in keypoint', rslt)
                continue
            try:
                kdata = get_keypoint_data(rslt)
                kname = list(kdata.keys())[0]
                if kname in kpts:
                    logging.warning(f"WARNING: Task {entry['id']}: Keypoint \"{kname}\" has already been parsed, replacing value")
                kpts.update(kdata)
            except:
                logging.info(rslt['value'])
                raise

    if keypoint_names is not None:
        annot['keypoints'] = sort_keypoints(keypoint_names, kpts)

    return {
        'file_name': get_image_path(entry),
        'width': original_width,
        'height': original_height,
        'image_id': entry['id'],
        'annotations': [cast(KptSegmAnnotation, annot)],
        'rescale_intensity': 1,
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
    logging.info(f"Number of Items: {len(annotations)}")

    logging.info("Image size range:")
    im_sizes = get_dataset_im_size_range(annotations)
    logging.info(f" -> Width: {im_sizes[0][0]} - {im_sizes[0][1]} px")
    logging.info(f" -> Height: {im_sizes[1][0]} - {im_sizes[1][1]} px")

    logging.info("Instance Bounding Box Sizes:")
    bbox_sizes = get_dataset_bbox_range(annotations)
    bbox_ratios = get_dataset_bbox_aspect_ratios(annotations)
    logging.info(f" -> Width: {bbox_sizes['width']['min']:.2f} - {bbox_sizes['width']['max']:.2f}; "
                 f"mean {bbox_sizes['width']['mean']:.2f} ± {bbox_sizes['width']['stdev']:.2f} stdev")
    logging.info(f" -> Height: {bbox_sizes['height']['min']:.2f} - {bbox_sizes['height']['max']:.2f}; "
                 f"mean {bbox_sizes['height']['mean']:.2f} ± {bbox_sizes['height']['stdev']:.2f} stdev")
    logging.info(f" -> Ratio: {bbox_ratios['min']:.2f} - {bbox_ratios['max']:.2f}; "
                 f"mean {bbox_ratios['mean']:.2f} ± {bbox_ratios['stdev']:.2f} stdev")

    logging.info("Pixel Intensity Statistics:")
    im_means, im_stdevs = get_dataset_statistics(annotations)
    for channel in range(im_means.shape[0]):
        logging.info(f" -> Ch{channel}: mean {im_means[channel]:.2f} ± {im_stdevs[channel]:.2f} stdev")


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
