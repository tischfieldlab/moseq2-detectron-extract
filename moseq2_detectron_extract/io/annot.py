import copy
import json
import os
import random
from typing import Iterable, Sequence, Union
from typing_extensions import TypedDict

import cv2
import numpy as np
import pycocotools
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from skimage.draw import polygon

from detectron2.data.detection_utils import transform_instance_annotations


class Annotation(TypedDict):
    bbox: Sequence[float]
    bbox_mode: BoxMode
    category_id: int
    segmentation: Union[Sequence[Sequence[float]], dict]
    keypoints: Sequence[float]

class DataItem(TypedDict):
    file_name: str
    width: int
    height: int
    image_id: str
    annotations: Sequence[Annotation]


def get_dataset_statistics(dset: Sequence[DataItem]):
    ''' Calculate mean a standard deviation of images over a dataset.
    
    online mean and stdev calculation: https://stackoverflow.com/a/15638726
    '''
    # online mean and stdev calculation
    # https://stackoverflow.com/a/15638726
    
    _count = 0
    _mean = 0
    _m2 = 0
    
    for d in dset:
        im = cv2.imread(d["file_name"])[:, :, 0]
        for x in im.ravel():
            _count += 1
            delta = x - _mean
            _mean += delta / _count
            _m2 += delta * (x - _mean)
    
    _variance = _m2 / (_count - 1)
    _stdev = np.sqrt(_variance)
    
    return (_mean, _stdev)

def get_dataset_im_size_range(dset: Sequence[DataItem]):
    ''' Calculate min/max image width and height
    
    Returns: ((min_width, max_width), (min_height, max_height))
    '''
    widths = [d['width'] for d in dset]
    heights = [d['height'] for d in dset]
    return (
        (np.min(widths), np.max(widths)),
        (np.min(heights), np.max(heights))
    )

def get_dataset_bbox_range(dset: Sequence[DataItem]):
    ''' Calculates bounding box min/mean/max with and height
    '''
    widths = []
    heights = []
    for d in dset:
        box = d['annotations'][0]['bbox']
        widths.append(box[2] - box[0])
        heights.append(box[3] - box[1])
    print("Width: {:.2f}/{:.2f}/{:.2f}".format(np.min(widths), np.mean(widths), np.max(widths)))
    print("Height: {:.2f}/{:.2f}/{:.2f}".format(np.min(heights), np.mean(heights), np.max(heights)))

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

def split_test_train(annotations: Sequence[DataItem], split: float=0.90):
    random.shuffle(annotations)
    split_idx = int(len(annotations) * split)

    def train_annotations():
        return annotations[:split_idx]

    def test_annotations():
        return annotations[split_idx:]

    return (train_annotations, test_annotations)

def register_datasets(annotations: Sequence[DataItem], keypoint_names):
    split_annot = split_test_train(annotations)
    for i, d in enumerate(['train', 'test']):
        DatasetCatalog.register("moseq_{}".format(d), split_annot[i])
        register_dataset_metadata("moseq_{}".format(d), keypoint_names)

def register_dataset_metadata(name: str, keypoint_names: Iterable[str]):
    MetadataCatalog.get(name).thing_classes = ["mouse"]
    MetadataCatalog.get(name).thing_colors = [(0, 0, 255)]
    MetadataCatalog.get(name).keypoint_names = list(keypoint_names)
    MetadataCatalog.get(name).keypoint_flip_map = [] #[('Left Ear', 'Right Ear'), ('Left Hip', 'Right Hip')]
    MetadataCatalog.get(name).keypoint_connection_rules = [
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


def poly_to_mask(poly, out_shape):
    ''' Convert a polygon mask into a bitmap mask
    '''
    mask = np.zeros((*out_shape, 1), dtype=np.uint8)
    rr,cc = polygon(poly[:,0], poly[:,1], out_shape)
    mask[cc, rr, 0] = 1
    return mask


def augment_annotations_with_rotation(annotations: Sequence[DataItem], angles: Iterable[int]=None) -> Sequence[DataItem]:
    if angles is None:
        angles = [0, 90, 180, 270]

    out_annotations = []
    for angle in angles:
        for annot in copy.deepcopy(annotations):
            annot['rotate'] = angle
            out_annotations.append(annot)
    return out_annotations


def read_annotations(annot_file, keypoint_names=None, mask_format='polygon', replace_path=None, rescale=1.0) -> Sequence[DataItem]:
    ''' Read annotations from json file output by labelstudio (coco-ish) format

        Parameters:
            annot_file (string): path to the annotation json file
            keypoint_names (list<string>): list of the keypoint names, in the order desired. If None, ignore keypoints
            mask_format (string): 'polygon'|'bitmask'
    '''
    if keypoint_names is None:
        print("WARNING: Ignoring any keypoint information because `keypoint_names` is None.")

    
    with open(annot_file, 'r') as fp:
        data = json.load(fp)

        completions = []
        for entry in data:
            annot = {}
            kpts = {}
            
            polyfound = False
            
            for rslt in entry['completions'][0]['result']:
                
                if rslt['type'] == 'polygonlabels':
                    poly = np.array(rslt['value']['points'])
                    poly[:,1] = (poly[:,1] * rslt['original_width']) / 100
                    poly[:,0] = (poly[:,0] * rslt['original_height']) / 100
                    
                    
                    annot['bbox'] = [
                        np.min(poly[:,0]),
                        np.min(poly[:,1]),
                        np.max(poly[:,0]),
                        np.max(poly[:,1]),
                    ]
                    annot['bbox_mode'] = BoxMode.XYXY_ABS
                    annot['category_id'] = 0
                    
                    if mask_format == 'polygon':
                        seg = np.empty((poly.size,), dtype=poly.dtype)
                        seg[0::2] = poly[:,0]
                        seg[1::2] = poly[:,1]
                        annot['segmentation'] = [list(seg)]
                    
                    elif mask_format == 'bitmask':
                        mask = poly_to_mask(poly, (rslt['original_height'], rslt['original_width']))
                        annot['segmentation'] = pycocotools.mask.encode(np.asfortranarray(mask))[0]
                    
                    else:
                        raise RuntimeError("Got unsupported mask_format '{}'".format(mask_format))
                    polyfound = True
                    
                elif rslt['type'] == 'keypointlabels':
                    if 'points' in rslt['value']:
                        #print('Skipping unexpected points in keypoint', rslt)
                        continue
                    try:
                        kpts[rslt['value']['keypointlabels'][0]] = {
                            'x': (rslt['value']['x'] * rslt['original_width']) / 100,
                            'y': (rslt['value']['y'] * rslt['original_height']) / 100,
                            'v': 2
                        }
                    except:
                        print(rslt['value'])
                        raise
            
            if keypoint_names is not None:
                annot_keypoints = []
                for kp in keypoint_names:
                    if kp in kpts:
                        k = kpts[kp]
                        annot_keypoints.extend([k['x'], k['y'], k['v']])
                    else:
                        #print('missing keypoint {} in {}'.format(kp, entry['id']))
                        annot_keypoints.extend([0, 0, 0])
                annot['keypoints'] = annot_keypoints
            
            tsk_path = ''
            if 'task_path' in entry:
                tsk_path = entry['task_path']
            else:
                tsk_path = entry['data']['image']
                #o = parse.urlparse(entry['data']['image'])
                #q = parse.parse_qs(o.query)
                #tsk_path = os.path.join(q['d'][0], os.path.basename(o.path))

            if replace_path is not None and replace_path[0] is not None and replace_path[1] is not None:
                tsk_path = tsk_path.replace(replace_path[0], replace_path[1])
            
            if not os.path.isfile(tsk_path):
                raise FileNotFoundError(tsk_path)
                
            if not polyfound:
                print('poly not found!')
                
            completions.append({
                'file_name': tsk_path,
                'width': rslt['original_width'],
                'height': rslt['original_height'],
                'image_id': entry['id'],
                'annotations': [annot],
                'rescale_intensity': rescale
            })
        
        return completions

def replace_data_path_in_annotations(annotations: Sequence[DataItem], search: str, replace: str):
    '''
    
    '''
    for annot in annotations:
        annotations['file_name'].replace(search, replace)
    return annotations


def show_dataset_info(annotations: Sequence[DataItem]):
    ''' Print some basic information about a list of annotations
        Includes the number of items, the size range of images, and the size range of bboxes
    '''
    print("Num Items: ", len(annotations))
    print("Image size range: ", get_dataset_im_size_range(annotations))
    #print(get_dataset_statistics(annotations))
    print(get_dataset_bbox_range(annotations))




