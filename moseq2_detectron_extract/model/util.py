import glob
import os
from typing import Dict, List

import torch
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.structures import Boxes, Instances


def get_last_checkpoint(path: str) -> str:
    ''' Get the path to the last model checkpoint in a model directory by looking
    at the "last_checkpoint" file in the directory

    Parameters:
    path (str): directory containing the modelling results

    Returns:
    Path to the last checkpoint
    '''
    with open(os.path.join(path, 'last_checkpoint'), 'r', encoding='utf-8') as chkpt_file:
        last_checkpoint = chkpt_file.read()
    return os.path.join(path, last_checkpoint)


def get_specific_checkpoint(path: str, iteration: int, ext: str='pth') -> str:
    ''' Get the path to the model at a specific checkpoint in a model directory

    Parameters:
    path (str): directory containing the modelling results
    iteration (int): iteration number to look for
    ext (str): file extension of the model file

    Returns:
    Path to checkpoint at `iteration`
    '''
    matches = glob.glob(os.path.join(path, f'*{iteration}.{ext}'))
    return matches[0]


def outputs_to_instances(inputs: List[Dict[str, torch.Tensor]], outputs: List[Dict[str, torch.Tensor]]) -> List[dict]:
    ''' Transform model outputs to Instances
    '''
    instances = []
    for i, o in zip(inputs, outputs):
        height = i.get("height", i['image'].shape[0])
        width = i.get("width", i['image'].shape[1])
        ins = Instances(
            (height, width),
            **{
                'pred_boxes': Boxes(o.pop('pred_boxes')),
                **o,
            }
        )
        ins = detector_postprocess(ins, height, width)
        instances.append({"instances": ins})

    return instances
