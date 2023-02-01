import collections
import glob
import os
import subprocess
import sys
from typing import Dict, List

import torch
import torchvision
import detectron2
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


def ensure_spawn_start_method() -> None:
    ''' Ensure that multiprocessing start method is set to spawn
    '''
    torch.multiprocessing.set_start_method('spawn', force=True)


def get_default_device() -> str:
    ''' Get the moniker for the current torch device
    '''
    ensure_spawn_start_method()
    if torch.cuda.is_available():
        return f'cuda:{torch.cuda.current_device()}'
    else:
        return 'cpu'


def get_available_devices() -> List[str]:
    ''' Get a list of available torch device monikers
    '''
    ensure_spawn_start_method()
    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda') # generic cuda is allowed
        for d in range(torch.cuda.device_count()):
            devices.append(f'cuda:{d}')
    return devices


def get_available_device_info():
    ''' Get a dictionary of available device information

    If fails, will return None

    Returns:
        dict|None
    '''
    try:
        cmd = [
            "nvidia-smi",
            "--format=csv",
            "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free"
        ]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        out, _ = process.communicate()

        data = []
        headers = []
        for line_i, line in enumerate(out.split('\n')):
            if line.strip() == '':
                continue
            fields = [f.strip() for f in line.split(',')]
            if line_i == 0:
                headers.extend(fields)
            else:
                item = collections.OrderedDict()
                for field_i, field_name in enumerate(fields):
                    item[headers[field_i]] = field_name
                data.append(item)

        return data
    except: # pylint: disable=bare-except
        return None


def get_system_versions():
    ''' Get a dictionary of framework versions

    Returns:
        dict: Contains keys `Framework` and `Version`, both containing lists of data.
    '''
    data = collections.OrderedDict()
    data['Framework'] = []
    data['Version'] = []

    data['Framework'].append('Python')
    data['Version'].append(sys.version)

    data['Framework'].append('PyTorch')
    data['Version'].append(torch.__version__)

    data['Framework'].append('TorchVision')
    data['Version'].append(torchvision.__version__)

    data['Framework'].append('CUDA')
    data['Version'].append(str(torch.version.cuda))

    data['Framework'].append('CUDNN')
    data['Version'].append(str(torch.backends.cudnn.version()))

    data['Framework'].append('Detectron2')
    data['Version'].append(str(detectron2.__version__))

    return data
