from contextlib import ExitStack
from typing import Any

import numpy as np
import torch
from detectron2.checkpoint.detection_checkpoint import DetectionCheckpointer
from detectron2.config.config import CfgNode
from detectron2.modeling.meta_arch.build import build_model
from moseq2_detectron_extract.model.util import outputs_to_instances


class Predictor:
    ''' Predictor class implementation
    '''
    def __init__(self, model: Any, is_torchscript: bool=False):
        '''

        Model should already be in eval mode!
        '''
        self.model = model
        self.is_torchscript = is_torchscript
        self.exit_stack = ExitStack()
        self.exit_stack.enter_context(torch.no_grad())

    @property
    def device(self):
        ''' Get the device the model is on
        '''
        return next(self.model.parameters()).device

    @classmethod
    def from_config(cls, cfg: CfgNode):
        ''' Create a predictor given a Detectron2 config
        '''
        cfg = cfg.clone()  # cfg can be modified by model
        model = build_model(cfg)
        model.eval()

        checkpointer = DetectionCheckpointer(model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        assert model.input_format == cfg.INPUT.FORMAT

        return cls(model)

    @classmethod
    def from_torchscript(cls, path):
        ''' Create a predictor given a path to a torchscript model
        '''
        model = torch.jit.load(path)
        return cls(model, is_torchscript=True)

    def __call__(self, original_image: np.ndarray):
        '''
        Parameters:
        original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).
        -- OR --
        original_image (np.ndarray): an image of shape (N, H, W, C) (in BGR order).

        Returns:
        predictions (dict):
            the output of the model for one image only.
            See :doc:`/tutorials/models` for details about the format.
        '''
        #with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
        # Apply pre-processing to image.
        return_as_list = True
        if len(original_image.shape) == 3:
            # if we were not given a batch, convert to a batch.
            # afterwards, origional_image will always be of shape (N, H, W, C)
            return_as_list = False
            original_image = original_image[None, ...]

        if self.model.input_format == 'RGB' and original_image.shape[3] == 1:
            # the model is expecting RGB format, but we were given single channel
            # grayscale. Convert by just copying the single channel 3 times.
            original_image = np.concatenate((original_image,) * 3, axis=3)

        inputs = []
        for i in range(original_image.shape[0]):
            image = original_image[i]
            height, width = (torch.tensor(x) for x in image.shape[:2])
            #image = self.aug.get_transform(original_image).apply_image(original_image)
            if isinstance(image, torch.Tensor):
                image = image.movedim(2, 0)
            elif isinstance(image, np.ndarray):
                image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
            else:
                raise TypeError(f'Expected torch.tensor or numpy.ndarray; got {type(image)}')
            inputs.append({"image": image, "height": height, "width": width})

        predictions = self.model(inputs)

        if self.is_torchscript:
            predictions = outputs_to_instances(inputs, predictions)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        if not return_as_list:
            return predictions[0]
        else:
            return predictions
