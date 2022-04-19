

import os
from functools import partial
import warnings

from moseq2_detectron_extract.model import Predictor
from moseq2_detectron_extract.model.config import get_base_config
from moseq2_detectron_extract.model.util import (get_last_checkpoint,
                                                 get_specific_checkpoint)
from moseq2_detectron_extract.pipeline.pipeline_step import ProcessPipelineStep
from moseq2_detectron_extract.proc.proc import scale_raw_frames

# pylint: disable=attribute-defined-outside-init

class InferenceStep(ProcessPipelineStep):
    ''' Step to provide inference on frame data
    '''

    def initialize(self):
        warnings.filterwarnings("ignore", category=UserWarning, module='torch') # disable UserWarning: floor_divide is deprecated

        self.scale = partial(scale_raw_frames, vmin=self.config['min_height'], vmax=self.config['max_height'])

        self.write_message('Loading model....')
        model_path: str = self.config['model']

        if os.path.isfile(model_path) and model_path.endswith('.ts'):
            self.write_message(f' -> Using torchscript model "{os.path.abspath(model_path)}"....')
            self.write_message(' -> WARNING: Ignoring --device parameter because this is a torchscript model')
            self.predictor = Predictor.from_torchscript(model_path)

        else:
            cfg = get_base_config()
            checkpoint = self.config['checkpoint']
            with open(os.path.join(self.config['model'], 'config.yaml'), 'r', encoding='utf-8') as cfg_file:
                cfg = cfg.load_cfg(cfg_file)

            if checkpoint == 'last':
                cfg.MODEL.WEIGHTS = get_last_checkpoint(self.config['model'])
                self.write_message(f' -> Using last model checkpoint: "{cfg.MODEL.WEIGHTS}"')
            else:
                cfg.MODEL.WEIGHTS = get_specific_checkpoint(self.config['model'], checkpoint)
                self.write_message(f' -> Using model checkpoint at iteration {checkpoint}: "{cfg.MODEL.WEIGHTS}"')

            self.write_message(f" -> Setting device to \"{self.config['device']}\"")
            cfg.MODEL.DEVICE = self.config['device']

            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6  # set a custom testing threshold
            cfg.TEST.DETECTIONS_PER_IMAGE = 1
            self.predictor = Predictor.from_config(cfg)

        self.write_message(f' -> Actually using device "{self.predictor.device}"')
        self.write_message('')

    def process(self, data):

        raw_frames = data['chunk']
        batch_size = min(self.config['batch_size'], raw_frames.shape[0])
        batches = range(0, raw_frames.shape[0], batch_size)

        # Do the inference
        outputs = []
        for i in batches:
            frames = self.scale(raw_frames[i:i+batch_size,:,:,None])
            pred = self.predictor(frames)
            outputs.extend([{ 'instances': p['instances'].to('cpu') } for p in pred])
            self.update_progress(frames.shape[0])

        data['inference'] = outputs
        return data
