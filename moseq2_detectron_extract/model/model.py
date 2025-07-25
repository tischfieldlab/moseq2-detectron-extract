import os
from typing import Optional

from albumentations.augmentations.transforms import GaussNoise
from detectron2.config.config import CfgNode
from detectron2.data import (build_detection_test_loader,
                             build_detection_train_loader)
from detectron2.data.transforms import (FixedSizeCrop, RandomBrightness,
                                        RandomContrast, RandomRotation)
from detectron2.engine.defaults import DefaultTrainer
from detectron2.evaluation import COCOEvaluator

from moseq2_detectron_extract.io.util import ensure_dir
from moseq2_detectron_extract.model.augmentations import (
    Albumentations, DoughnutGRFNoiseAugmentation, ParticleNoiseAugmentation,
    RandomFieldNoiseAugmentation, ScaleAugmentation)
from moseq2_detectron_extract.model.hooks import LossEvalHook, MemoryUsageHook
from moseq2_detectron_extract.model.mapper import MoseqDatasetMapper


class Trainer(DefaultTrainer):
    '''
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can write your
    own training loop. You can use "tools/plain_train_net.py" as an example.
    '''

    @classmethod
    def build_train_loader(cls, cfg: CfgNode):
        augs = [
            RandomRotation([0, 359], expand=False, sample_style='range'),
            ScaleAugmentation(0.75, 1.2, 250, 250),
            FixedSizeCrop((250, 250), pad=True, pad_value=0),
            RandomBrightness(0.9, 1.1),
            RandomContrast(0.9, 1.1),
            Albumentations(GaussNoise()),
            DoughnutGRFNoiseAugmentation(p=0.5),
            ParticleNoiseAugmentation(p=0.5),
            RandomFieldNoiseAugmentation(p=0.5),

            # did not see much improvement from adding these:
            # Albumentations(GlassBlur(sigma=0.2, max_delta=1, iterations=1, p=0.5)),
            # Albumentations(MotionBlur(blur_limit=3, p=0.5)),
        ]
        mapper = MoseqDatasetMapper(cfg, is_train=True, augmentations=augs, use_instance_mask=True, use_keypoint=True, recompute_boxes=True)
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_test_loader(cls, cfg: CfgNode, dataset_name: str):
        mapper = MoseqDatasetMapper(cfg, is_train=False, augmentations=[], use_instance_mask=True, use_keypoint=True, recompute_boxes=True)
        return build_detection_test_loader(cfg, dataset_name, mapper=mapper)

    @classmethod
    def build_evaluator(cls, cfg: CfgNode, dataset_name: str, output_folder: Optional[str]=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "coco_eval")
            ensure_dir(output_folder)

        return COCOEvaluator(dataset_name, ("bbox", "segm", "keypoints"), True, output_dir=output_folder, kpt_oks_sigmas=cfg.TEST.KEYPOINT_OKS_SIGMAS)

    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(-1, LossEvalHook(
            self.cfg.TEST.EVAL_PERIOD,
            self.model,
            build_detection_test_loader(
                self.cfg,
                self.cfg.DATASETS.TEST[0],
                MoseqDatasetMapper(self.cfg, is_train=True, augmentations=[])
            )
        ))
        hooks.insert(-1, MemoryUsageHook())
        return hooks
