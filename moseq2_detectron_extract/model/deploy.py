
import os
from typing import Dict, List

import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config.config import CfgNode
from detectron2.evaluation import print_csv_format
from detectron2.export import (add_export_config, dump_torchscript_IR,
                               scripting_with_instances)
from detectron2.modeling import GeneralizedRCNN, build_model
from detectron2.structures import Boxes
from detectron2.utils.env import TORCH_VERSION
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import setup_logger
from moseq2_detectron_extract.model.eval import \
    inference_on_dataset_readonly_model
from moseq2_detectron_extract.model.model import Trainer
from torch import Tensor, nn


def export_model(cfg: CfgNode, output: str, run_eval: bool=True):
    logger = setup_logger()

    torch._C._jit_set_bailout_depth(1) # type: ignore

    # cuda context is initialized before creating dataloader, so we don't fork anymore
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg = add_export_config(cfg)
    cfg.freeze()


    # create a torch model
    torch_model = build_model(cfg)
    DetectionCheckpointer(torch_model).resume_or_load(cfg.MODEL.WEIGHTS)
    torch_model.eval()

    exported_model, path = export_scripting(torch_model, output)

    exported_model = torch.jit.load(path)

    if run_eval:
        assert exported_model is not None, (
            "Python inference is not yet implemented for "
            #f"export_method={args.export_method}, format={args.format}."
        )
        logger.info("Running evaluation ... this takes a long time if you export to CPU.")
        dataset = cfg.DATASETS.TEST[0]
        data_loader = Trainer.build_test_loader(cfg, dataset)

        evaluator = Trainer.build_evaluator(cfg, dataset, output)
        metrics = inference_on_dataset_readonly_model(exported_model, data_loader, evaluator)
        print_csv_format(metrics)


def export_scripting(torch_model, output: str):
    assert TORCH_VERSION >= (1, 8)
    fields = {
        "proposal_boxes": Boxes,
        "objectness_logits": Tensor,
        "pred_boxes": Boxes,
        "scores": Tensor,
        "pred_classes": Tensor,
        "pred_masks": Tensor,
        "pred_keypoints": torch.Tensor,
        "pred_keypoint_heatmaps": torch.Tensor,
    }
    #assert args.format == "torchscript", "Scripting only supports torchscript format."

    class ScriptableAdapterBase(nn.Module):
        # Use this adapter to workaround https://github.com/pytorch/pytorch/issues/46944
        # by not retuning instances but dicts. Otherwise the exported model is not deployable
        def __init__(self):
            super().__init__()
            self.model = torch_model
            self.eval()

    if isinstance(torch_model, GeneralizedRCNN):
        class GRCNNScriptableAdapter(ScriptableAdapterBase):
            def forward(self, inputs: List[Dict[str, torch.Tensor]]) -> List[Dict[str, Tensor]]:
                instances = self.model.inference(inputs, do_postprocess=False)
                return [i.get_fields() for i in instances]

        ts_model = scripting_with_instances(GRCNNScriptableAdapter(), fields)

    else:
        class ScriptableAdapter(ScriptableAdapterBase):
            def forward(self, inputs: List[Dict[str, torch.Tensor]]) -> List[Dict[str, Tensor]]:
                instances = self.model(inputs)
                return [i.get_fields() for i in instances]

        ts_model = scripting_with_instances(ScriptableAdapter(), fields)

    with PathManager.open(os.path.join(output, "model.ts"), "wb") as f:
        torch.jit.save(ts_model, f)
    dump_torchscript_IR(ts_model, output)

    # TODO inference in Python now missing postprocessing glue code
    return ts_model, os.path.join(output, "model.ts")
