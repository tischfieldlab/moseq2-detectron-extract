import datetime
import logging
import time

import detectron2.utils.comm as comm
import numpy as np
import torch
from detectron2.engine.hooks import HookBase
from detectron2.utils.logger import log_every_n_seconds


class MemoryUsageHook(HookBase):
    ''' Hook to log torch memory statistics to tensorboard
    '''
    def after_step(self):
        current_device = torch.cuda.current_device()
        stats = torch.cuda.memory_stats(current_device)
        with self.trainer.storage.name_scope("Memory"):
            self.trainer.storage.put_scalars(**stats)


class LossEvalHook(HookBase):
    ''' Hook to compute loss on evaluation dataset
    '''
    def __init__(self, eval_period, model, data_loader):
        self._model = model
        self._period = eval_period
        self._data_loader = data_loader


    def _do_loss_eval(self):
        '''
        Copying inference_on_dataset from evaluator.py
        '''
        total = len(self._data_loader)
        num_warmup = min(5, total - 1)

        start_time = time.perf_counter()
        total_compute_time = 0
        losses = {}
        for idx, inputs in enumerate(self._data_loader):
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0
            start_compute_time = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time
            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_img = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_img > 5:
                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    f"Loss on Validation done {idx + 1}/{total}. {seconds_per_img:.4f} s / img. ETA={eta}",
                    n=5,
                )
            loss_batch = self._get_loss(inputs)
            for key, value in loss_batch.items():
                try:
                    losses[key].append(value)
                except KeyError:
                    losses[key] = [value]

        for key, value in losses.items():
            with self.trainer.storage.name_scope(key):
                self.trainer.storage.put_scalar(f'validation_{key}', np.mean(value))
        comm.synchronize()

        return losses


    def _get_loss(self, data):
        # How loss is calculated on train_loop
        metrics_dict = self._model(data)
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        metrics_dict['total_loss'] = sum(loss for loss in metrics_dict.values())
        return metrics_dict


    def after_step(self):
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if is_final or (self._period > 0 and next_iter % self._period == 0):
            self._do_loss_eval()
        # self.trainer.storage.put_scalars(timetest=12) # Why we log a constant??
