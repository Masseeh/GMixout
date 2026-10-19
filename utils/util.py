import torch
import numpy as np
import os
import torch.distributed as dist
import logging
from .distributed import from_ddp

def breakpoint(rank=0):
    if dist.get_rank() == rank:import ipdb; ipdb.set_trace()

class DummyWriter:
    def __init__(self, _writer) -> None:
        self._writer = _writer
    def add_scalar(self, *args, **kwargs):
        if self._writer is not None:
            self._writer.add_scalar(*args, **kwargs)
    def close(self):
        if self._writer is not None:
            self._writer.close()

class ModelSelection:
    def __init__(self, device="cuda", key='bal_acc'):
        self.best_metric = None
        self.best_epoch = None
        self.patience = 0
        self.device = device
        self.key = key

        min_fn = lambda m0, m1: m0 <= m1
        max_fn = lambda m0, m1: m0 >= m1

        if key in ['bal_acc', 'acc', 'macro_f1']:
            self.op = max_fn
        else:
            self.op = min_fn
    
    def update(self, metric, epoch):
        c_metric = metric[self.key]
        if self.best_metric is None or self.op(c_metric, self.best_metric[self.key]):
            self.best_metric = metric
            self.best_epoch = epoch
            self.patience = 0

            return True
        else:
            self.patience += 1

            return False
    
    def state_dict(self):
        return {
            'best_metric': self.best_metric,
            'best_epoch': self.best_epoch,
            'patience': self.patience
        }

    def load_state_dict(self, state_dict):
        self.best_metric = state_dict['best_metric']
        self.best_epoch = state_dict['best_epoch']
        self.patience = state_dict['patience']

    def stats(self):
        return self.best_epoch, self.best_metric

def safe_load(model, state_dict):
    missing_keys, unexpected_keys =  model.load_state_dict(state_dict, strict=False)
    assert len(unexpected_keys) == 0

def memory_usage():
    allocated_memory = torch.cuda.memory_allocated() / 1e9  # Convert to GB
    cached_memory = torch.cuda.memory_reserved() / 1e9  # Convert to GB

    return allocated_memory, cached_memory

def T(w):
    return w.transpose(0, 1)
 
def linear_interpolate(frac, start, end):
    return frac * end + (1 - frac) * start

def log_interpolate(frac, start, end):
    gen = np.exp(np.log(start) * (1 - frac) + np.log(end) * frac)
    gen = np.where(gen < end, end, gen)
    return gen

def cosine_interpolate(frac, start, end):
    return start + (end - start) * (1 - np.cos(np.pi * frac)) / 2

def exp_interpolate(frac, start, end, k=10):
    return end + (start - end) * np.exp(-k * frac)