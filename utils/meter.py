import torch
from torch import distributed as dist

class AverageMeter:
    """Compute and store the average and current value.

    Examples::
        >>> # 1. Initialize a meter to record loss
        >>> losses = AverageMeter()
        >>> # 2. Update meter after every mini-batch update
        >>> losses.update(loss_value, batch_size)
    """

    def __init__(self, ema=False):
        """
        Args:
            ema (bool, optional): apply exponential moving average.
        """
        self.ema = ema
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def all_reduce(self):    
        total = torch.tensor([self.avg, self.val], dtype=torch.float32, device='cuda')
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.avg, self.val = total.tolist()
        self.avg = self.avg / dist.get_world_size()    
        self.val = self.val / dist.get_world_size()

    def update(self, val, n=1):
        if isinstance(val, torch.Tensor):
            val = val.item()

        self.val = val
        self.sum += val * n
        self.count += n

        if self.ema:
            self.avg = val if self.count == n else self.avg * 0.9 + self.val * 0.1
        else:
            self.avg = self.sum / self.count