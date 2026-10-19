import torch
import numpy as np
import random
from torch.utils.data.sampler import Sampler
from torch.utils.data.distributed import DistributedSampler
from utils.util import breakpoint

class InfiniteFFCVLoader:
    def __init__(self, loader):
        self.loader = loader
        self.iter = None

    def __iter__(self):
        self.iter = iter(self.loader)
        return self

    def _reload(self):
        self.iter = iter(self.loader)
    
    def __next__(self):
        try:
            out = next(self.iter)
        except StopIteration:
            self._reload()
            out = next(self.iter)
        return out

class _InfiniteSampler(torch.utils.data.Sampler):
    """Wraps another Sampler to yield an infinite stream."""

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            for batch in self.sampler:
                yield batch

class InfiniteDataLoader:
    def __init__(self, dataset, batch_size, num_workers, seed, ddp=False, class_balanced=False):
        super().__init__()

        if class_balanced:
            sampler = ClassAwareSampler(dataset)
        else:
            if ddp:
                batch_sampler = DistributedSampler(
                    dataset,
                    shuffle=True,
                    seed=seed
                )
            else:
                sampler = torch.utils.data.RandomSampler(dataset, replacement=True)
 
        if not ddp:
            batch_sampler = torch.utils.data.BatchSampler(
                sampler, batch_size=batch_size, drop_last=False
            )

        self._infinite_iterator = iter(
            torch.utils.data.DataLoader(
                dataset,
                num_workers=num_workers,
                pin_memory=True,
                batch_sampler=_InfiniteSampler(batch_sampler),
            )
        )

        self.sampler = batch_sampler

    def __iter__(self):
        while True:
            yield next(self._infinite_iterator)

    def __len__(self):
        raise ValueError

class RandomCycleIter:

    def __init__ (self, data, test_mode=False):
        self.data_list = list(data)
        self.length = len(self.data_list)
        self.i = self.length - 1
        self.test_mode = test_mode
        
    def __iter__ (self):
        return self
    
    def __next__ (self):
        self.i += 1
        
        if self.i == self.length:
            self.i = 0
            if not self.test_mode:
                random.shuffle(self.data_list)
            
        return self.data_list[self.i]
    
def class_aware_sample_generator(cls_iter, data_iter_list, n, num_samples_cls=1):

    i = 0
    j = 0
    while i < n:
        
#         yield next(data_iter_list[next(cls_iter)])
        
        if j >= num_samples_cls:
            j = 0
    
        if j == 0:
            temp_tuple = next(zip(*[data_iter_list[next(cls_iter)]]*num_samples_cls))
            yield temp_tuple[j]
        else:
            yield temp_tuple[j]
        
        i += 1
        j += 1

class ClassAwareSampler(Sampler):
    def __init__(self, data_source, num_samples_cls=4,):
        # pdb.set_trace()
        num_classes = data_source.num_classes
        self.class_iter = RandomCycleIter(range(num_classes))
        cls_data_list = [list() for _ in range(num_classes)]
        for i, label in enumerate(data_source.labels):
            cls_data_list[label].append(i)
        self.data_iter_list = [RandomCycleIter(x) for x in cls_data_list]
        self.num_samples = max([len(x) for x in cls_data_list]) * len(cls_data_list)
        # self.num_samples = sum([len(x) for x in cls_data_list])
        self.num_samples_cls = num_samples_cls
        
    def __iter__ (self):
        return class_aware_sample_generator(self.class_iter, self.data_iter_list,
                                            self.num_samples, self.num_samples_cls)
    
    def __len__ (self):
        return self.num_samples
    

class DownSampler(Sampler):
    def __init__(self, data_source, n_max=100):
        self.num_classes = data_source.num_classes
        self.cls_data_list = [list() for _ in range(self.num_classes)]
        for i, label in enumerate(data_source.labels):
            self.cls_data_list[label].append(i)

        self.n_max = n_max
        self.cls_num_list = [min(n_max, len(x)) for x in self.cls_data_list]
        self.num_samples = sum(self.cls_num_list)
        
    def __iter__(self):
        data_list = []
        for y in range(self.num_classes):
            random.shuffle(self.cls_data_list[y])
            data_list.extend(self.cls_data_list[y][:self.n_max])
        random.shuffle(data_list)
        
        for i in range(self.num_samples):
            yield data_list[i]

    def __len__(self):
        return self.num_samples


class ReSampler(Sampler):
    def __init__(self, data_source, n_max=100):
        # pdb.set_trace()
        self.num_classes = data_source.num_classes

        cls_data_list = [list() for _ in range(self.num_classes)]
        for i, y in enumerate(data_source.labels):
            cls_data_list[y].append(i)
        self.data_iter_list = [RandomCycleIter(x) for x in cls_data_list]
        cls_num_list = [len(x) for x in cls_data_list]

        self.sampled_cls_num_list = [min(n_max, n) for n in cls_num_list]
        cls_id_list = []
        for y in range(self.num_classes):
            cls_id_list.extend([y] * self.sampled_cls_num_list[y])
        self.cls_iter = RandomCycleIter(cls_id_list)

        self.num_samples = len(data_source.labels)
        
    def __iter__(self):
        for _ in range(self.num_samples):
            yield next(self.data_iter_list[next(self.cls_iter)])

    def __len__(self):
        return self.num_samples

