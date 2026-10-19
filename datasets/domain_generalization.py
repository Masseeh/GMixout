from collections import defaultdict
import os
import torchvision
import numpy as np

class MultiDomainDataset(torchvision.datasets.ImageFolder):
    val_pct = 0.3  # 30% data for validation

    def __init__(self, root, split="train",
                 transform=None, target_transform=None):
        super().__init__(root, transform=transform, target_transform=target_transform)

        if split != "all":
            np_targets = np.array(self.targets)
            targets_idx = []
            for cls_idx in range(len(self.classes)):
                val_per_class = int(self.val_pct * np.sum(np_targets == cls_idx))

                if split == "val" or split == "test":
                    f_cls_idx = np.where(np_targets == cls_idx)[0][:val_per_class]
                else:
                    f_cls_idx = np.where(np_targets == cls_idx)[0][val_per_class:]

                targets_idx.extend(f_cls_idx)
            
            self.targets = np_targets[targets_idx].tolist()
            self.imgs = [self.imgs[i] for i in targets_idx]
        
        self.classnames = self.classes
        self.labels = self.targets
        self.samples = self.imgs
        self.cls_num_list = self.get_cls_num_list()
        self.num_classes = len(self.cls_num_list)
        
    def get_cls_num_list(self):
        counter = defaultdict(int)
        for label in self.labels:
            counter[label] += 1
        labels = list(counter.keys())
        labels.sort()
        cls_num_list = [counter[label] for label in labels]
        return cls_num_list


    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img, target, index

class DomainNet_Real_Train(MultiDomainDataset):
    domain = "real"

    def __init__(self, root, split="train", transform=None, cfg=None):
        p = os.path.join(root, self.domain)
        super().__init__(p, split=split, transform=transform)

# class DomainNet_Quickdraw_Train(MultiDomainDataset):
#     domain = "quickdraw"

#     def __init__(self, root, split="train", transform=None, cfg=None):
#         p = os.path.join(root, self.domain)
#         super().__init__(p, split=split, transform=transform)

class DomainNet_Sketch_Train(MultiDomainDataset):
    domain = "sketch"

    def __init__(self, root, split="train", transform=None, cfg=None):
        p = os.path.join(root, self.domain)
        super().__init__(p, split=split, transform=transform)

class DomainNet_Clipart_Train(MultiDomainDataset):
    domain = "clipart"

    def __init__(self, root, split="train", transform=None, cfg=None):
        p = os.path.join(root, self.domain)
        super().__init__(p, split=split, transform=transform)

class DomainNet_Painting_Train(MultiDomainDataset):
    domain = "painting"

    def __init__(self, root, split="train", transform=None, cfg=None):
        p = os.path.join(root, self.domain)
        super().__init__(p, split=split, transform=transform)

class DomainNet_Real(MultiDomainDataset):
    domain = "real"

    def __init__(self, root, split="all", transform=None, cfg=None):
        p = os.path.join(root, self.domain)
        super().__init__(p, split=split, transform=transform)

# class DomainNet_Quickdraw(MultiDomainDataset):
#     domain = "quickdraw"

#     def __init__(self, root, split="all", transform=None, cfg=None):
#         p = os.path.join(root, self.domain)
#         super().__init__(p, split="all", transform=transform)

class DomainNet_Sketch(MultiDomainDataset):
    domain = "sketch"

    def __init__(self, root, split="all", transform=None, cfg=None):
        p = os.path.join(root, self.domain)
        super().__init__(p, split=split, transform=transform)

class DomainNet_Painting(MultiDomainDataset):
    domain = "painting"

    def __init__(self, root, split="all", transform=None, cfg=None):
        p = os.path.join(root, self.domain)
        super().__init__(p, split=split, transform=transform)

class DomainNet_Clipart(MultiDomainDataset):
    domain = "clipart"

    def __init__(self, root, split="all", transform=None, cfg=None):
        p = os.path.join(root, self.domain)
        super().__init__(p, split=split, transform=transform)