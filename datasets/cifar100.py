from collections import defaultdict
import os
import torchvision
import numpy as np

class IMBALANCECIFAR100(torchvision.datasets.CIFAR100):
    cls_num = 100
    val_per_class = 10
    random_seed = 0

    def __init__(self, root, imb_factor=None, random_seed=0, split="train",
                 transform=None, target_transform=None, download=True):
        
        train = True if split == "train" else False
        super().__init__(root, train, transform, target_transform, download)

        if train and imb_factor is not None:
            self.random_seed = random_seed
            img_num_list = self.get_img_num_per_cls(self.cls_num, imb_factor)
            self.gen_imbalanced_data(img_num_list)
            self.beton = os.path.join(root, "lt.beton")
        else:
            self.beton = os.path.join(root, f"{split}.beton")

        if split == "val" or split == "test":
            np_targets = np.array(self.targets)
            targets_idx = []
            for cls_idx in range(len(self.classes)):
                
                if split == "val":
                    f_cls_idx = np.where(np_targets == cls_idx)[0][:self.val_per_class]
                else:
                    f_cls_idx = np.where(np_targets == cls_idx)[0][self.val_per_class:]

                targets_idx.extend(f_cls_idx)
            
            self.targets = np_targets[targets_idx].tolist()
            self.data = self.data[targets_idx]
            self.targets_idx = targets_idx
        
        self.classnames = self.classes
        self.labels = self.targets
        self.cls_num_list = self.get_cls_num_list()
        self.num_classes = len(self.cls_num_list)
        
    def get_img_num_per_cls(self, cls_num, imb_factor):
        img_max = len(self.data) / cls_num

        img_num_per_cls = []
        for cls_idx in range(cls_num):
            num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
            img_num_per_cls.append(int(num))
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        rng = np.random.default_rng(self.random_seed)

        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            rng.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets
        
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


class CIFAR100(IMBALANCECIFAR100):
    def __init__(self, root, split="train", transform=None, cfg=None):
        super().__init__(root, imb_factor=None, split=split, transform=transform)

class CIFAR100C(IMBALANCECIFAR100):
    def __init__(self, root, cor_type, cor_level=5, transform=None, cfg=None):
        super().__init__(root, imb_factor=None, split="test", transform=transform)
        size = 10000
        dataset_raw = np.load(root + f'/CIFAR-100-C/{cor_type}.npy')
        dataset_raw = dataset_raw[(cor_level - 1)*size: cor_level*size][self.targets_idx]
        self.data = dataset_raw

class CIFAR100_LT(IMBALANCECIFAR100):
    def __init__(self, root, split="train", transform=None, cfg=None):
        super().__init__(root, imb_factor=0.01, split=split, transform=transform)