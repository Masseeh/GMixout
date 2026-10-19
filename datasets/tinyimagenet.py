from collections import defaultdict
import os
import torchvision
import numpy as np

class IMBALANCETinyImageNet(torchvision.datasets.ImageFolder):
    classnames_txt = "./datasets/ImageNet/classnames.txt"
    cls_num = 200
    val_pct = 0.2  # 20% data for validation

    def __init__(self, root, train=True, imb_factor=None, random_seed=0, split="train",
                 transform=None, target_transform=None):
        super().__init__(root, transform=transform, target_transform=target_transform)

        if train and imb_factor is not None:
            self.random_seed = random_seed
            img_num_list = self.get_img_num_per_cls(self.cls_num, imb_factor)
            self.gen_imbalanced_data(img_num_list)
            self.beton = os.path.join(root, "lt.beton")
        else:
            self.beton = os.path.join(root, f"{split}.beton")

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
            self.beton = os.path.join(root, f"{split}.beton")
        
        self.classnames = self.read_classnames()
        self.labels = self.targets
        self.samples = self.imgs
        self.cls_num_list = self.get_cls_num_list()
        self.num_classes = len(self.cls_num_list)
        
    def get_img_num_per_cls(self, cls_num, imb_factor):
        img_max = len(self.samples) / cls_num

        img_num_per_cls = []
        for cls_idx in range(cls_num):
            num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
            img_num_per_cls.append(int(num))
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        rng = np.random.default_rng(self.random_seed)

        new_imgs = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            rng.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_imgs.extend([self.imgs[i] for i in selec_idx])
            new_targets.extend([the_class, ] * the_img_num)

        self.imgs = new_imgs
        self.targets = new_targets
        
    def get_cls_num_list(self):
        counter = defaultdict(int)
        for label in self.labels:
            counter[label] += 1
        labels = list(counter.keys())
        labels.sort()
        cls_num_list = [counter[label] for label in labels]
        return cls_num_list
    
    def read_classnames(self):
        folder_names = set(self.classes)

        classnames = []
        with open(self.classnames_txt, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(" ")
                folder = line[0]
                if folder not in folder_names:
                    continue
                classname = " ".join(line[1:])
                classnames.append(classname)
        return classnames

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img, target, index


class TinyImageNet(IMBALANCETinyImageNet):
    def __init__(self, root, split="train", transform=None, cfg=None):
        train = True if split == "train" else False
        p = os.path.join(root, "train" if train else "val")
        super().__init__(p, train=train, imb_factor=None, split=split, transform=transform)

class TinyImageNetC(IMBALANCETinyImageNet):
    def __init__(self, root, cor_type, cor_level=5, transform=None, cfg=None):
        p = os.path.join(root, "tinyimagenet-c", cor_type, str(cor_level))
        super().__init__(p, train=False, split="all", imb_factor=None, transform=transform)

class ImageNetC(IMBALANCETinyImageNet):
    cls_num = 1000
    def __init__(self, root, cor_type, cor_level=5, transform=None, cfg=None):
        p = os.path.join(root, "imagenet-c", cor_type, str(cor_level))
        super().__init__(p, train=False, split="all", imb_factor=None, transform=transform)

class TinyImageNet_LT(IMBALANCETinyImageNet):
    def __init__(self, root, split="train", transform=None, cfg=None):
        train = True if split == "train" else False
        p = os.path.join(root, "train" if train else "val")
        super().__init__(p, train=train, imb_factor=0.01, split=split, transform=transform)