import os
from collections import defaultdict
from PIL import Image
from torch.utils.data import Dataset


class LT_Dataset(Dataset):
    train_txt = ""
    test_txt = ""
    val_txt = ""
    delimiter = " "
    subclasses = []

    def __init__(self, root, split='train', n_max=-1, transform=None):
        self.img_path = []
        self.labels = []
        self.split = split
        self.transform = transform
        self.n_max = n_max

        if split == 'train':
            self.txt = self.train_txt
        elif split == 'val':
            self.txt = self.val_txt
        else:
            self.txt = self.test_txt
        
        with open(self.txt) as f:
            for line in f:
                im_path = os.path.join(root, line.split(self.delimiter)[0])
                lbl = int(line.split(self.delimiter)[1])
                if len(self.subclasses) > 0 and lbl not in self.subclasses:
                    continue
                self.img_path.append(im_path)
                self.labels.append(lbl)

        self.labels = self.label_mapper(self.labels)

        if split == 'train' and self.n_max > 0:
            # sample n_max data for each class
            num_classes = len(list(set(self.labels)))
            cls_data_list = [list() for _ in range(num_classes)]
            for i, label in enumerate(self.labels):
                cls_data_list[label].append(i)
            
            new_img_path = list()
            new_labels = list()
            for i in range(num_classes):
                new_img_path.extend([self.img_path[j] for j in cls_data_list[i][:self.n_max]])
                new_labels.extend([self.labels[j] for j in cls_data_list[i][:self.n_max]])

            self.img_path = new_img_path
            self.labels = new_labels

        self.cls_num_list = self.get_cls_num_list()
        self.num_classes = len(self.cls_num_list)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        path = self.img_path[index]
        label = self.labels[index]

        with open(path, 'rb') as f:
            image = Image.open(f).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        return image, label
    
    def get_cls_num_list(self):
        counter = defaultdict(int)
        for label in self.labels:
            counter[label] += 1
        labels = list(counter.keys())
        labels.sort()
        cls_num_list = [counter[label] for label in labels]
        return cls_num_list

    def label_mapper(self, labels):
        if len(self.subclasses) == 0:
            return labels
        else:
            return [self.subclasses.index(label) for label in labels]
