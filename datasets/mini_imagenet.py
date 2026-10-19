from .imagenet import _ImageNet

class _Mini_ImageNet(_ImageNet):
    subclasses = [6, 11, 13, 22, 23, 39, 47, 71, 76, 79, 90, 94,
            96, 97, 99, 105, 107, 113, 125, 130, 132, 144, 150,
            151, 207, 234, 235, 254, 277, 291, 301, 308, 309, 310,
            311, 314, 315, 319, 323, 327, 330, 334, 335, 347, 361,
            372, 397, 401, 407, 425, 428, 430, 437, 457, 462, 470,
            472, 483, 579, 609, 658, 701, 763, 768, 774, 776, 779,
            780, 815, 820, 833, 847, 907, 932, 933, 934, 937, 943,
            945, 947, 951, 954, 957, 980, 981, 988]

    def __init__(self, root, split='train', transform=None, cfg=None):
        super().__init__(root, split=split, transform=transform, cfg=cfg)

class Mini_ImageNet(_Mini_ImageNet):
    train_txt = "./datasets/ImageNet/imagenet_train.txt"
    val_txt = "./datasets/ImageNet/imagenet_test.txt"

    def __init__(self, root, split="train", transform=None, cfg=None):
        super().__init__(root, split=split, transform=transform, cfg=cfg)

class Mini_ImageNet_S(_Mini_ImageNet):
    test_txt = "./datasets/ImageNet/imagenet_s.txt"

    def __init__(self, root, split="test", transform=None, cfg=None):
        super().__init__(root, split=split, transform=transform, cfg=cfg)

class Mini_ImageNet_A(_Mini_ImageNet):
    test_txt = "./datasets/ImageNet/imagenet_a.txt"
    delimiter = ","

    def __init__(self, root, split="test", transform=None, cfg=None):
        super().__init__(root, split=split, transform=transform, cfg=cfg)

class Mini_ImageNet_R(_Mini_ImageNet):
    test_txt = "./datasets/ImageNet/imagenet_r.txt"

    def __init__(self, root, split="test", transform=None, cfg=None):
        super().__init__(root, split=split, transform=transform, cfg=cfg)

class Mini_ImageNet_V2(_Mini_ImageNet):
    test_txt = "./datasets/ImageNet/imagenet_v2.txt"

    def __init__(self, root, split="test", transform=None, cfg=None):
        super().__init__(root, split=split, transform=transform, cfg=cfg)

class Mini_ImageNet_LT(_Mini_ImageNet):
    train_txt = "./datasets/ImageNet/imagenet_LT_train.txt"
    val_txt = "./datasets/ImageNet/imagenet_LT_val.txt"

    def __init__(self, root, split="train", transform=None, cfg=None):
        super().__init__(root, split=split, transform=transform, cfg=cfg)