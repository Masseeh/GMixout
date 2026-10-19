import numpy as np
from PIL import ImageOps, ImageEnhance, ImageDraw, Image
import random
import torch
import torchvision.transforms.v2 as TF

try:
    import ffcv
    from ffcv.fields.decoders import IntDecoder
    import ffcv.transforms as ffcv_TF
    from ffcv.transforms.common import Squeeze
except ImportError:
    ffcv = None

def get_torch_transforms(cfg):
    resolution = cfg.resolution

    if cfg.backbone.startswith("CLIP"):
        mean = [0.48145466, 0.4578275, 0.40821073]
        std = [0.26862954, 0.26130258, 0.27577711]
    else:
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]

    tf_train = [
        TF.RandomResizedCrop(resolution), 
        TF.RandomHorizontalFlip()
    ]
    if cfg.randaug:
        tf_train += [TF.RandAugment(2, 10)]
    tf_train += [
        TF.ToImage(),
        TF.ToDtype(torch.float32, scale=True),
        TF.Normalize(mean, std),
    ]

    transform_train = TF.Compose(tf_train)

    transform_test = TF.Compose([
        TF.Resize(resolution * 8 // 7),
        TF.CenterCrop(resolution),
        TF.ToImage(),
        TF.ToDtype(torch.float32, scale=True),
        TF.Normalize(mean, std),
    ])

    return transform_train, transform_test

def get_ffcv_transforms(cfg, device):

    resolution = cfg.resolution

    if cfg.backbone.startswith("CLIP"):
        mean = [0.48145466, 0.4578275, 0.40821073]
        std = [0.26862954, 0.26130258, 0.27577711]
    else:
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]  

    label_pipeline = [IntDecoder(), ffcv_TF.ToTensor(), ffcv_TF.ToDevice(device), Squeeze()]

    aug_image_pipeline = [
        ffcv_TF.ToTensor(),
        ffcv_TF.ToDevice(device, non_blocking=True),
        ffcv_TF.ToTorchImage(),
        TF.RandomResizedCrop(resolution), 
        TF.RandomHorizontalFlip(),
    ]
    if cfg.randaug:
        aug_image_pipeline += [TF.RandAugment(2, 10)]
    aug_image_pipeline += [
        TF.ToDtype(torch.float32, scale=True),
        TF.Normalize(mean, std),
    ]

    basic_image_pipeline = [
        ffcv_TF.ToTensor(),
        ffcv_TF.ToDevice(device, non_blocking=True),
        ffcv_TF.ToTorchImage(),
        TF.Resize(resolution * 8 // 7),
        TF.CenterCrop(resolution),
        TF.ToDtype(torch.float32, scale=True),
        TF.Normalize(mean, std),
    ]

    return aug_image_pipeline, basic_image_pipeline, label_pipeline


class MultipleApply:
    """Apply a list of transformations to an image and get multiple transformed images.

    Args:
        transforms (list or tuple): list of transformations

    Example:
        
        >>> transform1 = T.Compose([
        ...     ResizeImage(256),
        ...     T.RandomCrop(224)
        ... ])
        >>> transform2 = T.Compose([
        ...     ResizeImage(256),
        ...     T.RandomCrop(224),
        ... ])
        >>> multiply_transform = MultipleApply([transform1, transform2])
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image):
        return [t(image) for t in self.transforms]


def AutoContrast(img, _):
    return ImageOps.autocontrast(img)


def Brightness(img, v):
    assert v >= 0.0
    return ImageEnhance.Brightness(img).enhance(v)


def Color(img, v):
    assert v >= 0.0
    return ImageEnhance.Color(img).enhance(v)


def Contrast(img, v):
    assert v >= 0.0
    return ImageEnhance.Contrast(img).enhance(v)


def Equalize(img, _):
    return ImageOps.equalize(img)


def Invert(img, _):
    return ImageOps.invert(img)


def Identity(img, v):
    return img


def Posterize(img, v):  # [4, 8]
    v = int(v)
    v = max(1, v)
    return ImageOps.posterize(img, v)


def Rotate(img, v):  # [-30, 30]
    # assert -30 <= v <= 30
    # if random.random() > 0.5:
    #    v = -v
    return img.rotate(v)


def Sharpness(img, v):  # [0.1,1.9]
    assert v >= 0.0
    return ImageEnhance.Sharpness(img).enhance(v)


def ShearX(img, v):  # [-0.3, 0.3]
    # assert -0.3 <= v <= 0.3
    # if random.random() > 0.5:
    #    v = -v
    return img.transform(img.size, Image.AFFINE, (1, v, 0, 0, 1, 0))


def ShearY(img, v):  # [-0.3, 0.3]
    # assert -0.3 <= v <= 0.3
    # if random.random() > 0.5:
    #    v = -v
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, v, 1, 0))


def TranslateX(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    # assert -0.3 <= v <= 0.3
    # if random.random() > 0.5:
    #    v = -v
    v = v * img.size[0]
    return img.transform(img.size, Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateXabs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    # assert v >= 0.0
    # if random.random() > 0.5:
    #    v = -v
    return img.transform(img.size, Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateY(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    # assert -0.3 <= v <= 0.3
    # if random.random() > 0.5:
    #    v = -v
    v = v * img.size[1]
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, v))


def TranslateYabs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    # assert 0 <= v
    # if random.random() > 0.5:
    #    v = -v
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, v))


def Solarize(img, v):  # [0, 256]
    assert 0 <= v <= 256
    return ImageOps.solarize(img, v)


def Cutout(img, v):  # [0, 60] => percentage: [0, 0.2] => change to [0, 0.5]
    assert 0.0 <= v <= 0.5
    if v <= 0.:
        return img

    v = v * img.size[0]
    return CutoutAbs(img, v)


def CutoutAbs(img, v):  # [0, 60] => percentage: [0, 0.2]
    # assert 0 <= v <= 20
    if v < 0:
        return img
    w, h = img.size
    x0 = np.random.uniform(w)
    y0 = np.random.uniform(h)

    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = min(w, x0 + v)
    y1 = min(h, y0 + v)

    xy = (x0, y0, x1, y1)
    color = (125, 123, 114)
    # color = (0, 0, 0)
    img = img.copy()
    ImageDraw.Draw(img).rectangle(xy, color)
    return img


def augment_list():
    l = [
        (AutoContrast, 0, 1),
        (Brightness, 0.05, 0.95),
        (Color, 0.05, 0.95),
        (Contrast, 0.05, 0.95),
        (Equalize, 0, 1),
        (Identity, 0, 1),
        (Posterize, 4, 8),
        (Rotate, -30, 30),
        (Sharpness, 0.05, 0.95),
        (ShearX, -0.3, 0.3),
        (ShearY, -0.3, 0.3),
        (Solarize, 0, 256),
        (TranslateX, -0.3, 0.3),
        (TranslateY, -0.3, 0.3)
    ]
    return l


class RandAugment:
    def __init__(self, n, m):
        self.n = n
        self.m = m  # [0, 30] in fixmatch, deprecated.
        self.augment_list = augment_list()

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)
        for op, min_val, max_val in ops:
            val = min_val + float(max_val - min_val) * random.random()
            img = op(img, val)
        cutout_val = random.random() * 0.5
        img = Cutout(img, cutout_val)  # for fixmatch
        return img