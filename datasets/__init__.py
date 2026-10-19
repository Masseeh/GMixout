from .cifar100 import *
from .cifar10 import *
from .imagenet import *
from .tinyimagenet import *
from .domain_generalization import *
from .mini_imagenet import *
from .inat2018 import *

try:
    import wilds
    from .wilds_datasets import *
except ImportError:
    wilds = None

from .augmentation import get_ffcv_transforms, get_torch_transforms

COMMON_CORRUPTIONS = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur',
                            'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog', 'brightness', 'contrast', 
                            'elastic_transform', 'pixelate', 'jpeg_compression']
