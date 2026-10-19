# import cv2
import functools
import os
import copy
import json
import math
import time
import datetime
import numpy as np
from tqdm import tqdm
import scipy as sp
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from clip import clip
import datasets
from datasets import get_torch_transforms, get_ffcv_transforms
import logging

try:
    import ffcv
    from ffcv.loader import Loader, OrderOption
except ImportError:
    ffcv = None
    
from datasets.fewshot_datasets import build_fewshot_dataset, fewshot_datasets
from datasets.fewshot_cls2names import cl2names as fewshot_cls2names
from datasets import COMMON_CORRUPTIONS
from models import *
from models.clip_text import CLIP_Text
from timm import create_model as timm_create_model
from timm.models.vision_transformer import vit_base_patch16_224, vit_base_patch16_384, vit_large_patch16_224

from utils.meter import AverageMeter
from utils.ema import update_ema, requires_grad
from utils.losses import *
from utils.evaluator import Evaluator
from utils.templates import ZEROSHOT_TEMPLATES
from utils.util import ModelSelection, safe_load, DummyWriter, breakpoint, memory_usage
from utils.distributed import from_ddp
from utils.samplers import InfiniteFFCVLoader, ClassAwareSampler
from utils.adamSPD import AdamSPD
from utils.adalora import RankAllocator, compute_orth_regu
from models.peft_modules import DenseGMixout, PACELoRA

def load_clip_to_cpu(backbone_name, prec, pretrained=True):
    backbone_name = backbone_name.lstrip("CLIP-")
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu").eval()

    model = clip.build_model(state_dict or model.state_dict(), pretrained=pretrained)

    assert prec in ["fp16", "fp32", "bf16"]
    if prec == "fp32" or prec == "bf16":
        # CLIP's default precision is fp16
        model.float()

    return model

def load_vit_to_cpu(backbone_name, prec, pretrained=True):

    if backbone_name == "IN21K-ViT-B/16":
        model = vit_base_patch16_224(pretrained=pretrained).eval()
    elif backbone_name == "IN1K-ViT-B/16":
        model = timm_create_model('vit_base_patch16_224', pretrained=pretrained)
    elif backbone_name == "IN21K-ViT-B/16@384px":
        model = vit_base_patch16_384(pretrained=pretrained).eval()
    elif backbone_name == "IN21K-ViT-L/16":
        model = vit_large_patch16_224(pretrained=pretrained).eval()
    elif backbone_name == "IN1K-ViT-S/16":
        model = timm_create_model('vit_small_patch16_224', pretrained=pretrained)
    elif backbone_name == "DINO-ViT-B/16":
        model = timm_create_model('vit_small_patch16_224.dino', pretrained=pretrained)
        # model = timm_create_model('vit_base_patch16_224.dino', pretrained=pretrained)

    assert prec in ["fp16", "fp32", "bf16"]
    if prec == "fp16":
        # ViT's default precision is fp32
        model.half()
    
    return model

class Trainer:
    def __init__(self, cfg, ddp_args):

        self.cfg = cfg
        self.ddp_args = ddp_args
        self._writer = None
        self.ddp = False
        
        if not cfg.test_only and dist.get_world_size() > 1:
            self.ddp = True

        if self.ddp:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cuda" if cfg.gpu is None else f"cuda:{cfg.gpu}")

        if cfg.o3:
            torch.set_float32_matmul_precision('high')

        self.ptdtype = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[cfg.prec]
        logging.info(f"precision: {self.ptdtype}")
        
        self.build_data_loader()
        self.build_model()
        self.build_training_artifacts()

        if not cfg.test_only:
            logging.info(f"Total training points: {sum(self.cls_num_list)}")
            logging.info(f"Number of remaning epochs: {cfg.num_epochs - self.starting_epoch}")
            logging.info(f"Number of steps per epoch: {self.one_epoch}")
            logging.info(f"Number of total steps: {self.n_steps}")

        self.test_evaluator = Evaluator(cfg, self.cls_num_list, self.eval_test_fn, stout=True)
        self.val_evaluator = Evaluator(cfg, self.cls_num_list, stout=False)

    def build_data_loader(self):
        cfg = self.cfg
        use_ffcv = cfg.use_ffcv
        root = cfg.root

        logging.info(f"Setting up indomain dataset {cfg.dataset}")

        if not use_ffcv:
            transform_train, transform_test = get_torch_transforms(self.cfg)
            aug_image_pipeline, basic_image_pipeline, label_pipeline = None, None, None
        else:
            transform_train, transform_test = None, None
            aug_image_pipeline, basic_image_pipeline, label_pipeline = get_ffcv_transforms(cfg, self.device)
        
        train_dataset = getattr(datasets, cfg.dataset)(root, split='train', transform=transform_train, cfg=cfg)
        val_dataset = getattr(datasets, cfg.dataset)(root, split='val', transform=transform_test, cfg=cfg)
        test_dataset = getattr(datasets, cfg.dataset)(root, split='test', transform=transform_test, cfg=cfg)

        logging.info(f"Train dataset size: {len(train_dataset)}")
        logging.info(f"Val dataset size: {len(val_dataset)}")
        logging.info(f"Test dataset size: {len(test_dataset)}")

        self.project_logits_fn = getattr(test_dataset, 'project_logits', None)
        self.project_labels_fn = getattr(test_dataset, 'project_label', None)
        self.eval_test_fn = getattr(test_dataset, 'eval', None)

        if use_ffcv:
            assert hasattr(train_dataset, 'beton'), "FFCV dataset must have a beton file"
            assert hasattr(val_dataset, 'beton'), "FFCV dataset must have a beton file"
            assert hasattr(test_dataset, 'beton'), "FFCV dataset must have a beton file"

        self.num_classes = train_dataset.num_classes
        self.cls_num_list = train_dataset.cls_num_list
        self.classnames = train_dataset.classnames
        if cfg.class_balanced: 
            assert use_ffcv == False, "Class balanced sampling is not supported with FFCV"
            assert self.ddp, "Class balanced sampling is not supported with DDP"
            logging.info("Balanced sampling with class frequencies")

        logging.info(f"Imbalance ratio: {max(self.cls_num_list) / min(self.cls_num_list)}")

        if not cfg.test_only:

            if use_ffcv:
                order = OrderOption.QUASI_RANDOM if "ImageNet" in cfg.dataset else OrderOption.RANDOM
                self.train_loader = InfiniteFFCVLoader(Loader(fname=train_dataset.beton, os_cache=False, batch_size=cfg.batch_size,
                                    num_workers=cfg.num_workers,
                                    order=order,
                                    drop_last=True,
                                    seed=cfg.seed,
                                    pipelines={'image': aug_image_pipeline,
                                                'label': copy.deepcopy(label_pipeline)}))
            else:
                if self.ddp:
                    train_sampler = DistributedSampler(train_dataset, shuffle=True, seed=cfg.seed, drop_last=True)
                    shuffle = None
                else:
                    if cfg.class_balanced:
                        train_sampler = ClassAwareSampler(train_dataset)
                    else:
                        train_sampler = None
                    shuffle = True
                
                self.train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=shuffle, num_workers=cfg.num_workers, pin_memory=True, sampler=train_sampler, persistent_workers=True)

        if use_ffcv:
            self.val_loader = Loader(fname=val_dataset.beton, os_cache=False, batch_size=cfg.test_batch_size,
                                    num_workers=cfg.num_workers,
                                    order=OrderOption.SEQUENTIAL,
                                    drop_last=False,
                                    pipelines={'image': copy.deepcopy(basic_image_pipeline),
                                                'label': copy.deepcopy(label_pipeline)})
            
            self.test_loader = Loader(fname=test_dataset.beton, os_cache=False, batch_size=cfg.test_batch_size,
                                    num_workers=cfg.num_workers,
                                    order=OrderOption.SEQUENTIAL,
                                    drop_last=False,
                                    pipelines={'image': copy.deepcopy(basic_image_pipeline),
                                                'label': copy.deepcopy(label_pipeline)})
        else:
            val_sampler = DistributedSampler(val_dataset, shuffle=False, drop_last=True) if self.ddp else None
            shuffle = None if self.ddp else False

            self.val_loader = DataLoader(val_dataset,
                batch_size=cfg.test_batch_size, shuffle=shuffle,
                num_workers=cfg.num_workers, pin_memory=True, sampler=val_sampler)

            # do not use DDP for test loader
            self.test_loader = DataLoader(test_dataset,
                batch_size=cfg.test_batch_size, shuffle=False,
                num_workers=cfg.num_workers, pin_memory=True)

        self.one_epoch = len(self.train_loader) if not cfg.test_only else 1
        self.n_steps = self.one_epoch * cfg.num_epochs
        self.starting_epoch = 0
        
        if cfg.dy_mask:
            logging.info(f"Maskr is enabled, {math.ceil(self.one_epoch / cfg.mask_refresh)} refresh per epoch")        
            
    def setup_corruption_loader(self, corruption):
        cfg = self.cfg
        root = cfg.root

        _, transform_test = get_torch_transforms(self.cfg)

        assert corruption in COMMON_CORRUPTIONS, f"Corruption {corruption} is not supported"
        assert cfg.dataset in ["CIFAR10", "CIFAR100", "TinyImageNet", "ImageNet"], "Corruption is only available for CIFAR10, CIFAR100, TinyImageNet and ImageNet"

        test_dataset = getattr(datasets, cfg.dataset + "C")(root, cor_type=corruption, cor_level=cfg.corruption_level, transform=transform_test, cfg=cfg)

        self.test_loader = DataLoader(test_dataset,
            batch_size=cfg.test_batch_size, shuffle=False,
            num_workers=cfg.num_workers, pin_memory=True)

    def setup_fewshot_loader(self, dataset_name):
        cfg = self.cfg
        root = cfg.root

        _, transform_test = get_torch_transforms(self.cfg)

        assert dataset_name.lower() in fewshot_datasets, f"{dataset_name} is not supported"
        test_dataset = build_fewshot_dataset(dataset_name, root, transform=transform_test, mode='test')

        logging.info(f"Setting up loader for {dataset_name}")

        self.test_loader = DataLoader(test_dataset,
            batch_size=cfg.test_batch_size, shuffle=False,
            num_workers=cfg.num_workers, pin_memory=True)
        
        self.classnames = fewshot_cls2names[dataset_name.lower()]
        self.num_classes = len(self.classnames)

        logging.info("Re-build classfication head with {} classes".format(self.num_classes))

        self.init_head_text_feat()

    def setup_outdomain_loader(self, dataset_name):
        cfg = self.cfg
        root = cfg.root

        _, transform_test = get_torch_transforms(self.cfg)

        test_dataset = getattr(datasets, dataset_name)(root, split='test', transform=transform_test, cfg=cfg)
        logging.info(f"Setting up loader for {dataset_name} with {len(test_dataset)} samples")

        self.test_loader = DataLoader(test_dataset,
            batch_size=cfg.test_batch_size, shuffle=False,
            num_workers=cfg.num_workers, pin_memory=True)

        self.project_logits_fn = getattr(test_dataset, 'project_logits', None)
        self.project_labels_fn = getattr(test_dataset, 'project_labels', None)

    def build_model(self):
        cfg = self.cfg
        num_classes = self.num_classes

        logging.info("Building model")

        if cfg.backbone.startswith("CLIP"):
            logging.info(f"Loading CLIP (backbone: {cfg.backbone})")
            clip_model = load_clip_to_cpu(cfg.backbone, cfg.prec, pretrained=True)
            self.model = PeftModelFromCLIP(cfg, clip_model, num_classes)
            self.model.to(self.device)
            self.text_encoder = CLIP_Text(clip_model)

        elif cfg.backbone.startswith("IN21K-ViT") or cfg.backbone.startswith("IN1K-ViT") or cfg.backbone.startswith("DINO-ViT"):
            logging.info(f"Loading ViT (backbone: {cfg.backbone})")

            assert cfg.init_head == "lp", "Only 'lp' initialization is supported for ViT"

            vit_model = load_vit_to_cpu(cfg.backbone, cfg.prec, pretrained=True)
            self.model = PeftModelFromViT(cfg, vit_model, num_classes)
            self.model.to(self.device)


        self.tuner = self.model.tuner
        self.head = self.model.head
        self.averaged_model = None

        if cfg.init_head == "text_feat":
            self.init_head_text_feat()

    def build_training_artifacts(self):
        cfg = self.cfg

        if not cfg.test_only:

            self.build_moving_average()
            self.build_model_checkpoint()

            if cfg.o3:
                logging.info(f"Compiling model with torch.compile")
                logging.warning("torch.compile may cause issues on mask refresh model, use with caution")
                self.model.compile()

            self.build_trainbles()
            self.build_optimizer()
            self.build_criterion()
            
            torch.cuda.empty_cache()

            if cfg.resume:
                self.resume_training()

            if self.ddp: self.model = DDP(self.model.to(self.device), device_ids=[self.ddp_args.gpu])

    def build_moving_average(self):
        cfg = self.cfg

        if cfg.bma:
            assert cfg.ema > 0, "BMA requires ema > 0"

        if cfg.ema > 0:
            if cfg.bma:
                logging.info(f"Building BMA model with beta parameter {cfg.ema}")
            else:
                logging.info(f"Building EMA model with decay {cfg.ema}")
            self.averaged_model = copy.deepcopy(self.model).to(self.device)  # Create an EMA of the model for use after training
            requires_grad(self.averaged_model, False)
            self.averaged_model.eval()
            update_ema(self.averaged_model, from_ddp(self.model), decay=0)  # Initialize EMA with synced weights
        else:
            logging.info("No EMA model is used")

    def build_model_checkpoint(self):
        cfg = self.cfg

        self.current_ckp = ModelSelection(device=self.device, key=cfg.model_selection)
        if self.averaged_model is not None:
            self.ema_ckp = ModelSelection(device=self.device, key=cfg.model_selection)
        else:
            self.ema_ckp = None

    def build_lr_scheduler(self, n_steps, optimizer, lr):
        cfg = self.cfg
        self.warmup_steps = int(cfg.warmup_steps * self.one_epoch)
        
        warmup = np.interp(np.arange(1+self.warmup_steps), [0, self.warmup_steps], [0.0, 1])
        ni = n_steps - self.warmup_steps

        # scheduler phase
        if cfg.scheduler == "cosine":
            xx = np.linspace(0, 1, ni, endpoint=True)  # Ensure it goes from 0 to 1
            cosine = (np.cos(np.pi * xx) + 1) / 2
            lr_schedule = np.concatenate([warmup, cosine])
            logging.info(f"cosine scheduler with warmup_steps={self.warmup_steps} and max lr: {lr}, min lr: {lr * lr_schedule[-1]}")
        elif cfg.scheduler == "linear":
            xx = np.linspace(0, 1, ni, endpoint=True)  # Ensure it goes from 0 to 1
            lr_schedule = np.concatenate([warmup, 1 - xx])
            logging.info(f"linear scheduler with warmup_steps={self.warmup_steps} and max lr: {lr}, min lr: {lr * lr_schedule[-1]}")
        elif cfg.scheduler == "sd": # stable-decay
            decay_steps = self.warmup_steps
            decay = np.interp(np.arange(decay_steps+1), [0, decay_steps], [1.0, 0.0])
            stable = np.ones(ni - len(decay))
            lr_schedule = np.concatenate([warmup, stable, decay])
            logging.info(f"stable-decay scheduler with warmup_steps={self.warmup_steps}, decay_step={len(decay)} and max lr: {lr}, min lr: {lr * lr_schedule[-1]}")
        else:
            lr_schedule = np.concatenate([warmup, np.ones(ni)])
            logging.info(f"constant scheduler with warmup_steps={self.warmup_steps} and lr: {lr}")

        lr_lambda = lambda x: lr_schedule[x]
        sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return sched

    def build_optimizer(self):
        cfg = self.cfg
        lr = cfg.lr * cfg.lrh_factor

        if cfg.optim == "adam":
            optim = torch.optim.AdamW(self.params_to_optimize,
                                        lr=lr, weight_decay=cfg.wd, betas=(cfg.momentum, 0.999), fused=True)
        elif cfg.optim == "adamSPD":
            
            # add pre to existing params, wd should be 1.5 for adamSPD
            if not (cfg.lora or cfg.lora_mlp):
                for param_group in self.params_to_optimize:
                    param_group['pre'] = copy.deepcopy(param_group['params'])
            else:
                for param_group in self.params_to_optimize:
                    param_group['pre'] = None

            optim = AdamSPD(self.params_to_optimize,
                            lr=lr, weight_decay=cfg.wd, betas=(cfg.momentum, 0.999))

        else:
            optim = torch.optim.SGD(self.params_to_optimize,
                                        lr=lr, weight_decay=cfg.wd, momentum=cfg.momentum, fused=True)

        
        self.optim = optim
        self.sched = self.build_lr_scheduler(self.n_steps, self.optim, lr)
        self.scaler = GradScaler(enabled=(cfg.prec =='fp16'))

        logging.info(f"setting up optimizer {optim.__class__.__name__}, scheduler and scaler")

    def build_trainbles(self):
        cfg = self.cfg

        for param in self.model.parameters():
            param.requires_grad_(False)
        
        logging.info("Turning on gradients in the tuner")
        for param_name, param in self.tuner.named_parameters():
            param.requires_grad_(True)
        
        params_to_optimize = [{"params": [p for p in self.tuner.parameters() if p.requires_grad == True]}]

        if cfg.head_tuning:
            logging.info(f"Turning on gradients in the head")
            for param in self.head.optim_params():
                param.requires_grad_(True)
            
            params_to_optimize.extend([{"params": [p for p in self.head.parameters() if p.requires_grad == True], "lr": cfg.lr * cfg.lrh_factor}])

        # print parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad == True)
        logging.info(f"Total params: {total_params}")
        logging.info(f"Tuned params: {trainable_params} -> {trainable_params / total_params * 100:.2f}% , rank {self.tuner.adapter_dim}")

        self.params_to_optimize = params_to_optimize

    def build_criterion(self):
        cfg = self.cfg

        cls_num_list = torch.Tensor(self.cls_num_list).to(self.device)

        if cfg.loss_type == "CE":
            self.criterion = nn.CrossEntropyLoss(label_smoothing=cfg.ls)
        elif cfg.loss_type == "LA": # https://arxiv.org/abs/2007.07314
            self.criterion = LogitAdjustedLoss(cls_num_list=cls_num_list, label_smoothing=cfg.ls)
        elif cfg.loss_type == "Focal": # https://arxiv.org/abs/1708.02002
            self.criterion = FocalLoss(weight=cls_num_list, gamma=2.0, label_smoothing=cfg.ls)
        elif cfg.loss_type == "MMS": # https://arxiv.org/pdf/2302.00864
            if hasattr(self.head, 'scale'):
                mms_scale = self.head.scale
                self.head.scale = 1.0  # set to 1, since MMS loss has its own scale
            else:
                mms_scale = cfg.scale
            self.criterion = MarginMetricSoftmaxLoss(text_features=self.head.weight.data, label_smoothing=cfg.ls, lam=cfg.mms_lam, scale=mms_scale)
        
    def get_tokenized_prompts(self, classnames, template):
        prompts = [template.format(c.replace("_", " ")) for c in classnames]
        # logging.info(f"Prompts: {prompts}")
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        prompts = prompts.to(self.device)
        return prompts

    @torch.no_grad()
    def init_head_text_feat(self):
        cfg = self.cfg
        classnames = self.classnames

        assert cfg.backbone.startswith("CLIP"), "Text feature initialization is only available for CLIP"

        self.text_encoder.to(self.device)
        text_encoder = self.text_encoder

        logging.info("Initialize head with text features")
        if cfg.prompt == "ensemble":
            if cfg.dataset.lower().startswith("imagenet"):
                logging.info("Using cached text features for ImageNet")
                text_features = torch.load("cache/imagenet_text_features.pt", map_location=self.device)
            elif cfg.dataset.lower().startswith("mini_imagenet"):
                logging.info("Using cached text features for Mini_ImageNet")
                text_features = torch.load("cache/mini_imagenet_text_features.pt", map_location=self.device)
            else:
                all_text_features = []
                for template in tqdm(ZEROSHOT_TEMPLATES['imagenet']):
                    prompts = self.get_tokenized_prompts(classnames, template)
                    text_features = text_encoder.encode_text(prompts)
                    text_features = F.normalize(text_features, dim=-1)
                    all_text_features.append(text_features)
                all_text_features = torch.stack(all_text_features)
                text_features = all_text_features.mean(dim=0)
        elif cfg.prompt == "best":
                all_text_features = []
                for template in tqdm(ZEROSHOT_TEMPLATES['imagenet_best']):
                    prompts = self.get_tokenized_prompts(classnames, template)
                    text_features = text_encoder.encode_text(prompts)
                    text_features = F.normalize(text_features, dim=-1)
                    all_text_features.append(text_features)
                all_text_features = torch.stack(all_text_features)
                text_features = all_text_features.mean(dim=0)
        elif cfg.prompt == "descriptor":
            with open("utils/descriptors_imagenet.json") as f:
                descriptors = json.load(f)
            template = "{}"
            all_class_features = []
            for cn in tqdm(classnames):
                prompts = self.get_tokenized_prompts(descriptors[cn], template)
                text_features = text_encoder.encode_text(prompts)
                text_features = F.normalize(text_features, dim=-1)
                all_class_features.append(text_features.mean(dim=0))
            text_features = torch.stack(all_class_features)
        elif cfg.prompt == "classname":
            template = "{}"
            prompts = self.get_tokenized_prompts(classnames, template)
            text_features = text_encoder.encode_text(prompts)
            text_features = F.normalize(text_features, dim=-1)
        elif cfg.prompt == "default":
            template = "a photo of a {}."
            prompts = self.get_tokenized_prompts(classnames, template)
            text_features = text_encoder.encode_text(prompts)
            text_features = F.normalize(text_features, dim=-1)
        if cfg.backbone.startswith("CLIP-ViT") and not cfg.use_proj:
            text_features = text_features @ self.model.image_encoder.proj.t()
            text_features = F.normalize(text_features, dim=-1)

        self.head.apply_weight(text_features)

        # ensure text encoder is on CPU
        self.text_encoder.to("cpu")

    def resume_training(self):
        cfg = self.cfg

        last_checkpoint = os.path.join(cfg.output_dir, "checkpoint_last.pth.tar")
        if not os.path.exists(last_checkpoint):
            logging.info(f"No checkpoint found at {last_checkpoint}, training from scratch")
        else:
            loaded = self.load_model(last_checkpoint, tuner=True, resume=True, device=self.device)
            if not loaded:
                logging.info(f"No checkpoint with optimizer state found at {last_checkpoint}, training from scratch")
            else:
                logging.info(f"Resuming training from {last_checkpoint}")

            if self.averaged_model is not None:
                last_ema_checkpoint = os.path.join(cfg.output_dir, "checkpoint_ema_last.pth.tar")
                if not os.path.exists(last_ema_checkpoint):
                    logging.info(f"No checkpoint found at {last_ema_checkpoint}, start moving average from scratch")
                    update_ema(self.averaged_model, from_ddp(self.model), decay=0)
                else:
                    logging.info(f"Loading moving average checkpoint from {last_ema_checkpoint}")
                    self.load_model(last_ema_checkpoint, tuner=True, resume=False, model_to_load=self.averaged_model, device=self.device)

    def init_pace(self):
        cfg = self.cfg
        model = from_ddp(self.model)

        block_ids = []
        for m in model.modules():
            if isinstance(m, PACELoRA):
                block_ids.append(m.block_id)

        # sigma value decreases linearly as block_id increases.
        num_blocks = max(block_ids) + 1
        block_start = 1
        sigmas = np.concatenate([np.zeros(block_start), np.linspace(0, cfg.pace_sigma, num_blocks - block_start + 1)[-1:0:-1]])

        for m in model.modules():
            if isinstance(m, PACELoRA):
                m.sigma = sigmas[m.block_id]

    def init_adalora(self):
        cfg = self.cfg
        model = from_ddp(self.model)

        rankallocator = RankAllocator(
            model, lora_r=cfg.adapter_dim, target_rank=int(cfg.adapter_dim / 1.5),
            init_warmup=int(0.10 * self.n_steps), final_warmup=int(0.20 * self.n_steps), mask_interval=10, 
            total_step=self.n_steps, beta1=0.85, beta2=0.85, 
        )

        return rankallocator

    def train(self):
        cfg = self.cfg
        params_for_gradnorm = [p for p in self.model.parameters() if p.requires_grad == True]

        if cfg.tensorboard:
            from torch.utils.tensorboard import SummaryWriter

            # Initialize summary writer
            writer_dir = os.path.join(cfg.output_dir, "tensorboard")
            os.makedirs(writer_dir, exist_ok=True)
            logging.info(f"Initialize tensorboard (log_dir={writer_dir})")
            self._writer = DummyWriter(SummaryWriter(log_dir=writer_dir))
        else:
            logging.info("Tensorboard is not used")
            self._writer = DummyWriter(None)

        # Initialize average meters
        memory_alloc = AverageMeter(ema=True)
        loss_meter = AverageMeter(ema=True)
        acc_meter = AverageMeter(ema=True)
        bal_acc_meter = AverageMeter(ema=True)
        cls_meters = [AverageMeter(ema=True) for _ in range(self.num_classes)]

        # Remember the starting time (for computing the elapsed time)
        time_start = time.time()
        
        self.model.train()
        averaged_model = self.averaged_model

        current_ckp = self.current_ckp
        ema_ckp = self.ema_ckp
        
        if cfg.dy_mask:
            for m in self.model.modules():
                if isinstance(m, DenseGMixout):
                    m.link_optimizer(self.optim)
        
        if cfg.pace:
            self.init_pace()
            history_out = torch.zeros(len(self.train_loader.dataset), self.num_classes, dtype=self.ptdtype).to(self.device)

        if cfg.adalora:
            rankallocator = self.init_adalora()

        val_stats = []

        global_step = self.starting_epoch * self.one_epoch

        ### Setup moving average decay scheduler
        if self.starting_epoch == 0:
            ma_len = self.n_steps - self.warmup_steps
        else:
            ma_len = self.n_steps - global_step
        if cfg.bma:
            beta_dist = sp.stats.beta(cfg.ema, cfg.ema)
            ma_decay = beta_dist.pdf((np.arange(ma_len) + 0.5) / (ma_len + 1))
            ma_decay = ma_decay / np.cumsum(ma_decay)
            ma_decay = 1/(ma_decay + 1)
        else:
            ma_decay = np.ones(ma_len + 1) * cfg.ema


        step_time = time.time()

        for epoch in range(self.starting_epoch, cfg.num_epochs):
            
            if self.ddp: self.train_loader.sampler.set_epoch(epoch)

            epoch_start_time = time.time()
            for step, batch in enumerate(self.train_loader):
                global_step += 1

                image = batch[0]
                label = batch[1]

                image = image.to(self.device, non_blocking=True)
                label = label.to(self.device, non_blocking=True)

                with autocast(device_type='cuda', dtype=self.ptdtype):
                    output, _ = self.model(image, return_feature=False)
                    loss = self.criterion(output, label)

                    if cfg.pace:
                        index = batch[2]
                        with torch.no_grad():
                            logits_recent = history_out[index]
                            history_out[index] = output
                        
                        if epoch > self.starting_epoch:
                            loss_pace = F.mse_loss(output, logits_recent)
                            loss = loss + cfg.pace_lam * loss_pace

                    if cfg.adalora:
                        loss = loss + compute_orth_regu(self.model, regu_weight=cfg.adalora_regu)

                self.scaler.scale(loss).backward()

                self.scaler.unscale_(self.optim)
                torch.nn.utils.clip_grad_norm_(params_for_gradnorm, 1.0)

                self.scaler.step(self.optim)
                self.scaler.update()

                if cfg.adalora:
                    rankallocator.update_and_mask(self.model, global_step)

                self.optim.zero_grad(set_to_none=True)

                with torch.no_grad():
                    pred = output.argmax(dim=1)
                    correct = pred.eq(label).float()
                    acc = correct.mean().mul_(100.0)

                current_lr = self.optim.param_groups[0]["lr"]
                loss_meter.update(loss.item())

                acc_meter.update(acc.item())

                for _c, _y in zip(correct, label):
                    cls_meters[_y].update(_c.mul_(100.0).item(), n=1)
                cls_accs = [cls_meters[i].avg for i in range(self.num_classes)]

                bal_acc_meter.update(np.mean(np.array(cls_accs)))

                _, cached_memory = memory_usage()
                memory_alloc.update(cached_memory)

                partial_epoch = ((step + 1) / self.one_epoch) + epoch

                if averaged_model is not None and global_step > self.warmup_steps:
                    update_ema(self.averaged_model, from_ddp(self.model), decay=cfg.ema)

                if global_step == self.warmup_steps:
                    logging.train(f"Warmup steps {self.warmup_steps} finished")

                meet_freq = (global_step + 1) % cfg.print_freq == 0
                if meet_freq:
                    # Measure training speed:
                    torch.cuda.synchronize()
                    step_per_sec = (time.time() - step_time) / cfg.print_freq

                    if self.ddp:
                        loss_meter.all_reduce()
                        acc_meter.all_reduce()

                    info = []
                    info += [f"epoch {partial_epoch:.1f}/{cfg.num_epochs}"]
                    info += [f"step [{global_step + 1}/{self.n_steps}]"]
                    info += [f"time(step) {step_per_sec:.2f}"]
                    info += [f"loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})"]
                    info += [f"acc {acc_meter.val:.4f} ({acc_meter.avg:.4f})"]
                    info += [f"bal_acc {bal_acc_meter.val:.4f} ({bal_acc_meter.avg:.4f})"]
                    info += [f"lr {current_lr:.4e}"]
                    info += [f"memory {memory_alloc.val:.2f} ({memory_alloc.avg:.2f}) GB"]
                    logging.train(" ".join(info))

                    step_time = time.time()

                    if self.ddp:
                        loss_meter.reset()
                        acc_meter.reset()
                
                self._writer.add_scalar("train/lr", current_lr, global_step)
                self._writer.add_scalar("train/loss", loss_meter.avg, global_step)
                self._writer.add_scalar("train/acc", acc_meter.avg, global_step)
                self._writer.add_scalar("train/bal_acc", bal_acc_meter.avg, global_step)
                
                self.sched.step()
                torch.cuda.empty_cache()

            if cfg.eval_on_val:
                eval_results = self.eval("val")
                eval_info = [f"validation @ epoch {epoch}:"]
                eval_info += [f"loss {eval_results['loss']:.4f} acc {eval_results['acc']:.4f} bal_acc {eval_results['bal_acc']:.4f} macro_f1 {eval_results['macro_f1']:.4f} eval_time {eval_results['eval_time']:.4f}"]
                logging.eval(" ".join(eval_info))

                for k, v in eval_results.items():
                    self._writer.add_scalar(f"val/{k}", v, epoch)

                val_stats.append(eval_info)

                updated = current_ckp.update(eval_results, epoch)
                if cfg.resume:
                    self.save_model(cfg.output_dir, current_epoch=epoch, with_optim=True, last=True)

                if cfg.save_model and updated:
                    self.save_model(cfg.output_dir, current_epoch=epoch)

                if averaged_model is not None:
                    eval_results = self.eval("val", ema_model=averaged_model)
                    eval_info = [f"EMA validation @ epoch {epoch}:"]
                    eval_info += [f"loss {eval_results['loss']:.4f} acc {eval_results['acc']:.4f} bal_acc {eval_results['bal_acc']:.4f} macro_f1 {eval_results['macro_f1']:.4f} eval_time {eval_results['eval_time']:.4f}"]
                    logging.eval(" ".join(eval_info))

                    val_stats.append(eval_info)

                    for k, v in eval_results.items():
                        self._writer.add_scalar(f"val_ema/{k}", v, epoch)

                    updated = ema_ckp.update(eval_results, epoch)
                    if cfg.resume:
                        self.save_model(cfg.output_dir, current_epoch=epoch, last=True, ema_model=averaged_model)

                    if cfg.save_model and updated:
                        self.save_model(cfg.output_dir, current_epoch=epoch, ema_model=averaged_model)


                if cfg.early_stop > 0:
                    if current_ckp.patience > 0:
                        logging.info(f"patience {current_ckp.patience}/{cfg.early_stop}")

                    if current_ckp.patience >= cfg.early_stop:
                        logging.info(f"Early stopping at iteration {step + 1}")
                        break
            
            torch.cuda.synchronize()
            epoch_per_sec = round((time.time() - epoch_start_time))
            logging.info(f"Epoch {epoch + 1}/{cfg.num_epochs} finished in {str(datetime.timedelta(seconds=epoch_per_sec))}")            


        logging.info("Finish training")
        # show elapsed time
        torch.cuda.synchronize()
        elapsed = round(time.time() - time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        logging.info(f"Time elapsed: {elapsed}")

        for val_stat in val_stats:
            logging.eval(f"stats: {val_stat}")
        
        if cfg.eval_on_val:
            if dist.get_rank() == 0:
                best_epoch, best_metric = current_ckp.stats()
                if best_epoch is not None:
                    logging.info(f"Best {current_ckp.key} at epoch {best_epoch}: loss {best_metric['loss']:.4f} acc {best_metric['acc']:.4f} bal_acc {best_metric['bal_acc']:.4f} macro_f1 {best_metric['macro_f1']:.4f}")
                    ckp = os.path.join(cfg.output_dir, "checkpoint.pth.tar")
                    self.load_model(ckp, tuner=True, device=self.device)
                    logging.info("Best model loaded")
                else:
                    logging.info("No best model found, using the current model")
                self.test("test")

                if averaged_model is not None:
                    best_epoch, best_metric = ema_ckp.stats()
                    if best_epoch is not None:
                        logging.info(f"Best EMA {current_ckp.key} at epoch {best_epoch}: loss {best_metric['loss']:.4f} acc {best_metric['acc']:.4f} bal_acc {best_metric['bal_acc']:.4f} macro_f1 {best_metric['macro_f1']:.4f}")
                        ckp = os.path.join(cfg.output_dir, "checkpoint_ema.pth.tar")
                        self.load_model(ckp, tuner=True, device=self.device)
                        logging.info("Best EMA model loaded")
                    
                    else:
                        logging.info("No best EMA model found, using the current model")
                    self.test("test", ema_model=self.model)
            
            if self.ddp:
                dist.barrier()
        else:
            if cfg.save_model:
                self.save_model(cfg.output_dir, current_epoch=cfg.num_epochs - 1)

            if dist.get_rank() == 0: self.test("test")
            if self.ddp: dist.barrier()


        # signal that the training is done
        if dist.get_rank() == 0:
            with open(os.path.join(cfg.output_dir, "DONE"), "w") as f:
                pass

            if os.path.exists(os.path.join(cfg.output_dir, "checkpoint_last.pth.tar")):
                logging.info("Removing last checkpoint")
                os.remove(os.path.join(cfg.output_dir, "checkpoint_last.pth.tar"))

        # Close writer
        self._writer.close()

    @torch.no_grad()
    def test(self, on="test", ema_model=None):
        if ema_model is None:
            model = self.model
            post_fix = ""
        else:
            model = ema_model
            post_fix = "ema"

        model.eval()

        self.test_evaluator.reset()

        if on == "val":
            logging.info(f"Evaluate on the validation set")
            data_loader = self.val_loader
        elif on == "test":
            logging.info(f"Evaluate on the test set")
            data_loader = self.test_loader
        
        for batch in tqdm(data_loader):
            image = batch[0]
            label = batch[1]

            label = self.project_labels_fn(label) if self.project_labels_fn is not None else label

            image = image.to(self.device, non_blocking=True)
            label = label.to(self.device, non_blocking=True)

            with autocast(device_type='cuda', dtype=self.ptdtype):
                output = model(image)[0]

                if self.project_logits_fn is not None:
                    output = self.project_logits_fn(output)

            self.test_evaluator.process(output, label)

        results = self.test_evaluator.evaluate()
        self.model.train()

        for k, v in results.items():
            tag = f"test/{k}"
            if self._writer is not None:
                self._writer.add_scalar(tag, v)

        if self.cfg.save_results:
            self.save_results_json(results, post_fix=post_fix)

        return results

    @torch.no_grad()
    def eval(self, on="val", ema_model=None):

        if self.ddp:
            torch.cuda.synchronize()
            dist.barrier()

        if ema_model is None:
            model = self.model
        else:
            model = ema_model

        model.eval()

        self.val_evaluator.reset()

        if on == "val":
            data_loader = self.val_loader
        elif on == "test":
            data_loader = self.test_loader

        loader = tqdm(data_loader) if self.cfg.test_only else data_loader

        start_time = time.time()

        for batch in loader:
            image = batch[0]
            label = batch[1]

            label = self.project_labels_fn(label) if self.project_labels_fn is not None else label

            image = image.to(self.device, non_blocking=True)
            label = label.to(self.device, non_blocking=True)

            with autocast(device_type='cuda', dtype=self.ptdtype):
                output = model(image)[0]
            
                if self.project_logits_fn is not None:
                    output = self.project_logits_fn(output)

            self.val_evaluator.process(output, label)

        if self.ddp:
            self.val_evaluator.all_gather()

        results = self.val_evaluator.evaluate()

        self.model.train()

        eval_time = time.time() - start_time
        results["eval_time"] = eval_time

        return results

    def eval_corruption(self):
        cfg = self.cfg

        corr_res = {}
        for corruption in cfg.corruptions:
            self.setup_corruption_loader(corruption)
            logging.info(f"Evaluating on corruption {corruption} ({cfg.corruption_level})")
            res = self.eval("test")
            logging.info(f"{corruption} results: {float(res['acc']):.3f}%")
            corr_res[corruption] = res
        
        corr_acc = {k: float(np.round(v["acc"], 3)) for k,v in corr_res.items()}
        avg_acc = sum(corr_acc.values()) / len(corr_acc)
        logging.eval(f"Average corruption accuracy: {avg_acc:.2f}%")
        logging.eval(f"Corruption results: {corr_acc}")

        sum_dict = {k: 0 for k in corr_res[corruption].keys()}
        for _, v in corr_res.items():
            for inner_k, inner_v in v.items():
                sum_dict[inner_k] += inner_v
        sum_dict = {k: v / len(cfg.corruptions) for k, v in sum_dict.items()}
        corr_res["average"] = sum_dict

        if cfg.save_results:
            self.save_results_json(corr_res, "corruption_results")
        
        return corr_res
    
    def eval_outdomain(self):
        cfg = self.cfg

        outdomain_res = {}
        for od_ds in cfg.outdomain_datasets:
            logging.info(f"Evaluating on out-domain dataset: {od_ds}")
            self.setup_outdomain_loader(od_ds)
            res = self.eval("test")
            logging.info(f"{od_ds} results: {float(res['acc']):.3f}%")
            outdomain_res[od_ds] = res
        
        od_acc = {k: float(np.round(v["acc"], 3)) for k,v in outdomain_res.items()}
        avg_acc = sum(od_acc.values()) / len(od_acc)
        logging.eval(f"Average out-domain accuracy: {avg_acc:.2f}%")
        logging.eval(f"Out-domain results: {od_acc}")

        sum_dict = {k: 0 for k in outdomain_res[od_ds].keys()}
        for _, v in outdomain_res.items():
            for inner_k, inner_v in v.items():
                sum_dict[inner_k] += inner_v
        sum_dict = {k: v / len(cfg.outdomain_datasets) for k, v in sum_dict.items()}
        outdomain_res["average"] = sum_dict

        if cfg.save_results:
            self.save_results_json(outdomain_res, "outdomain_results")
        
        return outdomain_res

    def eval_fewshot(self):
        cfg = self.cfg

        assert cfg.backbone.startswith("CLIP") == True, "Backbone must be CLIP"

        fewshot_res = {}
        for fw_ds in cfg.fewshot_datasets:
            self.setup_fewshot_loader(fw_ds)
            res = self.eval("test")
            logging.info(f"{fw_ds} results: {float(res['acc']):.3f}%")
            fewshot_res[fw_ds] = res
        
        fw_acc = {k: float(np.round(v["acc"], 3)) for k,v in fewshot_res.items()}
        avg_acc = sum(fw_acc.values()) / len(fw_acc)
        logging.eval(f"Average few-shot accuracy: {avg_acc:.2f}%")
        logging.eval(f"Few-shot results: {fw_acc}")

        sum_dict = {k: 0 for k in fewshot_res[fw_ds].keys()}
        for _, v in fewshot_res.items():
            for inner_k, inner_v in v.items():
                sum_dict[inner_k] += inner_v
        sum_dict = {k: v / len(cfg.fewshot_datasets) for k, v in sum_dict.items()}
        fewshot_res["average"] = sum_dict

        if cfg.save_results:
            self.save_results_json(fewshot_res, "few_shot_results")
        
        return fewshot_res

    def merge(self, alpha, theta_0, theta_1):
        w = {
            key: (1 - alpha) * theta_0[key] + alpha * theta_1[key]
            for key in theta_0.keys()
        }

        return copy.deepcopy(w)

    def search_alpha(self, zs_model, ft_model):
        zs_image_encoder_sd = zs_model.image_encoder.state_dict()
        ft_image_encoder_sd = ft_model.image_encoder.state_dict()
        zs_tuner_sd = zs_model.tuner.state_dict()
        ft_tuner_sd = ft_model.tuner.state_dict()
        zs_head_sd = zs_model.head.state_dict()
        ft_head_sd = ft_model.head.state_dict()

        if self.cfg.wise_alpha is None:
            best_alpha = 0.0
            best_res = {"bal_acc": 0.0}

            logging.info("Searching for the best alpha for Wise")

            for alpha in tqdm(np.arange(0, 1.1, 0.1)):
                image_encoder_theta = self.merge(alpha, zs_image_encoder_sd, ft_image_encoder_sd)
                self.model.image_encoder.load_state_dict(image_encoder_theta, strict=True)

                tuner_theta = self.merge(alpha, zs_tuner_sd, ft_tuner_sd)
                self.tuner.load_state_dict(tuner_theta, strict=True)

                head_theta = self.merge(alpha, zs_head_sd, ft_head_sd)
                self.head.load_state_dict(head_theta, strict=True)

                res = self.eval("val")
                if res["bal_acc"] > best_res["bal_acc"]:
                    best_res = res
                    best_alpha = alpha

            logging.eval(f"Wise alpha {best_alpha:.1f} - loss: {best_res['loss']:.4f}, acc: {best_res['acc']:.4f}, bal_acc: {best_res['bal_acc']:.4f}")
        
        else:
            best_alpha = self.cfg.wise_alpha
            logging.info(f"Using fixed alpha for Wise: {best_alpha}")


        image_encoder_theta = self.merge(best_alpha, zs_image_encoder_sd, ft_image_encoder_sd)
        self.model.image_encoder.load_state_dict(image_encoder_theta, strict=True)

        tuner_theta = self.merge(best_alpha, zs_tuner_sd, ft_tuner_sd)
        self.tuner.load_state_dict(tuner_theta, strict=True)

        head_theta = self.merge(best_alpha, zs_head_sd, ft_head_sd)
        self.head.load_state_dict(head_theta, strict=True)

    def save_results_json(self, results, post_fix=""):
        if post_fix != "":
            post_fix = "_" + post_fix

        log_num = int(logging.getLogger().handlers[-1].baseFilename.split("/")[-1].lstrip("log.").rstrip(".log"))
        with open(os.path.join(self.cfg.output_dir, f"results{post_fix}.{log_num}.json"), "w") as f:
            json.dump(results, f, indent=4)

    def save_model(self, directory, current_epoch, with_optim=False, ema_model=None, last=False, optional_postfix=[]):
        cfg = self.cfg

        if dist.get_rank() == 0:
            
            post_fixes = []

            if ema_model is None:
                model = self.model
            else:
                model = ema_model
                post_fixes.append("ema")
            
            if last:
                post_fixes.append("last")

            if len(optional_postfix) > 0:
                post_fixes.extend(optional_postfix)

            cp_model = copy.deepcopy(from_ddp(model))
            
            cp_model.half()
            cp_model.eval()

            if cfg.dy_mask and not cfg.mixout:
                tuner_dict = cp_model.image_encoder.state_dict()
            else:
                tuner_dict = cp_model.tuner.state_dict()

            head_dict = cp_model.head.state_dict()

            checkpoint = {
                "tuner": tuner_dict,
                "head": head_dict,
            }
            logging.info("Adding tuner and head weights to checkpoint")

            keys = list(checkpoint.keys())
            # remove 'module.' in state_dict's keys
            for key in keys:
                state_dict = checkpoint[key]
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    if k.startswith("module."):
                        k = k[7:]
                    new_state_dict[k] = v
                checkpoint[key] = new_state_dict

            if with_optim:
                checkpoint["config"] = cfg
                checkpoint["epoch"] = current_epoch
                checkpoint["optimizer"] = self.optim.state_dict()
                checkpoint["scheduler"] = self.sched.state_dict()
                checkpoint["scaler"] = self.scaler.state_dict()
                checkpoint["model_ckp"] = self.current_ckp.state_dict()

                logging.info("Adding optimizer, scheduler and scaler states to checkpoint")

            # save model
            if len(post_fixes) > 0:
                post_fixes = "_" + "_".join(post_fixes)
            else:
                post_fixes = ""

            save_path = os.path.join(directory, f"checkpoint{post_fixes}.pth.tar")
            torch.save(checkpoint, save_path)
            logging.info(f"Checkpoint saved to {save_path}")

        if self.ddp: dist.barrier()

    def load_model(self, load_path, tuner=True, skip_head=False, resume=False, model_to_load=None, device=None):
        cfg = self.cfg
        loaded = False

        if model_to_load is None:
            model_to_load = self.model
        
        model_to_load = from_ddp(model_to_load)

        if not os.path.exists(load_path):
            logging.error('Checkpoint not found at "{}"'.format(load_path))
            raise FileNotFoundError('Checkpoint not found at "{}"'.format(load_path))

        checkpoint = torch.load(load_path, map_location=self.device if device==None else device, weights_only=False)

        if resume and 'optimizer' not in checkpoint:
            return False
    
        if tuner:
            tuner_dict = checkpoint["tuner"]
            logging.info("Loading tuner weights to from {}".format(load_path))

            ## this is a special case in which the weights are from the full-ft model and the current model is PEFT
            key0 = list(tuner_dict.keys())[0] if len(tuner_dict) > 0 else ""
            if key0.startswith("block_tuned") and (not cfg.full_tuning):
                logging.info("Loading tuner weights from full-ft to PEFT")
                new_tuner_dict = {}
                for k, v in tuner_dict.items():
                    # replace block_tuned with blocks
                    new_key = k.replace("block_tuned", "blocks")
                    new_tuner_dict[new_key] = v        
                model_to_load.image_encoder.load_state_dict(new_tuner_dict, strict=False)
            elif cfg.mask and cfg.dy_mask and not cfg.mixout:
                model_to_load.image_encoder.load_state_dict(tuner_dict, strict=False)
            else:
                model_to_load.tuner.load_state_dict(tuner_dict, strict=False)
                
            loaded = True

        if not skip_head:
            head_dict = checkpoint["head"]
            logging.info("Loading head weights to from {}".format(load_path))
            if head_dict["weight"].shape == model_to_load.head.weight.shape:
                safe_load(model_to_load.head, head_dict)
            
            loaded = True
        
        model_to_load.float()

        if resume:
            if 'optimizer' not in checkpoint:
                logging.error("Checkpoint does not contain 'optimizer' key for resuming training")
                raise ValueError("Checkpoint does not contain 'optimizer' key for resuming training")
            
            if 'epoch' not in checkpoint:
                logging.warning("Checkpoint does not contain 'epoch' key, using step to determine the starting epoch")
                starting_step = checkpoint["step"] + 1
                self.starting_epoch = starting_step // self.one_epoch
            else:
                self.starting_epoch = checkpoint["epoch"] + 1
            
            self.optim.load_state_dict(checkpoint["optimizer"])
            self.sched.load_state_dict(checkpoint["scheduler"])
            self.scaler.load_state_dict(checkpoint["scaler"])
            if "model_ckp" not in checkpoint:
                logging.warning("Checkpoint does not contain 'model_ckp' key, this might cause overtraining")
            else:
                self.current_ckp.load_state_dict(checkpoint["model_ckp"])

            logging.info("Optimizer, scheduler and scaler states loaded from {}".format(load_path))
            logging.info("Resuming training from epoch {}".format(self.starting_epoch))

        assert loaded == True, "No weights loaded"

        return loaded
