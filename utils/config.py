from yacs.config import CfgNode as CN

_C = CN()

_C.dataset = ""  # Dataset name
_C.root = ""  # dataset directory name
_C.imb_factor = None  # for long-tailed cifar dataset
_C.tensorboard = False  # Use tensorboard for logging
_C.use_ffcv = False  # Use ffcv for fast data loading
_C.o3 = False  # Use optimzed code
_C.model_selection = "bal_acc"

_C.backbone = ""
_C.resolution = 224
_C.n_max = -1

_C.output_dir = None  # Directory to save the output files (like log.txt and model weights)
_C.print_freq = 10  # How often (batch) to print training information

_C.seed = None  # use manual seed
_C.deterministic = False  # output deterministic results
_C.gpu = None      # assign a single gpu 
_C.num_workers = 8
_C.prec = "fp16"  # fp32, amp

_C.num_epochs = 10
_C.batch_size = 128
_C.test_batch_size = 256
_C.lr = 1e-4
_C.wd = 0.1
_C.momentum = 0.9
_C.loss_type = "CE"  # "CE" / "Focal" / "LA" / "MMS"
_C.ls = 0.0  # label smoothing
_C.resume = False

_C.classifier = "CosineClassifier"
_C.scale = 30  # for cosine classifier
_C.use_proj = True         # use projection layer for image encoder

_C.full_tuning = False  # full fine-tuning
_C.bias_tuning = False  # only fine-tuning the bias 
_C.ln_tuning = False  # only fine-tuning the layer norm
_C.bn_tuning = False  # only fine-tuning the batch norm (only for resnet)
_C.vpt_shallow = False
_C.vpt_deep = False
_C.adapter = False
_C.adaptformer = False
_C.lora = False
_C.lora_mlp = False
_C.dora = False
_C.soma = False
_C.pace = False
_C.pace_sigma = 1.2
_C.pace_lam = 1.0
_C.adalora = False
_C.adalora_regu = 0.1
_C.ssf_attn = False
_C.ssf_mlp = False
_C.ssf_ln = False
_C.partial = None  # fine-tuning (or parameter-efficient fine-tuning) partial block layers
_C.vpt_len = None  # length of VPT sequence
_C.adapter_dim = None  # bottle dimension for adapter / adaptformer / lora.
_C.adaptformer_scale = "learnable"  # "learnable" or scalar


_C.mask = False   # fine-tuning a specific proportion of all parameters
_C.mixout = False  # mixout fine-tuning
_C.mask_ratio = None
_C.mask_seed = None
_C.mask_refresh = 50
_C.mask_ema = 0.0
_C.mask_momentum = True  # keep momentum for masked parameters
_C.dy_mask = False
_C.det_mask = False  # deterministic mask
_C.save_mem = False

_C.init_head = None  # "text_feat" (only for CLIP) / "class_mean" / "lp"
_C.lp_head = None  # "lp" for linear probing head
_C.prompt = "default" # "classname" / "default" / "ensemble" / "descriptor"
_C.randaug = False
_C.wise = False  # use adaptive weight for fine-tuning
_C.wise_alpha = None  # initial alpha for Wise

_C.test_only = False  # load model and test
_C.model_dir = None

########################################################

_C.corruptions = None  # corruption type for CIFAR-10-C / CIFAR-100-C
_C.corruption_level = 5  # severity level for CIFAR-10-C / CIFAR-100-C

_C.fewshot_datasets = None  # few-shot dataset name
_C.outdomain_datasets = None  # out-of-domain datasets for evaluation
_C.outdomain_split = "test"  # split for out-of-domain datasets

########################################################

_C.head_tuning = True       # Head fine-tuning
_C.re_head = False          # Reinitialize head
_C.class_balanced = False
_C.eval_on_val = True       # evaluate on validation set
_C.save_model = True
_C.save_results = True
_C.scheduler = 'cosine'
_C.early_stop = 10
_C.warmup_steps = 1.0 # 1 epoch

_C.lora_alpha = 1
_C.lrh_factor = 1.0
_C.ema = 0.0
_C.bma = False  # Beta moving avg
_C.optim = "adam" # "adam" / "sgd" / "adamSPD"
_C.mms_lam = 0.3  # lambda for Margin Metric Softmax Loss