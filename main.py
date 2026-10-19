import os
os.environ["TORCH_HUB"] = "cache/"
os.environ['HF_HOME'] = "cache/"

import random
import argparse
import numpy as np
import torch
from pathlib import Path
from copy import deepcopy

from utils.config import _C as cfg
from utils.logger import setup_logging, dummy_logger
import torch.distributed as dist
from utils.distributed import init_distributed_mode

import logging

from trainer import Trainer


def main(args):
    cfg_data_file = os.path.join("./configs/data", args.data + ".yaml")
    cfg_model_file = os.path.join("./configs/model", args.model + ".yaml")
    cfg_eval_file = os.path.join("./configs/eval", args.eval_conf + ".yaml") if args.eval_conf else None

    cfg.defrost()
    cfg.merge_from_file(cfg_model_file)
    cfg.merge_from_file(cfg_data_file)

    if cfg_eval_file is not None:
        cfg.merge_from_file(cfg_eval_file)

    cfg.merge_from_list(args.opts)

    # Setup DDP:
    ddp_args = argparse.Namespace()
    init_distributed_mode(ddp_args)
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    torch.cuda.set_device(device)

    if cfg.output_dir is None:
        cfg_name = "_".join([args.data, args.model])
        opts_name = "".join(["_" + item for item in args.opts])
        cfg.output_dir = os.path.join(args.outdir, cfg_name + opts_name)
    else:
        cfg.output_dir = os.path.join(args.outdir, cfg.output_dir)

    cfg.root = os.path.join(args.data_dir, cfg.root)

    if rank == 0:
        os.makedirs(cfg.output_dir, exist_ok=True)

        # Set up logging
        if (not setup_logging(console_log_output="stdout", console_log_level="info", console_log_color=True,
                            logfile_file=os.path.join(cfg.output_dir, "log.0.log"), logfile_log_level="info", logfile_log_color=False,
                            log_line_template="%(color_on)s[%(levelname)-2s] %(message)s%(color_off)s")):
            print("Failed to set up logging, aborting.")
            return 1
    else:
        dummy_logger()

    logging.info("Output directory: {}".format(cfg.output_dir))    
    logging.info("** Config **")
    logging.info(cfg)

    if cfg.seed is not None:
        seed = cfg.seed * dist.get_world_size() + rank
        logging.info("Setting fixed seed: {}".format(seed))
        random.seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    if cfg.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    logging.info(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    trainer = Trainer(cfg, ddp_args)

    if cfg.wise:
        zs_model = deepcopy(trainer.model)
        zs_model.eval()
        assert cfg.model_dir is not None, "Model directory must be specified for Wise."
    
    if cfg.lp_head is not None:
        trainer.load_model(cfg.lp_head, tuner=False, skip_head=False)

    if cfg.model_dir is not None:
        trainer.load_model(cfg.model_dir, tuner=True, skip_head=cfg.re_head)

    if cfg.wise:
        ft_model = deepcopy(trainer.model)
        ft_model.eval()

        trainer.search_alpha(zs_model, ft_model)

    if cfg.test_only == True:
        if cfg.corruptions is not None:
            trainer.eval_corruption()
        elif cfg.fewshot_datasets is not None:
            trainer.eval_fewshot()
        elif cfg.outdomain_datasets is not None:
            trainer.eval_outdomain()
        else:
            trainer.test("test")

        if dist.is_initialized(): dist.destroy_process_group()

        return
    
    try:
        trainer.train()
    finally:
        if dist.is_initialized(): dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/export/livia/home/vision/masih/local/masih/dataset", help="data directory")
    parser.add_argument("--outdir", type=str, default="./output", help="output directory")
    parser.add_argument("--data", "-d", type=str, default="", help="data config file")
    parser.add_argument("--model", "-m", type=str, default="", help="model config file")
    parser.add_argument("--eval_conf", "-e", type=str, default="", help="eval config file")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER,
                        help="modify config options using the command-line")
    args = parser.parse_args()
    main(args)
