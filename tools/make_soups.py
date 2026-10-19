import argparse
import os
import torch
from copy import deepcopy

def merge(theta_0, theta_1):
    w = {
        key: theta_0[key] + theta_1[key]
        for key in theta_0.keys()
    }
    return deepcopy(w)

def load_ckpt(load_path):
    if not os.path.exists(load_path):
        raise FileNotFoundError('Checkpoint not found at "{}"'.format(load_path))

    checkpoint = torch.load(load_path, map_location='cpu', weights_only=False)

    if "tuner" not in checkpoint or "head" not in checkpoint:
        raise KeyError('Checkpoint must contain "tuner" and "head" keys')

    return checkpoint["tuner"], checkpoint["head"]


parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", nargs='+', type=str, default=[], help="checkpoint paths to merge")
parser.add_argument("--output", type=str, required=True, help="output path for the merged checkpoint")
parser.add_argument("--outdir", type=str, default="./output", help="output directory")
args = parser.parse_args()

cpkt_paths = args.ckpt

if not isinstance(cpkt_paths, list):
    cpkt_paths = [cpkt_paths]

soups_tuner, soups_head = load_ckpt(cpkt_paths[0])
print(f"Successfully loaded checkpoint from {cpkt_paths[0]}")

for cpkt_path in cpkt_paths[1:]:
    try:
        checkpoint_tuner, checkpoint_head = load_ckpt(cpkt_path)
        soups_tuner = merge(soups_tuner, checkpoint_tuner)
        soups_head = merge(soups_head, checkpoint_head)
        print(f"Successfully loaded checkpoint from {cpkt_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint from {cpkt_path}: {e}")

soups_tuner = {k: (v.float()/len(cpkt_paths)).half() for k, v in soups_tuner.items()}
soups_head = {k: (v.float()/len(cpkt_paths)).half() for k, v in soups_head.items()}

checkpoint = {
    "tuner": soups_tuner,
    "head": soups_head
}

args.output = os.path.join(args.outdir, args.output)
os.makedirs(args.output, exist_ok=True)
output_path = os.path.join(args.output, "checkpoint.pth.tar")
torch.save(checkpoint, output_path)
print(f"Merged checkpoint saved to {output_path}")
