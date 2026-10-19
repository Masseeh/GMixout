import argparse
import os
import torch

def load_ckpt(load_path):
    if not os.path.exists(load_path):
        raise FileNotFoundError('Checkpoint not found at "{}"'.format(load_path))

    checkpoint = torch.load(load_path, map_location='cpu', weights_only=False)

    if "tuner" not in checkpoint or "head" not in checkpoint:
        raise KeyError('Checkpoint must contain "tuner" and "head" keys')

    return checkpoint["tuner"], checkpoint["head"]


parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=str, required=True, help="checkpoint path")
args = parser.parse_args()

tuner, head = load_ckpt(args.ckpt)

checkpoint = {
    "tuner": tuner,
    "head": head
}

torch.save(checkpoint, args.ckpt)
print(f"Optimizer striped from checkpoint")