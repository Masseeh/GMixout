import os
os.environ["TORCH_HUB"] = "cache/"
os.environ['HF_HOME'] = "cache/"
import torch
from torch.amp import autocast
from torch.profiler import profile, ProfilerActivity
from utils.config import _C as cfg
import argparse
from models import *
from trainer import load_clip_to_cpu

parser = argparse.ArgumentParser()
parser.add_argument("--model", "-m", type=str, default="", help="model config file")
parser.add_argument("--eval_conf", "-e", type=str, default="", help="eval config file")
parser.add_argument("--verbose", "-v", action='store_true', help="verbose profiling info")
parser.add_argument("opts", default=None, nargs=argparse.REMAINDER,
                    help="modify config options using the command-line")
args = parser.parse_args()

cfg_model_file = os.path.join("./configs/model", args.model + ".yaml")
cfg_eval_file = os.path.join("./configs/eval", args.eval_conf + ".yaml") if args.eval_conf else None

cfg.defrost()
cfg.merge_from_file(cfg_model_file)

if cfg_eval_file is not None:
    cfg.merge_from_file(cfg_eval_file)

cfg.merge_from_list(args.opts)


dtype  = torch.bfloat16
device = "cuda:0"   

clip_model = load_clip_to_cpu(cfg.backbone, "bf16", pretrained=False)
model = PeftModelFromCLIP(cfg, clip_model, 1000)
model = model.to(device)
tuner = model.tuner
head = model.head


for param in model.parameters():
    param.requires_grad_(False)

for param_name, param in tuner.named_parameters():
    param.requires_grad_(True)

params_to_optimize = [{"params": [p for p in tuner.parameters() if p.requires_grad == True]}]

if len(params_to_optimize[0]["params"]) > 0:
    print(f"Turning on gradients in the encoder")

if cfg.head_tuning:
    print(f"Turning on gradients in the head")
    for param in head.optim_params():
        param.requires_grad_(True)

    params_to_optimize.extend([{"params": [p for p in head.parameters() if p.requires_grad == True]}])


if cfg.optim == "adam":
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.lr, fused=True, weight_decay=cfg.wd)
else:
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.wd)

encoder_params = sum(p.numel() for p in model.image_encoder.parameters())
tunable_params = sum(p.numel() for p in tuner.parameters() if p.requires_grad)
head_params = sum(p.numel() for p in head.parameters())
trainable_params = tunable_params + head_params
print(f"Total parameters: {(encoder_params+head_params)/1e6:.2f}M")
print(f"Total tunable parameters: {tunable_params/1e6:.2f}M")
print(f"Total head parameters: {head_params/1e6:.2f}M")
print(f"Total trainable parameters: {trainable_params/1e6:.2f}M")
print(f"Trainable ratio: {trainable_params/(encoder_params+head_params):.4f}")


input_size = (cfg.batch_size, 3, cfg.resolution, cfg.resolution)
iters = 5
for _ in range(iters):
    x = torch.randn(input_size, device=device, dtype=dtype)

    opt.zero_grad()                           # reset gradients for next iteration

    with torch.autocast(device_type='cuda', dtype=dtype, enabled=False if dtype == torch.float32 else True):
        loss = model(x)[0].mean()
    
    loss.backward()                          # backward pass
    opt.step()                               # optimizer step

# ---------- PROFILING ----------
with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,        # needed so PyTorch can infer FLOPs
        with_flops=True,           # turn on FLOP counting
        profile_memory=True       # cheaper; flip to True if you also want memory
) as prof:

    x = torch.randn(input_size, device=device, dtype=dtype)

    opt.zero_grad()                           # reset gradients for next iteration

    with autocast(device_type='cuda', dtype=dtype, enabled=False if dtype == torch.float32 else True):
        loss = model(x)[0].mean()
    
    loss.backward()                          # backward pass
    opt.step()                               # optimizer step

    torch.cuda.synchronize()       # make sure all kernels finish before leaving scope

if args.verbose: print(prof.key_averages().table(sort_by="flops", row_limit=20))
avg_stats = prof.key_averages().total_average()
total_flops = avg_stats.flops
print(f"Total per-step: {total_flops/1e12:.3f} TFLOPs")

step_ms = []
mem_c = []
for _ in range(20):
    torch.cuda.empty_cache()  # clear cache to avoid memory fragmentation
    torch.cuda.synchronize()   # make sure all kernels finish before starting the timer
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)

    start.record()
    # ---- repeat the **same** training step once more ----
    x = torch.randn(input_size, device=device, dtype=dtype)

    opt.zero_grad()                           # reset gradients for next iteration
    with torch.autocast(device_type='cuda', dtype=dtype, enabled=False if dtype == torch.float32 else True):
        loss = model(x)[0].mean()

    loss.backward()                          # backward pass
    opt.step()                               # optimizer step

    mem_c.append(torch.cuda.memory_reserved() / 1e9)  # in GB

    opt.zero_grad()                           # reset gradients for next iteration

    end.record()

    torch.cuda.synchronize()
    step_ms.append(start.elapsed_time(end))  # milliseconds

avg_mem_c = sum(mem_c) / len(mem_c)
print(f"Average memory reserved: {avg_mem_c:.3f} GB")
avg_step_ms = sum(step_ms) / len(step_ms)
print(f"time/step: {avg_step_ms:.3f} ms")

peak_flops = 34e12 if dtype == torch.float32 else 71.1e12
achieved   = total_flops / (avg_step_ms / 1e3)
mfu        = achieved / peak_flops
print(f"MFU: {mfu:.2%}")