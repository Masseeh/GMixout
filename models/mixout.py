import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import torch._dynamo
torch._dynamo.config.force_parameter_static_shapes = False

# forward (unchanged except we return the mask seed)
@triton.jit
def _seeded_mixout_fw(x_ptr,          # float*
                      out_ptr,        # float*
                      tgt_ptr,        # float*
                      n_elem, p, scale, seed,
                      BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elem
    x  = tl.load(x_ptr  + offs, mask=mask)
    t  = tl.load(tgt_ptr+ offs, mask=mask)
    keep = tl.rand(seed, offs) > p
    out = tl.where(keep, x, t)  # same seed ⇒ same mask
    out = (out - t * p) * scale  # scale to keep mean at 0
    tl.store(out_ptr + offs, out, mask=mask)

# backward – regenerate Bernoulli mask and scale grad
@triton.jit
def _seeded_mixout_bw(grad_ptr,
                      out_ptr,       # grad_input
                      n_elem, p, seed,
                      BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elem
    g = tl.load(grad_ptr + offs, mask=mask)
    keep = tl.rand(seed, offs) > p    # same seed ⇒ same mask
    g *= keep.to(g.dtype)
    tl.store(out_ptr + offs, g, mask=mask)

class SeededMixout(torch.autograd.Function):

    @staticmethod
    def forward(ctx, inp, tgt, p: float, seed: int):
        inp = inp if inp.is_contiguous() else inp.contiguous()

        out = torch.empty_like(inp)
        n = inp.numel()
        grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
        scale = 1.0 / (1 - p)

        _seeded_mixout_fw[grid](inp, out, tgt, n, p, scale, seed, BLOCK_SIZE=1024)
        # stash only scalars needed for backward
        ctx.p, ctx.seed, ctx.n_elem = p, seed, n
        return out

    @staticmethod
    def backward(ctx, grad_out):
        # grad_in = torch.empty_like(grad_out)
        grid = lambda meta: (triton.cdiv(ctx.n_elem, meta['BLOCK_SIZE']),)
        _seeded_mixout_bw[grid](grad_out, grad_out,
                                ctx.n_elem, ctx.p, ctx.seed,
                                BLOCK_SIZE=1024)
        return grad_out, None, None, None         # no grad for tgt, p, seed

class RandomMaskingLinear(torch.nn.Module):
    def __init__(self, base_weight: torch.Tensor, masking_prob: float = 0.0, mask_refresh: int = 1, mask_ema : float = 0.0):
        super().__init__()
        self.p = masking_prob
        self.delta_w = nn.Parameter(torch.zeros_like(base_weight))
        self.delta_w.data.copy_(base_weight.data)

        self.step = 1
        self.refresh = mask_refresh

        self.ema = mask_ema
        self.seed = random.randrange(2**32)

    @torch._dynamo.disable
    @torch.no_grad()
    def _maybe_refresh(self, p_weight, ema):
        self.seed = random.randrange(2**32)
        
        if ema < 1.0:
            p_weight.data.copy_(p_weight.data * ema + self.delta_w.data * (1 - ema))

        self.delta_w.data.copy_(p_weight.data)  # copy base values into deltas
    
    @torch._dynamo.disable
    def _kernel(self, p_weight):
        return SeededMixout.apply(self.delta_w,
                                       p_weight,
                                       self.p, self.seed)

    def forward(self, x, p_weight, p_bias):
        if self.training:
            if self.step == 1:
                self._maybe_refresh(p_weight, 1.0)
            if self.step % self.refresh == 0:
                self._maybe_refresh(p_weight, self.ema)
            
            self.step += 1
            w_mix = self._kernel(p_weight)
            return F.linear(x, w_mix, p_bias)
        else:
            return F.linear(x, p_weight, p_bias)