import math
from operator import mul
from functools import reduce

import torch
import torch.nn as nn
import torch.nn.functional as F

def deterministic_permutation(
    num_weights, subset_size, iteration, layer_id, base_seed=0x1265EC, device="cpu"
):
    seed = ((iteration * 0x9E3779B97F4A7C15)
           ^ (layer_id * 0xBF58476D1CE4E5B9)
           ^ base_seed) & 0xFFFFFFFF_FFFFFFFF
    
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    perm = torch.randperm(num_weights, generator=g, device=device)
    return perm[:subset_size]

class VPT(nn.Module):
    def __init__(self, vpt_len, seq_len, patch_size, emb_dim, dtype=None):
        super().__init__()
        self.seq_len = seq_len
        self.prompt = nn.Parameter(torch.empty(vpt_len, emb_dim, dtype=dtype))
        init_val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + emb_dim))
        nn.init.uniform_(self.prompt, -init_val, init_val)
    
    @property
    def dtype(self):
        return self.prompt.dtype

    def forward(self, x):
        x = x[:, :self.seq_len, :]
        prompt = self.prompt.expand(x.shape[0], -1, -1)
        x = torch.cat([x, prompt], dim=1)
        return x

class Adapter(nn.Module):
    def __init__(self, in_dim, bottle_dim, dtype=None):
        super().__init__()
        self.ln = nn.LayerNorm(in_dim, dtype=dtype)
        self.down_proj = nn.Linear(in_dim, bottle_dim, dtype=dtype)
        self.relu = nn.ReLU(inplace=True)
        self.up_proj = nn.Linear(bottle_dim, in_dim, dtype=dtype)

        nn.init.kaiming_normal_(self.down_proj.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.bias)
    
    @property
    def dtype(self):
        return self.ln.weight.dtype
    
    def forward(self, x):
        x = self.ln(x)
        x = self.down_proj(x)
        x = self.relu(x)
        x = self.up_proj(x)
        return x

class AdaptFormer(nn.Module):
    def __init__(self, in_dim, bottle_dim, dtype=None):
        super().__init__()
        self.ln = nn.LayerNorm(in_dim, dtype=dtype)
        self.down_proj = nn.Linear(in_dim, bottle_dim, dtype=dtype)
        self.relu = nn.ReLU(inplace=True)
        self.up_proj = nn.Linear(bottle_dim, in_dim, dtype=dtype)
        self.scale = nn.Parameter(torch.ones(1, dtype=dtype))

        nn.init.kaiming_normal_(self.down_proj.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.bias)

    @property
    def dtype(self):
        return self.ln.weight.dtype

    def forward(self, x):
        x = self.ln(x)
        x = self.down_proj(x)
        x = self.relu(x)
        x = self.up_proj(x)
        x = x * self.scale
        return x

class SSF(nn.Module):
    def __init__(self, in_dim, dtype=None):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(in_dim, dtype=dtype))
        self.shift = nn.Parameter(torch.zeros(in_dim, dtype=dtype))
        nn.init.normal_(self.scale, mean=1.0, std=0.02)
        nn.init.normal_(self.shift, std=0.02)

    @property
    def dtype(self):
        return self.scale.dtype

    def forward(self, x):
        if len(x.shape) == 4:  # for CNN
            return x * self.scale.view(1, -1, 1, 1) + self.shift.view(1, -1, 1, 1)
        else:
            return x * self.scale + self.shift

class IA3(nn.Module):
    def __init__(self, in_dim, dtype=None):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(in_dim, dtype=dtype))
        self.reset_parameters()

    @property
    def dtype(self):
        return self.scale.dtype

    def reset_parameters(self):
        nn.init.ones_(self.scale)

    def forward(self, x):
        if len(x.shape) == 4:  # for CNN
            return x * self.scale.view(1, -1, 1, 1)
        else:
            return x * self.scale

class MaskedLinear(nn.Module):
    def __init__(self, weight, bias, ratio=0.0, generator=None):
        super().__init__()
        # weight: (out_dim, in_dim)
        # bias: (out_dim)
        out_dim, in_dim = weight.shape
        num_params = out_dim * in_dim + out_dim
        ratio = float(eval(ratio)) if isinstance(ratio, str) else float(ratio)
        num_masked = int(num_params * ratio)

        # randomly select the optimized parameters
        masked_indexs = torch.randperm(num_params, generator=generator)[:num_masked]
        mask = torch.zeros(num_params, dtype=bool).scatter(dim=0, index=masked_indexs, value=True)
        mask = mask.reshape(out_dim, in_dim + 1)
        self.mask_weight = mask[:,:-1]
        self.mask_bias = mask[:,-1]

        self.optimized_weight = nn.Parameter(torch.masked_select(weight.detach(), mask=self.mask_weight))
        self.optimized_bias = nn.Parameter(torch.masked_select(bias.detach(), mask=self.mask_bias))

    def forward(self, x, weight, bias):
        self.mask_weight = self.mask_weight.to(weight.device)
        self.mask_bias = self.mask_bias.to(bias.device)

        if self.mask_weight.sum() > 0:
            weight = torch.masked_scatter(weight, mask=self.mask_weight, source=self.optimized_weight)
        if self.mask_bias.sum() > 0:
            bias = torch.masked_scatter(bias, mask=self.mask_bias, source=self.optimized_bias)
        return F.linear(x, weight, bias)

class LoRA(nn.Module):
    def __init__(self, in_dim, bottle_dim, alpha=1, out_dim=None, dtype=None):
        super().__init__()
        self.in_dim = in_dim
        self.bottle_dim = bottle_dim
        self.out_dim = in_dim if out_dim is None else out_dim

        self.lora_A = nn.Parameter(torch.zeros(self.in_dim, self.bottle_dim, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(self.bottle_dim, self.out_dim, dtype=dtype))
        self.scaling = float(alpha) / self.bottle_dim

        self.reset_parameters()

    @property
    def dtype(self):
        return self.lora_A.dtype

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        x = x @ self.lora_A
        x = x @ self.lora_B
        x = x * self.scaling
        return x

class DoRALinear(nn.Module):
    def __init__(self, weight, bias, bottle_dim, alpha=1, dtype=None):
        super().__init__()

        out_dim, in_dim = weight.shape

        self.lora=LoRA(
            in_dim, bottle_dim, alpha, out_dim, dtype=dtype
        )
        self.m = nn.Parameter(
            weight.norm(p=2, dim=0, keepdim=True))

    def forward(self, x, weight, bias):
        lora = self.lora.lora_A @ self.lora.lora_B
        numerator = weight + self.lora.scaling*lora.T
        denominator = numerator.norm(p=2, dim=0, keepdim=True)
        directional_component = numerator / denominator
        new_weight = self.m * directional_component
        return F.linear(x, new_weight, bias)

class AdaLoRA(nn.Module):
    def __init__(self, in_dim, bottle_dim, alpha=1, out_dim=None, dtype=None):
        super().__init__()
        self.in_dim = in_dim
        self.bottle_dim = bottle_dim
        self.out_dim = in_dim if out_dim is None else out_dim

        self.lora_A = nn.Parameter(torch.zeros(self.in_dim, self.bottle_dim, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(self.bottle_dim, self.out_dim, dtype=dtype))
        self.lora_E = nn.Parameter(torch.zeros(1, self.bottle_dim, dtype=dtype)) 
        self.ranknum = nn.Parameter(
            torch.tensor(
                [float(self.bottle_dim)],
                dtype=dtype,
            ),
            requires_grad=False,
        )

        self.scaling = float(alpha)
        self.reset_parameters()

    @property
    def dtype(self):
        return self.lora_A.dtype

    def reset_parameters(self):
        if not hasattr(self, "lora_A"):
            return

        # Both factors must be nonzero so lora_E receives a gradient.
        nn.init.normal_(self.lora_A, mean=0.0, std=0.02)
        nn.init.normal_(self.lora_B, mean=0.0, std=0.02)

        if hasattr(self, "lora_E"):
            nn.init.zeros_(self.lora_E)

    def forward(self, x):
        # A: (in_dim, r)
        # E: (1, r), broadcast over A's input dimension
        # B: (r, out_dim)
        x = x @ (self.lora_A * self.lora_E)
        x = x @ self.lora_B

        return (
            x
            * self.scaling
            / (self.ranknum + 1e-5)
        )

class PACELoRA(nn.Module):
    def __init__(self, in_dim, bottle_dim, block_id, alpha=1, out_dim=None, dtype=None):
        super().__init__()
        self.in_dim = in_dim
        self.bottle_dim = bottle_dim
        self.out_dim = in_dim if out_dim is None else out_dim

        self.lora_A = nn.Parameter(torch.zeros(self.in_dim, self.bottle_dim, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(self.bottle_dim, self.out_dim, dtype=dtype))
        self.scaling = float(alpha) / self.bottle_dim

        self.sigma = 1.0  # initial sigma for PACELoRA
        self.shape = 'BC'  # noise shape: 'BC' for (batch_size, out_dim) or 'full' for full weight noise
        self.block_id = block_id

        self.reset_parameters()

    @property
    def dtype(self):
        return self.lora_A.dtype

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def _get_noise_shape(self, x):
        if len(x.shape) >= 2:
            if self.shape == 'BC':
                return [x.shape[0]] + [1] * (x.ndim - 2) + [x.shape[-1]]
            else:
                return x.shape
        else:
            raise NotImplementedError

    def _forward(self, x):
        x = x @ self.lora_A
        x = x @ self.lora_B
        x = x * self.scaling
        return x

    def forward(self, x):
        lora_out = self._forward(x)
        if self.training and self.sigma > 0:

            with torch.no_grad():
                noise_shape = self._get_noise_shape(lora_out)
                noise = torch.randn(*noise_shape, device=lora_out.device) * self.sigma + 1

            return lora_out * noise
        return lora_out
    
class SoMALinear(LoRA):
    def __init__(self, weight, qkv, bottle_dim, alpha=1, out_dim=None, dtype=None):
        out_dim, in_dim = weight.shape
        super().__init__(in_dim, bottle_dim, alpha, out_dim, dtype=dtype)

        with torch.no_grad():
            if qkv == "q":
                w = weight[:in_dim, :]
            elif qkv == "k":
                w = weight[in_dim:2 * in_dim, :]
            elif qkv == "v":
                w = weight[2 * in_dim:3 * in_dim, :]
            else:
                w = weight

            self.first_eigen = False

            self.soma_init(w)

    def soma_init(self, weight):
        dtype = weight.dtype
        if dtype not in [torch.float32, torch.float16, torch.bfloat16]:
            raise TypeError(
                "Please initialize SoRA under float32, float16, or bfloat16. "
                "Subsequently, re-quantize the residual model to help minimize quantization errors."
            )

        orig = weight
        weight_fp32 = orig.data.float()
        
        # Perform SVD and initialize SoRA with singular components
        U, S, Vh = torch.linalg.svd(weight_fp32, full_matrices=False)
        if self.first_eigen:
            Ur = U[:, :self.bottle_dim]
            Sr = S[:self.bottle_dim]
            Sr /= self.scaling
            Vhr = Vh[:self.bottle_dim, :]
        else:
            Ur = U[:, -self.bottle_dim:]
            Sr = S[-self.bottle_dim:]
            Sr /= self.scaling
            Vhr = Vh[-self.bottle_dim:, :]

        lora_A = torch.diag(torch.sqrt(Sr)) @ Vhr
        lora_B = Ur @ torch.diag(torch.sqrt(Sr))

        self.lora_A.data = lora_A.T
        self.lora_B.data = lora_B.T

        residual = weight_fp32 - self.scaling * (lora_B @ lora_A)
        orig.data.copy_(residual.to(orig.dtype))

    def forward(self, x):
        return super().forward(x)
    
class DenseGMixout(nn.Module):
    def __init__(self,
                 base_weight: torch.Tensor,
                 base_bias  : torch.Tensor,
                 *,
                 ratio        : float = 0.0,
                 mask_refresh : int   = 100,
                 mask_ema     : float = 0.0,
                 keep_momentum: bool = True,
                 use_mixout: bool = False,
                 generator=None):

        super().__init__()
        self.out_dim, self.in_dim = base_weight.shape
        self.num_w = base_weight.numel()
        self.num_b = base_bias.numel()
        self.keep_momentum = keep_momentum
        self.use_mixout = use_mixout

        self.ratio = float(ratio)
        self.sel_w = int(self.num_w * self.ratio)
        self.sel_b = int(self.num_b * self.ratio)
        self.refresh = int(mask_refresh)
        self.ema = float(mask_ema)

        self.generator = generator

        # ---- index list for current subset (buffers) ---------------------
        self.register_buffer("step", torch.tensor(1, dtype=torch.long))
        dev = base_weight.device
        self.register_buffer("w_idx", torch.empty(self.sel_w, dtype=torch.long, device=dev))
        self.register_buffer("b_idx", torch.empty(self.sel_b, dtype=torch.long, device=dev))
        self._sample_subset(base_weight.device)                       # fills w_idx / b_idx

        # ---- trainable deltas (full size) --------------------------------
        self.delta_w = nn.Parameter(torch.zeros_like(base_weight))
        self.delta_w.data.copy_(base_weight.data)  # copy base values into deltas
        self.delta_b = nn.Parameter(torch.zeros_like(base_bias))
        self.delta_b.data.copy_(base_bias.data)  # copy base values into deltas

        # ---- single gradient-mask hook -----------------------------------
        self._hook_w = self.delta_w.register_hook(self._make_grad_hook(self.w_idx))
        self._hook_b = self.delta_b.register_hook(self._make_grad_hook(self.b_idx))

        self._optim = None                          # filled by link_optimizer()
        if self.generator is not None: self.generator = torch.Generator('cuda').manual_seed(self.generator.initial_seed())

    # ------------------------------------------------------------------ API
    def link_optimizer(self, optim: torch.optim.Optimizer):
        """Call exactly once *after* the optimiser is created."""
        self._optim = optim

    # ---------------------------------------------------------------- helpers
    @torch.no_grad()
    def _sample_subset(self, device):
        """Draw a new random subset (on current device)."""
        if self.sel_w:
            self.w_idx.copy_(torch.randperm(self.num_w, device=device, generator=self.generator)[: self.sel_w])
        if self.sel_b:
            self.b_idx.copy_(torch.randperm(self.num_b, device=device, generator=self.generator)[: self.sel_b])

    @staticmethod
    def _make_grad_hook(idx):
        """Return a closure that zeros grads outside idx."""
        if idx.numel() == 0:          # allow ratio == 0
            return lambda g: torch.zeros_like(g)

        idx = idx.clone()             # capture a *copy* that never changes

        def hook(grad):
            flat = grad.view(-1).clone()
            flat.zero_()              # faster than masked_fill for sparse set
            flat[idx] = grad.view(-1)[idx]
            return flat.view_as(grad)
        return hook

    @torch.no_grad()
    def _update_optimizer_state(self, param, o_idx, n_idx):
        if self._optim is None:
            return
        if param in self._optim.state:
            for k, v in self._optim.state[param].items():
                if k == "step":
                    continue
                if torch.is_tensor(v):
                    cp_v = v.view(-1).clone()
                    v.zero_()
                    v = v.view(-1)
                    v[n_idx] = cp_v[o_idx]
                    v = v.view_as(param)

    @torch.no_grad()
    def _merge_and_refresh(self, p_weight, p_bias, ema):
        """EMA-merge deltas, reset optimiser state, resample subset."""

        p_weight.data.copy_(p_weight.data * ema + self.delta_w.data * (1 - ema))
        p_bias.data.copy_(p_bias.data * ema + self.delta_b.data * (1 - ema))

        if not self.use_mixout:
            self.delta_w.data.copy_(p_weight.data)  # copy base values into deltas
            self.delta_b.data.copy_(p_bias.data)  # copy base values into deltas

        old_w_idx = self.w_idx.clone()
        old_b_idx = self.b_idx.clone()

        self._sample_subset(p_weight.device)

        if self.keep_momentum:
            self._update_optimizer_state(self.delta_w, old_w_idx, self.w_idx)
            self._update_optimizer_state(self.delta_b, old_b_idx, self.b_idx)

        self._hook_w.remove()
        self._hook_b.remove()
        self._hook_w = self.delta_w.register_hook(self._make_grad_hook(self.w_idx))
        self._hook_b = self.delta_b.register_hook(self._make_grad_hook(self.b_idx))

    # ---------------------------------------------------------------- forward
    def _stitch_weight(self, p_weight):
        # Build full weight as autograd-tracked tensor (cheap).
        flat = p_weight.view(-1).clone()
        if self.sel_w:
            flat.index_copy_(0, self.w_idx, self.delta_w.view(-1)[self.w_idx])
        return flat.view_as(p_weight)

    def _stitch_bias(self, p_bias):
        flat = p_bias.view(-1).clone()
        if self.sel_b:
            flat.index_copy_(0, self.b_idx, self.delta_b.view(-1)[self.b_idx])
        return flat.view_as(p_bias)

    def forward(self, x, p_weight, p_bias):
        if self.training:
            if self.step == 1:
                # First step: copy base values into deltas. Especially important for resuming training.
                self._merge_and_refresh(p_weight, p_bias, 1.0)
            if self.step % self.refresh == 0:
                self._merge_and_refresh(p_weight, p_bias, self.ema)

            self.step += 1

            w = self._stitch_weight(p_weight)
            b = self._stitch_bias(p_bias)

            if self.ratio > 0 and self.ratio != 1.0:
                inv_r = 1.0 / self.ratio

                w = (w - p_weight * (1-self.ratio)) * inv_r
                b = (b - p_bias * (1-self.ratio)) * inv_r

            return F.linear(x, w, b)
        else:
            if self.use_mixout:
                return F.linear(x, self.delta_w, self.delta_b)
            else:
                return F.linear(x, p_weight, p_bias)

class DetDyMaskedLinear(DenseGMixout):
    def __init__(self,
                 base_weight: torch.Tensor,
                 base_bias  : torch.Tensor,
                 layer_idx  : int,
                 *,
                 ratio        : float = 0.0,
                 mask_refresh : int   = 100,
                 mask_ema     : float = 0.0,
                 use_mixout: bool = False,
                 mask_seed=0):

        self.mask_seed = mask_seed
        self.layer_idx = layer_idx

        super().__init__(base_weight=base_weight,
                         base_bias=base_bias,
                         ratio=ratio,
                         mask_refresh=mask_refresh,
                         mask_ema=mask_ema)

    @torch.no_grad()
    def _sample_subset(self, device):
        """Draw a new random subset (on current device)."""
        if self.sel_w:
            wp = deterministic_permutation(
                self.num_w, self.sel_w, self.step.item(), self.layer_idx, base_seed=self.mask_seed, device=device)
            self.w_idx.copy_(wp)
        if self.sel_b:
            bp = deterministic_permutation(
                self.num_b, self.sel_b, self.step.item(), self.layer_idx + 1, base_seed=self.mask_seed, device=device)
            self.b_idx.copy_(bp)
