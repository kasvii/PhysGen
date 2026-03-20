import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import GELU

from craftsman.utils.typing import *
from craftsman.utils.checkpoint import checkpoint
from craftsman.models.transformers.utils import init_linear, MLP

def min_max_norm(x, eps=1e-4):
    min_val = x.min(dim=-1, keepdim=True).values
    max_val = x.max(dim=-1, keepdim=True).values
    denom = (max_val - min_val).clamp(min=eps)
    return (x - min_val) / denom

class MultiheadAttention(nn.Module):
    def __init__(self, *, n_ctx: int, width: int, heads: int,
                 init_scale: float, qkv_bias: bool, use_flash: bool = False):
        super().__init__()
        self.n_ctx = n_ctx
        self.width = width
        self.heads = heads
        self.c_qkv = nn.Linear(width, width * 3, bias=qkv_bias)
        self.c_proj = nn.Linear(width, width)
        self.attention = QKVMultiheadAttention(heads=heads, n_ctx=n_ctx, use_flash=use_flash)
        init_linear(self.c_qkv, init_scale)
        init_linear(self.c_proj, init_scale)

    def forward(self, x):
        x = self.c_qkv(x)
        x = checkpoint(self.attention, (x,), (), True)
        x = self.c_proj(x)
        return x

class QKVMultiheadAttention(nn.Module):
    def __init__(self, *, heads: int, n_ctx: int, use_flash: bool = False):
        super().__init__()
        self.heads = heads
        self.n_ctx = n_ctx
        self.use_flash = use_flash

    def forward(self, qkv):
        bs, n_ctx, width = qkv.shape
        attn_ch = width // self.heads // 3
        scale = 1 / math.sqrt(math.sqrt(attn_ch))
        qkv = qkv.view(bs, n_ctx, self.heads, -1)
        q, k, v = torch.split(qkv, attn_ch, dim=-1)

        if self.use_flash:
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)
            out = F.scaled_dot_product_attention(q, k, v).permute(0, 2, 1, 3).reshape(bs, n_ctx, -1)
        else:
            weight = torch.einsum("bthc,bshc->bhts", q * scale, k * scale)
            wdtype = weight.dtype
            weight = torch.softmax(weight.float(), dim=-1).type(wdtype)
            out = torch.einsum("bhts,bshc->bthc", weight, v).reshape(bs, n_ctx, -1)

        return out

class ResidualAttentionBlock(nn.Module):
    def __init__(self, *, n_ctx: int, width: int, heads: int,
                 init_scale: float = 1.0, qkv_bias: bool = True,
                 use_flash: bool = False, use_checkpoint: bool = False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.attn = MultiheadAttention(
            n_ctx=n_ctx, width=width, heads=heads,
            init_scale=init_scale, qkv_bias=qkv_bias, use_flash=use_flash
        )
        self.ln_1 = nn.LayerNorm(width)
        self.mlp = MLP(width=width, init_scale=init_scale)
        self.ln_2 = nn.LayerNorm(width)

    def _forward(self, x: torch.Tensor):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

    def forward(self, x: torch.Tensor):
        return checkpoint(self._forward, (x,), self.parameters(), self.use_checkpoint)

class AdaptMultiheadAttention(nn.Module):
    def __init__(self, *, n_ctx: int, width: int, heads: int,
                 init_scale: float, qkv_bias: bool, use_flash: bool = False):
        super().__init__()
        self.n_ctx = n_ctx
        self.width = width
        self.heads = heads
        self.c_qkv = nn.Linear(width, width * 3, bias=qkv_bias)
        self.c_proj = nn.Linear(width, width)
        self.attention = AdaptQKVMultiheadAttention(heads=heads, n_ctx=n_ctx, use_flash=use_flash)
        init_linear(self.c_qkv, init_scale)
        init_linear(self.c_proj, init_scale)

    def forward(self, x):
        x = self.c_qkv(x)
        x = checkpoint(self.attention, (x,), (), True)
        x = self.c_proj(x)
        return x

class AdaptQKVMultiheadAttention(nn.Module):
    """Self-attention with adaptive position-wise weighting."""

    def __init__(self, *, heads: int, n_ctx: int, use_flash: bool = False):
        super().__init__()
        self.heads = heads
        self.n_ctx = n_ctx
        self.use_flash = use_flash

    def _compute_adaptive_weight(self, q, k, scale, dim_seq):
        """Compute position-wise adaptive weight from Q-K interaction."""
        k_mean = k.mean(dim=dim_seq)  # [B, H, C]
        if dim_seq == 2:
            # Flash path: q is [B, H, T, C], need [B, T, H, C] for einsum
            q_reshaped = q.permute(0, 2, 1, 3)
        else:
            q_reshaped = q
        q_k_interaction = torch.einsum("bthc,bhc->bth", q_reshaped * scale, k_mean * scale)
        position_strength = q_k_interaction.sum(dim=-1)  # [B, T]

        min_vals = position_strength.min(dim=1, keepdim=True).values
        max_vals = position_strength.max(dim=1, keepdim=True).values
        active_weight = (position_strength - min_vals) / (max_vals - min_vals + 1e-6)
        return active_weight.unsqueeze(-1).unsqueeze(-1)  # [B, T, 1, 1]

    def forward(self, qkv):
        bs, n_ctx, width = qkv.shape
        attn_ch = width // self.heads // 3
        scale = 1 / math.sqrt(math.sqrt(attn_ch))
        qkv = qkv.view(bs, n_ctx, self.heads, -1)
        q, k, v = torch.split(qkv, attn_ch, dim=-1)

        if self.use_flash:
            q = q.permute(0, 2, 1, 3)  # [B, H, T, C]
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)
            out = F.scaled_dot_product_attention(q, k, v).permute(0, 2, 1, 3)  # [B, T, H, C]
            active_weight = self._compute_adaptive_weight(q, k, scale, dim_seq=2)
        else:
            weight = torch.einsum("bthc,bshc->bhts", q * scale, k * scale)
            wdtype = weight.dtype
            weight = torch.softmax(weight.float(), dim=-1).type(wdtype)
            out = torch.einsum("bhts,bshc->bthc", weight, v)  # [B, T, H, C]
            active_weight = self._compute_adaptive_weight(q, k, scale, dim_seq=1)

        out = active_weight * out
        return out.reshape(bs, n_ctx, -1)

class AdaptResidualAttentionBlock(nn.Module):
    def __init__(self, *, n_ctx: int, width: int, heads: int,
                 init_scale: float = 1.0, qkv_bias: bool = True,
                 use_flash: bool = False, use_checkpoint: bool = False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.attn = MultiheadAttention(
            n_ctx=n_ctx, width=width, heads=heads,
            init_scale=init_scale, qkv_bias=qkv_bias, use_flash=use_flash
        )
        self.ln_1 = nn.LayerNorm(width)
        self.mlp = MLP(width=width, init_scale=init_scale)
        self.ln_2 = nn.LayerNorm(width)

    def _forward(self, x: torch.Tensor):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

    def forward(self, x: torch.Tensor):
        return checkpoint(self._forward, (x,), self.parameters(), self.use_checkpoint)

class ParallelAdaptResidualAttentionBlock(nn.Module):
    """Parallel block with spatial attention, channel attention, and MLP branches."""

    def __init__(self, *, n_ctx: int, width: int, heads: int,
                 init_scale: float = 1.0, qkv_bias: bool = True,
                 use_flash: bool = False, use_checkpoint: bool = False):
        super().__init__()
        self.use_checkpoint = use_checkpoint

        # Spatial attention branch
        self.attn = AdaptMultiheadAttention(
            n_ctx=n_ctx, width=width, heads=heads,
            init_scale=init_scale, qkv_bias=qkv_bias, use_flash=use_flash
        )
        self.ln_1 = nn.LayerNorm(width)

        # Channel attention branch
        self.channel_attn = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width // 4),
            GELU(approximate="tanh"),
            nn.Linear(width // 4, width),
            nn.Sigmoid()
        )

        # MLP branch
        self.mlp = MLP(width=width, init_scale=init_scale)
        self.ln_2 = nn.LayerNorm(width)

        self.gate = nn.Parameter(torch.ones(3) / 3)

    def _forward(self, x: torch.Tensor):
        spatial_out = self.attn(self.ln_1(x))
        channel_weights = self.channel_attn(x)
        channel_out = x * channel_weights
        mlp_out = self.mlp(self.ln_2(x))

        weights = F.softmax(self.gate, dim=0)
        combined = weights[0] * spatial_out + weights[1] * channel_out + weights[2] * mlp_out
        return x + combined

    def forward(self, x: torch.Tensor):
        return checkpoint(self._forward, (x,), self.parameters(), self.use_checkpoint)

class ParallelResidualAttentionBlock(nn.Module):
    """Parallel block with spatial attention, channel attention, and MLP branches."""

    def __init__(self, *, n_ctx: int, width: int, heads: int,
                 init_scale: float = 1.0, qkv_bias: bool = True,
                 use_flash: bool = False, use_checkpoint: bool = False):
        super().__init__()
        self.use_checkpoint = use_checkpoint

        # Spatial attention branch
        self.attn = MultiheadAttention(
            n_ctx=n_ctx, width=width, heads=heads,
            init_scale=init_scale, qkv_bias=qkv_bias, use_flash=use_flash
        )
        self.ln_1 = nn.LayerNorm(width)

        # Channel attention branch
        self.channel_attn = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width // 4),
            GELU(approximate="tanh"),
            nn.Linear(width // 4, width),
            nn.Sigmoid()
        )

        # MLP branch
        self.mlp = MLP(width=width, init_scale=init_scale)
        self.ln_2 = nn.LayerNorm(width)

        self.gate = nn.Parameter(torch.ones(3) / 3)

    def _forward(self, x: torch.Tensor):
        spatial_out = self.attn(self.ln_1(x))
        channel_weights = self.channel_attn(x)
        channel_out = x * channel_weights
        mlp_out = self.mlp(self.ln_2(x))

        weights = F.softmax(self.gate, dim=0)
        combined = weights[0] * spatial_out + weights[1] * channel_out + weights[2] * mlp_out
        return x + combined

    def forward(self, x: torch.Tensor):
        return checkpoint(self._forward, (x,), self.parameters(), self.use_checkpoint)

class AdaptMultiheadCrossAttention(nn.Module):
    def __init__(self, *, width: int, heads: int, init_scale: float,
                 qkv_bias: bool = True, use_flash: bool = False,
                 n_data: Optional[int] = None, data_width: Optional[int] = None):
        super().__init__()
        self.n_data = n_data
        self.width = width
        self.heads = heads
        self.data_width = width if data_width is None else data_width
        self.c_q = nn.Linear(width, width, bias=qkv_bias)
        self.c_kv = nn.Linear(self.data_width, width * 2, bias=qkv_bias)
        self.c_proj = nn.Linear(width, width)
        self.attention = AdaptQKVMultiheadCrossAttention(
            heads=heads, n_data=n_data, use_flash=use_flash
        )
        init_linear(self.c_q, init_scale)
        init_linear(self.c_kv, init_scale)
        init_linear(self.c_proj, init_scale)

    def forward(self, x, data):
        x = self.c_q(x)
        data = self.c_kv(data)
        x = checkpoint(self.attention, (x, data), (), True)
        x = self.c_proj(x)
        return x

class AdaptQKVMultiheadCrossAttention(nn.Module):
    """Cross-attention with adaptive position-wise weighting."""

    def __init__(self, *, heads: int, use_flash: bool = False,
                 n_data: Optional[int] = None):
        super().__init__()
        self.heads = heads
        self.n_data = n_data
        self.use_flash = use_flash

    def forward(self, q, kv):
        _, n_ctx, _ = q.shape
        bs, n_data, width = kv.shape
        attn_ch = width // self.heads // 2
        scale = 1 / math.sqrt(math.sqrt(attn_ch))
        q = q.view(bs, n_ctx, self.heads, -1)
        kv = kv.view(bs, n_data, self.heads, -1)
        k, v = torch.split(kv, attn_ch, dim=-1)

        q = q.permute(0, 2, 1, 3)  # [B, H, T, C]
        k = k.permute(0, 2, 1, 3)  # [B, H, S, C]
        v = v.permute(0, 2, 1, 3)  # [B, H, S, C]
        out = F.scaled_dot_product_attention(q, k, v).permute(0, 2, 1, 3).reshape(bs, n_ctx, -1)

        k_sum = (k * scale).sum(dim=2)  # [B, H, C]
        q = q.permute(0, 2, 1, 3)  # [B, T, H, C]
        weight = (q * scale * k_sum.unsqueeze(1)).sum(dim=-1).sum(dim=-1)  # [B, T]
        min_vals = weight.min(dim=1, keepdim=True).values
        max_vals = weight.max(dim=1, keepdim=True).values
        active_weight = (weight - min_vals) / (max_vals - min_vals + 1e-6)
        out = active_weight.unsqueeze(-1) * out

        return out

class AdaptResidualCrossAttentionBlock(nn.Module):
    def __init__(self, *, n_data: Optional[int] = None, width: int, heads: int,
                 data_width: Optional[int] = None, init_scale: float = 0.25,
                 qkv_bias: bool = True, use_flash: bool = False,
                 use_checkpoint: bool = False):
        super().__init__()
        if data_width is None:
            data_width = width
        self.use_checkpoint = use_checkpoint
        self.attn = AdaptMultiheadCrossAttention(
            n_data=n_data, width=width, heads=heads, data_width=data_width,
            init_scale=init_scale, qkv_bias=qkv_bias, use_flash=use_flash,
        )
        self.ln_1 = nn.LayerNorm(width)
        self.ln_2 = nn.LayerNorm(data_width)
        self.mlp = MLP(width=width, init_scale=init_scale)
        self.ln_3 = nn.LayerNorm(width)

    def _forward(self, x: torch.Tensor, data: torch.Tensor):
        x = x + self.attn(self.ln_1(x), self.ln_2(data))
        x = x + self.mlp(self.ln_3(x))
        return x

    def forward(self, x: torch.Tensor, data: torch.Tensor):
        return checkpoint(self._forward, (x, data), self.parameters(), self.use_checkpoint)
