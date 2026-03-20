import torch
import torch.nn as nn

from .attention import (
    ParallelResidualAttentionBlock,
    AdaptResidualAttentionBlock,
    ParallelAdaptResidualAttentionBlock,
)

class ParallelPerceiver(nn.Module):
    def __init__(self, *, n_ctx: int, width: int, layers: int, heads: int,
                 init_scale: float = 0.25, qkv_bias: bool = True,
                 use_flash: bool = False, use_checkpoint: bool = False):
        super().__init__()
        self.n_ctx = n_ctx
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList([
            ParallelResidualAttentionBlock(
                n_ctx=n_ctx, width=width, heads=heads,
                init_scale=init_scale, qkv_bias=qkv_bias,
                use_flash=use_flash, use_checkpoint=use_checkpoint
            )
            for _ in range(layers)
        ])

    def forward(self, x: torch.Tensor):
        for block in self.resblocks:
            x = block(x)
        return x

class AdaptPerceiver(nn.Module):
    def __init__(self, *, n_ctx: int, width: int, layers: int, heads: int,
                 init_scale: float = 0.25, qkv_bias: bool = True,
                 use_flash: bool = False, use_checkpoint: bool = False):
        super().__init__()
        self.n_ctx = n_ctx
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList([
            AdaptResidualAttentionBlock(
                n_ctx=n_ctx, width=width, heads=heads,
                init_scale=init_scale, qkv_bias=qkv_bias,
                use_flash=use_flash, use_checkpoint=use_checkpoint
            )
            for _ in range(layers)
        ])

    def forward(self, x: torch.Tensor):
        for block in self.resblocks:
            x = block(x)
        return x

class ParallelAdaptPerceiver(nn.Module):
    def __init__(self, *, n_ctx: int, width: int, layers: int, heads: int,
                 init_scale: float = 0.25, qkv_bias: bool = True,
                 use_flash: bool = False, use_checkpoint: bool = False):
        super().__init__()
        self.n_ctx = n_ctx
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList([
            ParallelAdaptResidualAttentionBlock(
                n_ctx=n_ctx, width=width, heads=heads,
                init_scale=init_scale, qkv_bias=qkv_bias,
                use_flash=use_flash, use_checkpoint=use_checkpoint
            )
            for _ in range(layers)
        ])

    def forward(self, x: torch.Tensor):
        for block in self.resblocks:
            x = block(x)
        return x