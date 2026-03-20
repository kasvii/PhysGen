"""
Multi-Decoder Model that integrates Shape, Physics, and Drag decoders.
"""
import torch
import torch.nn as nn
import numpy as np
import sys
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import craftsman
from craftsman.models.autoencoders.utils import DiagonalGaussianDistribution, get_embedder
from craftsman.utils.base import BaseModule
from craftsman.utils.typing import *

@craftsman.register("multi-decoder-model")
class MultiDecoderModel(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        shape_model: dict = None
        physics_decoder: dict = None
        drag_decoder: dict = None

        shape_model_ckpt: Optional[str] = None
        physics_model_ckpt: Optional[str] = None
        drag_model_ckpt: Optional[str] = None

        freeze_encoder: bool = False
        freeze_shape_decoder: bool = False
        freeze_physics_decoder: bool = False
        freeze_drag_decoder: bool = False

    cfg: Config

    def configure(self):
        super().configure()

        self.shape_model = craftsman.find("michelangelo-autoencoder")(self.cfg.shape_model)

        embedder, _ = get_embedder(
            multires=self.cfg.physics_decoder.num_freqs,
            input_dims=3,
            include_input=True,
            include_pi=self.cfg.physics_decoder.include_pi
        )

        sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'physdec'))
        from pressure_net import PerceiverCrossAttentionDecoder as PhysicsDecoder

        self.physics_decoder = PhysicsDecoder(
            num_latents=self.cfg.physics_decoder.num_latents,
            out_dim=self.cfg.physics_decoder.out_dim,
            embedder=embedder,
            width=self.cfg.physics_decoder.width,
            heads=self.cfg.physics_decoder.heads,
            init_scale=self.cfg.physics_decoder.init_scale,
            qkv_bias=self.cfg.physics_decoder.qkv_bias,
            use_flash=self.cfg.physics_decoder.use_flash,
            use_checkpoint=self.cfg.physics_decoder.use_checkpoint
        )

        sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dragdec'))
        from drag_net import DragCoefficientDecoder

        self.drag_decoder = DragCoefficientDecoder(
            width=self.cfg.drag_decoder.width,
            heads=self.cfg.drag_decoder.heads,
            use_checkpoint=self.cfg.drag_decoder.use_checkpoint
        )

        self._load_pretrained_weights()
        self._freeze_components()

    def _load_pretrained_weights(self):
        """Load pretrained weights from checkpoints."""
        if self.cfg.shape_model_ckpt:
            shape_ckpt = torch.load(self.cfg.shape_model_ckpt, map_location='cpu')
            if 'state_dict' in shape_ckpt:
                shape_state_dict = {}
                for k, v in shape_ckpt['state_dict'].items():
                    if k.startswith('shape_model.'):
                        shape_state_dict[k.replace('shape_model.', '')] = v
                self.shape_model.load_state_dict(shape_state_dict, strict=False)
            else:
                self.shape_model.load_state_dict(shape_ckpt, strict=False)

        if self.cfg.physics_model_ckpt:
            physics_ckpt = torch.load(self.cfg.physics_model_ckpt, map_location='cpu')
            if 'state_dict' in physics_ckpt:
                physics_state_dict = {}
                for k, v in physics_ckpt['state_dict'].items():
                    if k.startswith('phys_decoder.'):
                        physics_state_dict[k.replace('phys_decoder.', '')] = v
                self.physics_decoder.load_state_dict(physics_state_dict, strict=False)

        if self.cfg.drag_model_ckpt:
            drag_ckpt = torch.load(self.cfg.drag_model_ckpt, map_location='cpu')
            if 'state_dict' in drag_ckpt:
                drag_state_dict = {}
                for k, v in drag_ckpt['state_dict'].items():
                    if k.startswith('drag_decoder.'):
                        drag_state_dict[k.replace('drag_decoder.', '')] = v
                self.drag_decoder.load_state_dict(drag_state_dict, strict=False)

    def _freeze_components(self):
        """Freeze specified components."""
        if self.cfg.freeze_encoder:
            for param in self.shape_model.encoder.parameters():
                param.requires_grad = False

        if self.cfg.freeze_shape_decoder:
            for param in self.shape_model.decoder.parameters():
                param.requires_grad = False

        if self.cfg.freeze_physics_decoder:
            for param in self.physics_decoder.parameters():
                param.requires_grad = False

        if self.cfg.freeze_drag_decoder:
            for param in self.drag_decoder.parameters():
                param.requires_grad = False

    def encode(self, coarse_surface, sharp_surface, split="train", sample_posterior=True):
        """Encode input surfaces to latent space."""
        return self.shape_model.encode(coarse_surface, sharp_surface, split, sample_posterior)

    def decode_shape(self, latents, queries, split="train"):
        """Decode latents to shape (SDF/occupancy)."""
        return self.shape_model.decode(latents, queries, split)

    def decode_physics(self, latents, queries):
        """Decode latents to physics field (pressure)."""
        return self.physics_decoder(queries, latents)

    def decode_drag(self, latents):
        """Decode latents to drag coefficient."""
        return self.drag_decoder(latents)

    def forward(
        self,
        coarse_surface: torch.Tensor,
        sharp_surface: torch.Tensor,
        queries: torch.Tensor,
        physics_queries: Optional[torch.Tensor] = None,
        sample_posterior: bool = True,
        split: str = "train"
    ) -> Dict[str, Any]:
        """
        Forward pass through all decoders.

        Args:
            coarse_surface: Coarse surface points [B, N_coarse, 6] (xyz + normal).
            sharp_surface: Sharp surface points [B, N_sharp, 6] (xyz + normal).
            queries: Query points for shape reconstruction [B, N_queries, 3].
            physics_queries: Query points for physics field [B, N_phys, 3] (optional).
            sample_posterior: Whether to sample from posterior.
            split: Train/val split.

        Returns:
            Dictionary with all outputs and latents.
        """
        _, latents, posterior = self.shape_model.encode(
            coarse_surface, sharp_surface, split, sample_posterior
        )

        shape_logits = self.decode_shape(latents, queries, split)

        physics_output = None
        if physics_queries is not None:
            physics_output = self.decode_physics(latents, physics_queries)

        drag_output = self.decode_drag(latents)

        outputs = {
            "latents": latents,
            "posterior": posterior,
            "shape_logits": shape_logits,
            "drag_output": drag_output,
        }

        if physics_output is not None:
            outputs["physics_output"] = physics_output

        return outputs

    def extract_geometry_by_diffdmc(self, latents, octree_depth=9, save_slice_dir=None):
        """Extract geometry using DiffDMC (delegated to shape model)."""
        return self.shape_model.extract_geometry_by_diffdmc(latents, octree_depth, save_slice_dir)
