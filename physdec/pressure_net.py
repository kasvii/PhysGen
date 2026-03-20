import torch
import os
import torch.nn as nn
import numpy as np
import pyvista as pv
import pytorch_lightning as pl
import torch.optim as optim

from craftsman.models.autoencoders.utils import FourierEmbedder, DiagonalGaussianDistribution
from craftsman.utils.checkpoint import checkpoint
from craftsman.utils.saving import SaverMixin
from physdec.perceiver_1d import ParallelPerceiver, ParallelAdaptPerceiver
from physdec.attention import AdaptResidualCrossAttentionBlock

from typing import Dict, Any
from physdec.utils.eval import calculate_metrics, rel_l2_loss_batchwise

def build_scheduler(optimizer, cfg):
    scheduler_name = cfg.scheduler.name
    scheduler_args = cfg.scheduler.args

    if not hasattr(optim.lr_scheduler, scheduler_name):
        raise ValueError(f"Scheduler {scheduler_name} not found in torch.optim.lr_scheduler")

    scheduler_cls = getattr(optim.lr_scheduler, scheduler_name)
    return scheduler_cls(optimizer, **scheduler_args)

def build_optimizer(model, cfg):
    optim_name = cfg.optimizer.name
    optim_args = cfg.optimizer.args

    if not hasattr(optim, optim_name):
        raise ValueError(f"Optimizer {optim_name} not found in torch.optim")

    optimizer_cls = getattr(optim, optim_name)
    params = filter(lambda p: p.requires_grad, model.parameters())
    return optimizer_cls(params, **optim_args)

class PerceiverCrossAttentionDecoder(nn.Module):

    def __init__(self,
                 num_latents: int,
                 out_dim: int,
                 embedder: FourierEmbedder,
                 width: int,
                 heads: int,
                 init_scale: float = 0.25,
                 qkv_bias: bool = True,
                 use_flash: bool = False,
                 use_checkpoint: bool = False):

        super().__init__()

        self.use_checkpoint = use_checkpoint
        self.embedder = embedder

        self.query_proj = nn.Linear(self.embedder.out_dim, width)

        self.cross_attn_decoder = AdaptResidualCrossAttentionBlock(
            n_data=num_latents,
            width=width,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            use_flash=use_flash,
            use_checkpoint=use_checkpoint
        )

        self.ln_post = nn.LayerNorm(width)
        self.output_proj = nn.Linear(width, out_dim)

    def _forward(self, queries: torch.FloatTensor, latents: torch.FloatTensor):
        queries = self.query_proj(self.embedder(queries))
        x = self.cross_attn_decoder(queries, latents)
        x = self.ln_post(x)
        x = self.output_proj(x)
        return x

    def forward(self, queries: torch.FloatTensor, latents: torch.FloatTensor):
        logits = checkpoint(self._forward, (queries, latents), self.parameters(), self.use_checkpoint)
        return logits

class PressureEstimator(pl.LightningModule, SaverMixin):
    def __init__(self,
                 encoder,
                 cfg,
                 pre_kl,
                 num_latents: int,
                 out_dim: int,
                 embedder: FourierEmbedder,
                 width: int,
                 heads: int,
                 init_scale: float = 0.25,
                 qkv_bias: bool = True,
                 use_flash: bool = False,
                 use_checkpoint: bool = False,
                 vis_mesh: bool = False):

        super().__init__()

        self.cfg = cfg
        self.use_checkpoint = use_checkpoint
        self.encoder = encoder
        self.embedder = embedder
        self.pre_kl = pre_kl
        self.vis_mesh = vis_mesh

        if self.cfg.embed_dim > 0:
            self.post_kl = nn.Linear(self.cfg.embed_dim, self.cfg.width)
            self.latent_shape = (self.cfg.num_latents, self.cfg.embed_dim)
        else:
            self.latent_shape = (self.cfg.num_latents, self.cfg.width)

        self.transformer = ParallelAdaptPerceiver(
            n_ctx=self.cfg.num_latents,
            width=self.cfg.width,
            layers=self.cfg.num_decoder_layers,
            heads=self.cfg.heads,
            init_scale=self.cfg.init_scale,
            qkv_bias=self.cfg.qkv_bias,
            use_flash=self.cfg.use_flash,
            use_checkpoint=self.cfg.use_checkpoint
        )

        self.decoder = PerceiverCrossAttentionDecoder(
            embedder=self.embedder,
            out_dim=self.cfg.out_dim,
            num_latents=self.cfg.num_latents,
            width=self.cfg.width,
            heads=self.cfg.heads,
            init_scale=self.cfg.init_scale,
            qkv_bias=self.cfg.qkv_bias,
            use_flash=self.cfg.use_flash,
            use_checkpoint=self.cfg.use_checkpoint
        )

        if self.cfg.pretrained_model_name_or_path != "":
            print(f"Loading pretrained physics model from {self.cfg.pretrained_model_name_or_path}")
            ckpt = torch.load(self.cfg.pretrained_model_name_or_path, map_location='cpu')
            if 'state_dict' in ckpt:
                physics_state_dict = {}
                state_dict = ckpt['state_dict']

                has_physics_prefix = any(k.startswith('physics_model.') for k in state_dict.keys())

                for k, v in state_dict.items():
                    if k.startswith('encoder.') or k.startswith('pre_kl.') or k.startswith('shape_model.'):
                        continue

                    if has_physics_prefix:
                        if k.startswith('physics_model.'):
                            new_key = k.replace('physics_model.', '')
                            if not new_key.startswith('encoder.') and not new_key.startswith('pre_kl.'):
                                physics_state_dict[new_key] = v
                    else:
                        physics_state_dict[k] = v
                        
                missing_keys, unexpected_keys = self.load_state_dict(physics_state_dict, strict=False)

                print("Physics model loaded successfully")

    def encode_kl_embed(self, latents: torch.FloatTensor, sample_posterior: bool = True):
        posterior = None
        if self.cfg.embed_dim > 0:
            moments = self.pre_kl(latents)
            posterior = DiagonalGaussianDistribution(moments, feat_dim=-1)
            if sample_posterior:
                kl_embed = posterior.sample()
            else:
                kl_embed = posterior.mode()
        else:
            kl_embed = latents
        return kl_embed, posterior

    def encode(self,
               coarse_surface: torch.FloatTensor,
               sharp_surface: torch.FloatTensor,
               sample_posterior: bool = True,
               split: str = 'val'):
        """
        Args:
            coarse_surface (torch.FloatTensor): [B, N, 3+C]
            sharp_surface (torch.FloatTensor): [B, M, 3+C]
            sample_posterior (bool): Whether to sample from posterior or use mode.
            split (str): Data split identifier.

        Returns:
            shape_latents (torch.FloatTensor): [B, num_latents, width]
            kl_embed (torch.FloatTensor): [B, num_latents, embed_dim]
            posterior (DiagonalGaussianDistribution or None)
        """
        coarse_pc, coarse_feats = coarse_surface[..., :3], coarse_surface[..., 3:]
        sharp_pc, sharp_feats = sharp_surface[..., :3], sharp_surface[..., 3:]
        shape_latents = self.encoder(coarse_pc, sharp_pc, coarse_feats, sharp_feats, split=split)
        kl_embed, posterior = self.encode_kl_embed(shape_latents, sample_posterior)
        return shape_latents, kl_embed, posterior

    def decode(self, latents: torch.FloatTensor):
        """
        Args:
            latents (torch.FloatTensor): [B, num_latents, embed_dim]
        Returns:
            latents (torch.FloatTensor): [B, num_latents, width]
        """
        latents = self.post_kl(latents)
        return self.transformer(latents)

    def query(self, queries: torch.FloatTensor, latents: torch.FloatTensor):
        """
        Args:
            queries (torch.FloatTensor): [B, N, 3]
            latents (torch.FloatTensor): [B, num_latents, width]
        Returns:
            logits (torch.FloatTensor): [B, N]
        """
        logits = self.decoder(queries, latents).squeeze(-1)
        return logits

    def _forward(self, coarse_surface, sharp_surface, phys_points):
        with torch.no_grad():
            latents, kl_embed, posterior = self.encode(
                coarse_surface, sharp_surface,
                sample_posterior=self.cfg.sample_posterior,
                split='val')

        latents = self.decode(kl_embed)
        logits = self.query(phys_points, latents)
        return logits, latents

    def forward(self, coarse_surface, sharp_surface, phys_points):
        logits = self._forward(coarse_surface, sharp_surface, phys_points)
        return logits

    def training_step(self, batch, batch_idx):
        y = batch["phys_pressures"]
        coarse_surface = batch["coarse_surface"]
        sharp_surface = batch["sharp_surface"]
        phys_points = batch["phys_points"]
        y_pred, latents = self(coarse_surface, sharp_surface, phys_points)

        loss_mse = nn.functional.mse_loss(y_pred, y)
        loss_mae = nn.functional.l1_loss(y_pred, y)
        loss_r_l2 = rel_l2_loss_batchwise(y_pred, y) * 0.2
        loss = loss_mse + loss_mae + loss_r_l2

        self.log("train/train_loss_mse", loss_mse)
        self.log("train/train_loss_mae", loss_mae)
        self.log("train/train_loss_r_l2", loss_r_l2)
        self.log("train/train_loss_total", loss)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        self.eval()
        y = batch["phys_pressures"]
        coarse_surface = batch["coarse_surface"]
        sharp_surface = batch["sharp_surface"]
        phys_points = batch["phys_points"]
        y_pred, latents = self(coarse_surface, sharp_surface, phys_points)

        loss = nn.functional.mse_loss(y_pred, y)
        self.log("val/val_loss", loss)

        loss_l2 = rel_l2_loss_batchwise(y_pred, y)
        self.log("val/val_loss_l2", loss_l2)

        if self.vis_mesh and self.global_step % 100 == 0:
            self._save_pressure_meshes(batch, coarse_surface, sharp_surface)

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        self.eval()
        y = batch["phys_pressures"]
        coarse_surface = batch["coarse_surface"]
        sharp_surface = batch["sharp_surface"]
        phys_points = batch["phys_points"]
        y_pred, latents = self(coarse_surface, sharp_surface, phys_points)

        metrics = calculate_metrics(y, y_pred)

        log_line = ""
        for metric_name, metric_value in metrics.items():
            self.log(f"test/test_{metric_name}", metric_value)
            log_line += f"{metric_name}: {metric_value:.4f}, "

    def _save_pressure_meshes(self, batch, coarse_surface, sharp_surface,
                              use_gen_points=False):
        """Save predicted pressure fields as VTP mesh files."""
        try:
            phys_points = batch["gen_points"] if use_gen_points else batch["gen_points"]
            vertices = batch["gen_points"][:, :, :3].cpu().numpy()
            faces = batch["gen_faces"].cpu().numpy()

            pressure_pred, _ = self(coarse_surface, sharp_surface, phys_points)
            pressure_pred = self.cfg.PRESSURE_STD * pressure_pred + self.cfg.PRESSURE_MEAN

            batch_size = pressure_pred.shape[0]
            for i in range(batch_size):
                try:
                    verts = vertices[i].astype(np.float32)
                    tris = faces[i].astype(np.int32)
                    pressures = np.squeeze(pressure_pred[i].cpu().numpy()).astype(np.float32)

                    faces_pv = np.hstack([np.full((tris.shape[0], 1), 3), tris]).astype(np.int32)
                    mesh = pv.PolyData(verts, faces_pv.flatten())
                    mesh.point_data["p"] = pressures

                    save_path = self.get_save_path(
                        f"it{self.global_step}/{os.path.basename(batch['uid'][i])}".replace('.npz', '.vtp'))
                    mesh.save(save_path)
                except Exception as e:
                    print(f"Warning: Failed to save mesh for sample {i}: {e}")
                    continue
        except Exception as e:
            print(f"Warning: Visualization failed at step {self.global_step}: {e}")

    def configure_optimizers(self):
        optimizer = build_optimizer(self, self.cfg)

        if hasattr(self.cfg, 'scheduler') and self.cfg.scheduler is not None:
            scheduler = build_scheduler(optimizer, self.cfg)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1
                }
            }
        else:
            return {"optimizer": optimizer}