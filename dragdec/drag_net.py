import os
import math

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim

from craftsman.models.autoencoders.utils import DiagonalGaussianDistribution
from craftsman.utils.checkpoint import checkpoint
from craftsman.utils.saving import SaverMixin
from physdec.perceiver_1d import ParallelAdaptPerceiver
from uncond_diffusion.models.diffusion.transport.transport import Transport, ModelType, PathType, WeightType

def scalar_relative_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Calculate mean relative error for scalar predictions."""
    pred_flat = pred.squeeze()
    target_flat = target.squeeze()
    rel_error = torch.abs(pred_flat - target_flat) / (torch.abs(target_flat) + eps)
    return torch.mean(rel_error)

def build_scheduler(optimizer, cfg):
    scheduler_name = cfg.scheduler.name
    scheduler_args = cfg.scheduler.args

    if not hasattr(optim.lr_scheduler, scheduler_name):
        raise ValueError(f"Scheduler {scheduler_name} not found in torch.optim.lr_scheduler")

    scheduler_cls = getattr(optim.lr_scheduler, scheduler_name)
    scheduler = scheduler_cls(optimizer, **scheduler_args)
    
    return scheduler

def build_optimizer(model, cfg):
    optim_name = cfg.optimizer.name
    optim_args = cfg.optimizer.args

    if not hasattr(optim, optim_name):
        raise ValueError(f"Optimizer {optim_name} not found in torch.optim")

    optimizer_cls = getattr(optim, optim_name)
    params = filter(lambda p: p.requires_grad, model.parameters())

    return optimizer_cls(params, **optim_args)

class DragCoefficientDecoder(nn.Module):
    """Decode shape latents into a scalar drag coefficient."""

    def __init__(self,
                 width: int,
                 heads: int = 8,
                 use_checkpoint: bool = False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.mlp = nn.Sequential(
            nn.Linear(width, width // 2),
            nn.LayerNorm(width // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(width // 2, width // 4),
            nn.LayerNorm(width // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(width // 4, 1),
        )

    def _forward(self, latents: torch.FloatTensor):
        pooled = self.global_pool(latents.transpose(1, 2)).squeeze(-1)
        drag_coeff = self.mlp(pooled)
        return drag_coeff

    def forward(self, latents: torch.FloatTensor):
        return checkpoint(self._forward, (latents,), self.parameters(), self.use_checkpoint)

class DragEstimator(pl.LightningModule, SaverMixin):
    def __init__(self,
                 encoder,
                 embedder,
                 pre_kl,
                 cfg,
                 out_dim: int = 1,
                 num_latents: int = 256,
                 width: int = 512,
                 heads: int = 8,
                 init_scale: float = 0.25,
                 qkv_bias: bool = True,
                 use_flash: bool = False,
                 use_checkpoint: bool = False,
                 vis_mesh: bool = False,
                 noisytrain: bool = False,
                 z_scale_factor: float = 1.0,
                 train_sample_type: str = 'uniform',
                 train_eps: float = 0.0,
                 sample_eps: float = 0.0
                 ):

        super().__init__()

        self.cfg = cfg
        self.use_checkpoint = use_checkpoint
        self.encoder = encoder
        self.embedder = embedder
        self.pre_kl = pre_kl
        self.vis_mesh = vis_mesh
        self.out_dim = out_dim
        self.noisytrain = noisytrain
        self.z_scale_factor = z_scale_factor
        self.test_outputs = []

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
        self.decoder = DragCoefficientDecoder(
            width=self.cfg.width,
            heads=self.cfg.heads,
            use_checkpoint=self.cfg.use_checkpoint
        )
        self.transport = Transport(
            model_type=ModelType.VELOCITY,
            path_type=PathType.LINEAR,
            loss_type=WeightType.NONE,
            train_eps=train_eps,
            sample_eps=sample_eps,
            train_sample_type=train_sample_type,
        )
        self.path_sampler = self.transport.path_sampler
        self.time_embed_dim = min(128, self.cfg.width)
        self.time_proj = nn.Sequential(
            nn.Linear(self.time_embed_dim, self.cfg.width),
            nn.SiLU(),
            nn.Linear(self.cfg.width, self.cfg.width)
        )
        if self.cfg.pretrained_model_name_or_path != "":
            print(f"Loading pretrained drag model from {self.cfg.pretrained_model_name_or_path}")
            checkpoint_data = torch.load(self.cfg.pretrained_model_name_or_path, map_location="cpu")
            if "state_dict" in checkpoint_data:
                drag_state_dict = {}
                state_dict = checkpoint_data["state_dict"]
                has_drag_prefix = any(k.startswith("drag_model.") for k in state_dict.keys())

                for k, v in state_dict.items():
                    if (
                        k.startswith("encoder.")
                        or k.startswith("pre_kl.")
                        or k.startswith("shape_model.")
                        or k.startswith("physics_model.")
                    ):
                        continue

                    if has_drag_prefix:
                        if k.startswith("drag_model."):
                            new_key = k.replace("drag_model.", "")
                            if not new_key.startswith("encoder.") and not new_key.startswith("pre_kl."):
                                drag_state_dict[new_key] = v
                    else:
                        drag_state_dict[k] = v
                
                missing_keys, unexpected_keys = self.load_state_dict(drag_state_dict, strict=False)
                
                print("Drag model loaded successfully")
        
    def set_save_dir(self, save_dir):
        """Set save directory for outputs (following PhysDec pattern)"""
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

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
               split: str = 'val'
               ):
        """
        Args:
            surface (torch.FloatTensor): [B, N, 3+C]
            sample_posterior (bool):

        Returns:
            shape_latents (torch.FloatTensor): [B, num_latents, width]
            kl_embed (torch.FloatTensor): [B, num_latents, embed_dim]
            posterior (DiagonalGaussianDistribution or None):
        """
        
        coarse_pc, coarse_feats = coarse_surface[..., :3], coarse_surface[..., 3:] 
        sharp_pc, sharp_feats = sharp_surface[..., :3], sharp_surface[..., 3:] 
        shape_latents = self.encoder(coarse_pc, sharp_pc, coarse_feats, sharp_feats, split=split)
        kl_embed, posterior = self.encode_kl_embed(shape_latents, sample_posterior)

        return shape_latents, kl_embed, posterior
    
    def decode(self, 
               latents: torch.FloatTensor):
        """Project latents to the decoder width and apply the transformer."""
        latents = self.post_kl(latents)
        return self.transformer(latents)

    def predict_drag(self, latents: torch.FloatTensor):
        return self.decoder(latents)

    def _noisify_latents_linear(self, kl_embed: torch.Tensor):
        """Apply linear path noise to latent embeddings."""
        t, x0, x1 = self.transport.sample(kl_embed)
        if t.dim() > 1:
            t = t.view(t.shape[0])
        B = t.shape[0]
        t_view = t.view(B, 1, 1)
        xt_lin = t_view * x1 + (1.0 - t_view) * x0
        mask = (t < 0.75).view(B, 1, 1)
        xt = torch.where(mask, x1, xt_lin)
        t_plan = t.clone()
        t_plan[mask.view(B)] = 1.0
        return xt, t_plan

    def _sinusoidal_time_embedding(self, t: torch.Tensor) -> torch.Tensor:
        """Encode scalar timesteps into sinusoidal embeddings."""
        device = t.device
        half = self.time_embed_dim // 2
        if half < 1:
            return t.unsqueeze(-1)
        freqs = torch.exp(
            torch.linspace(0, -math.log(10000.0), steps=half, device=device)
        )
        args = t.unsqueeze(1) * freqs
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if emb.shape[1] < self.time_embed_dim:
            emb = torch.cat([emb, torch.zeros(emb.shape[0], 1, device=device)], dim=1)
        return emb[:, :self.time_embed_dim]

    def _time_condition(self, t: torch.Tensor) -> torch.Tensor:
        """Generate a time-conditioning vector with width-matched channels."""
        emb = self._sinusoidal_time_embedding(t)
        return self.time_proj(emb)

    def _forward(self, coarse_surface, sharp_surface):
        with torch.no_grad():
            _, kl_embed, _ = self.encode(
                coarse_surface,
                sharp_surface,
                sample_posterior=self.cfg.sample_posterior,
                split="val",
            )

        if self.noisytrain and self.training:
            kl_embed = kl_embed * self.z_scale_factor
            kl_embed, t = self._noisify_latents_linear(kl_embed)
            self.last_t = t
        else:
            kl_embed = kl_embed * self.z_scale_factor
            t = torch.ones(kl_embed.shape[0], device=kl_embed.device)
            self.last_t = t
        latents = self.decode(kl_embed)
        t_cond = self._time_condition(t).unsqueeze(1)
        latents = latents + t_cond
        return self.predict_drag(latents)

    def forward(self, coarse_surface, sharp_surface):
        drag_coeff = self._forward(coarse_surface, sharp_surface)
        return drag_coeff

    def training_step(self, batch, batch_idx):
        y = batch["drag_coefficient"]
        coarse_surface = batch["coarse_surface"]
        sharp_surface = batch["sharp_surface"]

        y_pred = self(coarse_surface, sharp_surface)
        loss_mse = nn.functional.mse_loss(y_pred.squeeze(), y.squeeze())
        loss_mae = nn.functional.l1_loss(y_pred.squeeze(), y.squeeze())
        loss = loss_mse + loss_mae

        self.log("train/train_loss_mse", loss_mse)
        self.log("train/train_loss_mae", loss_mae)
        self.log("train/train_loss_total", loss)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        self.eval()

        y = batch["drag_coefficient"]
        coarse_surface = batch["coarse_surface"]
        sharp_surface = batch["sharp_surface"]

        y_pred = self(coarse_surface, sharp_surface)
        loss_mse = nn.functional.mse_loss(y_pred.squeeze(), y.squeeze())
        loss_mae = nn.functional.l1_loss(y_pred.squeeze(), y.squeeze())
        loss_r_l2 = scalar_relative_loss(y_pred, y)

        self.log("val/val_loss_mse", loss_mse)
        self.log("val/val_loss_mae", loss_mae)
        self.log("val/val_loss_r_l2", loss_r_l2)
        self.log("val/val_loss", loss_mse)

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        self.eval()

        y = batch["drag_coefficient"]
        coarse_surface = batch["coarse_surface"]
        sharp_surface = batch["sharp_surface"]

        y_pred = self(coarse_surface, sharp_surface)
        mse = nn.functional.mse_loss(y_pred.squeeze(), y.squeeze())
        mae = nn.functional.l1_loss(y_pred.squeeze(), y.squeeze())

        y_pred_flat = y_pred.squeeze().cpu().numpy()
        y_true_flat = y.squeeze().cpu().numpy()

        if y_pred_flat.ndim == 0:
            y_pred_flat = np.array([y_pred_flat])
        if y_true_flat.ndim == 0:
            y_true_flat = np.array([y_true_flat])

        output = {
            "test_mse": mse.item(),
            "test_mae": mae.item(),
            "preds": y_pred_flat,
            "targets": y_true_flat,
            "batch_size": y.size(0),
        }
        self.test_outputs.append(output)

        return output

    @torch.no_grad()
    def on_test_epoch_end(self):
        """Aggregate test metrics across all batches."""
        if not self.test_outputs:
            return

        total_mse = 0.0
        total_mae = 0.0
        max_mae = 0.0
        total_samples = 0
        all_preds = []
        all_targets = []

        for output in self.test_outputs:
            total_mse += output["test_mse"]
            total_mae += output["test_mae"]
            max_mae = max(max_mae, output["test_mae"])
            total_samples += output["batch_size"]
            all_preds.append(output["preds"])
            all_targets.append(output["targets"])

        avg_mse = total_mse / len(self.test_outputs)
        avg_mae = total_mae / len(self.test_outputs)

        self.log("test/test_mse", avg_mse)
        self.log("test/test_mae", avg_mae)
        self.log("test/test_max_mae", max_mae)

        print(
            f"Test MSE: {avg_mse:.6f}, Test MAE: {avg_mae:.6f}, "
            f"Max MAE: {max_mae:.6f}"
        )
        print(f"Total samples: {total_samples}")
        self.test_outputs.clear()

    def configure_optimizers(self):
        optimizer = build_optimizer(self, self.cfg)

        if hasattr(self.cfg, "scheduler") and self.cfg.scheduler is not None:
            scheduler = build_scheduler(optimizer, self.cfg)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        else:
            return {"optimizer": optimizer}