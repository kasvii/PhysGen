"""
Comprehensive multi-task system that integrates shape, physics, and drag decoders.
"""
import os
import sys
import torch
import torch.nn as nn
import torch.optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import pytorch_lightning as pl
from omegaconf import OmegaConf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import craftsman
from craftsman.systems.base import BaseSystem
from craftsman.utils.typing import *
from craftsman.utils.misc import get_rank
from craftsman.utils.saving import SaverMixin
from craftsman.models.autoencoders.utils import FourierEmbedder, DiagonalGaussianDistribution

@craftsman.register("comprehensive-multi-task-system")
class ComprehensiveMultiTaskSystem(BaseSystem):
    @dataclass
    class Config(BaseSystem.Config):
        shape_model_type: str = "michelangelo-autoencoder"
        shape_model: dict = field(default_factory=dict)
        shape_model_ckpt: str = ""

        phys_model: dict = field(default_factory=dict)
        physics_model_ckpt: str = ""

        drag_model: dict = field(default_factory=dict)
        drag_model_ckpt: str = ""

        sample_posterior: bool = True
        current_task: str = "shape"

        loss_weights: dict = field(default_factory=lambda: {
            "shape_coarse": 1.0,
            "shape_sharp": 2.0,
            "shape_kl": 0.1,
            "physics_mse": 1.0,
            "physics_rel": 0.1,
            "drag_mse": 1.0,
            "drag_mae": 0.1
        })

        freeze_encoder: bool = False
        freeze_shape_decoder: bool = False

    cfg: Config

    def configure(self):
        super().configure()

        self.is_multitask_checkpoint = (
            self.cfg.shape_model_ckpt == self.cfg.physics_model_ckpt == self.cfg.drag_model_ckpt
            and os.path.exists(self.cfg.shape_model_ckpt)
        )

        encoder_cls = craftsman.find(self.cfg.shape_model_type)

        shape_config = self.cfg.shape_model.copy()
        if self.is_multitask_checkpoint and "pretrained_model_name_or_path" in shape_config:
            shape_config["pretrained_model_name_or_path"] = ""

        shape_model_cfg = OmegaConf.structured(encoder_cls.Config(**shape_config))
        self.shape_model = encoder_cls(shape_model_cfg)

        if os.path.exists(self.cfg.shape_model_ckpt) and not self.is_multitask_checkpoint:
            pretrained_ckpt = torch.load(self.cfg.shape_model_ckpt, map_location="cpu")
            _pretrained_ckpt = {}

            state_dict = pretrained_ckpt['state_dict']
            has_shape_prefix = any(k.startswith('shape_model.') for k in state_dict.keys())

            if has_shape_prefix:
                for k, v in state_dict.items():
                    if k.startswith('shape_model.'):
                        _pretrained_ckpt[k.replace('shape_model.', '')] = v
            else:
                _pretrained_ckpt = state_dict

            self.shape_model.load_state_dict(_pretrained_ckpt, strict=False)

        if self.cfg.current_task in ["physics", "joint"]:
            self._initialize_physics_model()

        if self.cfg.current_task in ["drag", "joint"]:
            self._initialize_drag_model()

        self._apply_freezing()

        if self.is_multitask_checkpoint:
            self._load_multitask_checkpoint()

    def _load_multitask_checkpoint(self):
        """Load weights from a multi-task checkpoint."""
        checkpoint = torch.load(self.cfg.shape_model_ckpt, map_location="cpu")

        if 'state_dict' not in checkpoint:
            return

        state_dict = checkpoint['state_dict']

        self._load_component_from_multitask('shape_model', self.shape_model, state_dict)

        if hasattr(self, 'physics_model'):
            self._load_component_from_multitask('physics_model', self.physics_model, state_dict)

        if hasattr(self, 'drag_model'):
            self._load_component_from_multitask('drag_model', self.drag_model, state_dict)

    def _load_component_from_multitask(self, component_name, model, state_dict):
        """Load a specific component from multi-task state dict."""
        component_state = {}
        prefix = f"{component_name}."

        for key, value in state_dict.items():
            if key.startswith(prefix):
                new_key = key[len(prefix):]
                component_state[new_key] = value

        if component_state:
            model.load_state_dict(component_state, strict=False)

    def _initialize_physics_model(self):
        """Initialize physics model components."""
        sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'physdec'))
        from pressure_net import PressureEstimator
        from physdec.perceiver_1d import ParallelAdaptPerceiver
        from physdec.attention import AdaptResidualCrossAttentionBlock

        encoder = self.shape_model.encoder
        pre_kl = self.shape_model.pre_kl
        embedder = self.shape_model.embedder

        physics_cfg = self.cfg.phys_model.copy()
        if self.is_multitask_checkpoint:
            physics_cfg.pretrained_model_name_or_path = ""

        self.physics_model = PressureEstimator(
            encoder=encoder,
            cfg=physics_cfg,
            pre_kl=pre_kl,
            num_latents=self.cfg.phys_model.num_latents,
            out_dim=self.cfg.phys_model.out_dim,
            embedder=embedder,
            width=self.cfg.phys_model.width,
            heads=self.cfg.phys_model.heads,
            init_scale=self.cfg.phys_model.init_scale,
            qkv_bias=self.cfg.phys_model.qkv_bias,
            use_flash=self.cfg.phys_model.use_flash,
            use_checkpoint=self.cfg.phys_model.use_checkpoint
        )

        if os.path.exists(self.cfg.physics_model_ckpt) and not self.is_multitask_checkpoint:
            physics_ckpt = torch.load(self.cfg.physics_model_ckpt, map_location="cpu")
            if 'state_dict' in physics_ckpt:
                physics_state_dict = {}
                state_dict = physics_ckpt['state_dict']
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

                self.physics_model.load_state_dict(physics_state_dict, strict=False)

    def _initialize_drag_model(self):
        """Initialize drag model components."""
        sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dragdec'))
        from drag_net import DragEstimator, DragCoefficientDecoder

        encoder = self.shape_model.encoder
        pre_kl = self.shape_model.pre_kl
        embedder = self.shape_model.embedder

        drag_cfg = self.cfg.drag_model.copy()
        if self.is_multitask_checkpoint:
            drag_cfg.pretrained_model_name_or_path = ""

        self.drag_model = DragEstimator(
            encoder=encoder,
            embedder=embedder,
            pre_kl=pre_kl,
            cfg=drag_cfg,
            out_dim=self.cfg.drag_model.out_dim,
            num_latents=self.cfg.drag_model.num_latents,
            width=self.cfg.drag_model.width,
            heads=self.cfg.drag_model.heads,
            init_scale=self.cfg.drag_model.init_scale,
            qkv_bias=self.cfg.drag_model.qkv_bias,
            use_flash=self.cfg.drag_model.use_flash,
            use_checkpoint=self.cfg.drag_model.use_checkpoint
        )

        if os.path.exists(self.cfg.drag_model_ckpt) and not self.is_multitask_checkpoint:
            drag_ckpt = torch.load(self.cfg.drag_model_ckpt, map_location="cpu")
            if 'state_dict' in drag_ckpt:
                drag_state_dict = {}
                state_dict = drag_ckpt['state_dict']
                has_drag_prefix = any(k.startswith('drag_model.') for k in state_dict.keys())

                for k, v in state_dict.items():
                    if k.startswith('encoder.') or k.startswith('pre_kl.') or k.startswith('shape_model.') or k.startswith('physics_model.'):
                        continue

                    if has_drag_prefix:
                        if k.startswith('drag_model.'):
                            new_key = k.replace('drag_model.', '')
                            if not new_key.startswith('encoder.') and not new_key.startswith('pre_kl.'):
                                drag_state_dict[new_key] = v
                    else:
                        drag_state_dict[k] = v

                self.drag_model.load_state_dict(drag_state_dict, strict=False)

    def _apply_freezing(self):
        """Apply freezing configurations."""
        if self.cfg.freeze_encoder:
            for param in self.shape_model.encoder.parameters():
                param.requires_grad = False

        if self.cfg.freeze_shape_decoder:
            for param in self.shape_model.decoder.parameters():
                param.requires_grad = False

    def forward(self, batch: Dict[str, Any], split: str) -> Dict[str, Any]:
        if self.cfg.current_task == "shape":
            return self._forward_shape_task(batch, split)
        elif self.cfg.current_task == "physics":
            self.shape_model.split = split
            _, kl_embed, _ = self.shape_model.encode(
                batch["coarse_surface"], batch["sharp_surface"],
                sample_posterior=self.cfg.sample_posterior
            )
            return self._forward_physics_task(batch, kl_embed, split)
        elif self.cfg.current_task == "drag":
            self.shape_model.split = split
            _, kl_embed, _ = self.shape_model.encode(
                batch["coarse_surface"], batch["sharp_surface"],
                sample_posterior=self.cfg.sample_posterior
            )
            return self._forward_drag_task(batch, kl_embed, split)
        elif self.cfg.current_task == "joint":
            return self._forward_joint_task(batch, split)
        else:
            raise ValueError(f"Unknown task: {self.cfg.current_task}")

    def _forward_shape_task(self, batch: Dict[str, Any], split: str) -> Dict[str, Any]:
        num = batch["number_sharp"][0].item() if "number_sharp" in batch else 10000
        rand_points = batch["rand_points"]

        if "sdf" in batch:
            target = batch["sdf"]
            criteria = torch.nn.MSELoss()
        elif "occupancies" in batch:
            target = batch["occupancies"]
            criteria = torch.nn.BCEWithLogitsLoss()
        else:
            raise NotImplementedError

        coarse_target = target[:, num:]
        sharp_target = target[:, :num]

        self.shape_model.split = split
        shape_latents, kl_embed, posterior = self.shape_model.encode(
            batch["coarse_surface"], batch["sharp_surface"],
            sample_posterior=self.cfg.sample_posterior
        )
        latents = self.shape_model.decode(kl_embed)
        logits = self.shape_model.query(rand_points, latents)

        mean_value = torch.mean(kl_embed).item()
        variance_value = torch.var(kl_embed).item()

        coarse_logits = logits[:, num:]
        sharp_logits = logits[:, :num]

        outputs = {
            "loss_shape_coarse": criteria(coarse_logits, coarse_target).mean(),
            "loss_shape_sharp": criteria(sharp_logits, sharp_target).mean(),
            "overall_logits": logits,
            "coarse_logits": coarse_logits,
            "sharp_logits": sharp_logits,
            "overall_target": target,
            "coarse_target": coarse_target,
            "sharp_target": sharp_target,
            "latents": latents,
            "mean_value": mean_value,
            "variance_value": variance_value,
            "kl_embed": kl_embed
        }

        if self.cfg.sample_posterior and posterior is not None:
            loss_kl = posterior.kl()
            loss_kl = torch.sum(loss_kl) / loss_kl.shape[0]
            outputs["loss_shape_kl"] = loss_kl

        return outputs

    def _forward_physics_task(self, batch: Dict[str, Any], kl_embed, split: str) -> Dict[str, Any]:
        if not hasattr(self, 'physics_model'):
            return {}

        latents = self.physics_model.decode(kl_embed)
        pred = self.physics_model.query(batch["query_points"], latents)
        target = batch["pressure"]

        sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'physdec'))
        from physdec.utils.eval import rel_l2_loss_batchwise

        mse_loss = nn.MSELoss()(pred, target)
        mae_loss = nn.L1Loss()(pred, target)
        rel_loss = rel_l2_loss_batchwise(pred, target) * 0.2

        return {
            "loss_physics_mse": mse_loss,
            "loss_physics_mae": mae_loss,
            "loss_physics_rel": rel_loss,
            "physics_pred": pred,
            "physics_target": target
        }

    def _forward_drag_task(self, batch: Dict[str, Any], kl_embed, split: str) -> Dict[str, Any]:
        if not hasattr(self, 'drag_model'):
            return {}

        kl_embed = kl_embed * self.drag_model.z_scale_factor

        if self.drag_model.noisytrain and self.training:
            kl_embed, t = self.drag_model._noisify_latents_linear(kl_embed)
        else:
            t = torch.ones(kl_embed.shape[0], device=kl_embed.device)

        latents = self.drag_model.decode(kl_embed)

        t_cond = self.drag_model._time_condition(t).unsqueeze(1)
        latents = latents + t_cond

        pred = self.drag_model.predict_drag(latents)
        target = batch["drag_coefficient"]

        if target.dim() == 1:
            target = target.unsqueeze(-1)

        mse_loss = nn.MSELoss()(pred, target)
        mae_loss = nn.L1Loss()(pred, target)

        return {
            "loss_drag_mse": mse_loss,
            "loss_drag_mae": mae_loss,
            "drag_pred": pred,
            "drag_target": target
        }

    def _forward_joint_task(self, batch: Dict[str, Any], split: str) -> Dict[str, Any]:
        outputs = {}

        shape_outputs = self._forward_shape_task(batch, split)
        outputs.update(shape_outputs)

        if hasattr(self, 'physics_model') and "pressure" in batch and "query_points" in batch:
            physics_outputs = self._forward_physics_task(batch, shape_outputs["kl_embed"], split)
            outputs.update(physics_outputs)

        if hasattr(self, 'drag_model') and "drag_coefficient" in batch:
            drag_outputs = self._forward_drag_task(batch, shape_outputs["kl_embed"], split)
            outputs.update(drag_outputs)

        return outputs

    def training_step(self, batch, batch_idx):
        out = self(batch, 'train')

        loss = 0.0

        for name, value in out.items():
            if name.startswith("loss_"):
                weight_name = name.replace("loss_", "")
                if weight_name in self.cfg.loss_weights:
                    weight = self.cfg.loss_weights[weight_name]
                    weighted_loss = value * weight
                    loss += weighted_loss
                    self.log(f"train/{name}", value, prog_bar=False)
                    self.log(f"train/{name}_weighted", weighted_loss, prog_bar=False)

        self.log("train/loss", loss, prog_bar=True, on_step=True)

        return {"loss": loss}

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        self.eval()
        out = self(batch, 'val')

        val_loss = 0.0

        for name, value in out.items():
            if name.startswith("loss_"):
                weight_name = name.replace("loss_", "")
                if weight_name in self.cfg.loss_weights:
                    weight = self.cfg.loss_weights[weight_name]
                    weighted_loss = value * weight
                    val_loss += weighted_loss
                    self.log(f"val/{name}", value, sync_dist=True)

        self.log("val/loss", val_loss, prog_bar=True, sync_dist=True)

        return out

    def _log_shape_test_metrics(self, out: Dict[str, Any], batch: Dict[str, Any]):
        if "overall_logits" not in out:
            return

        threshold = 0
        overall_outputs = out["overall_logits"]
        overall_labels = (out["overall_target"] >= threshold).float()
        overall_pred = torch.zeros_like(overall_outputs)
        overall_pred[overall_outputs >= threshold] = 1

        overall_accuracy = (overall_pred == overall_labels).float().sum(dim=1) / overall_labels.shape[1]
        overall_accuracy = overall_accuracy.mean()

        overall_intersection = (overall_pred * overall_labels).sum(dim=1)
        overall_union = (overall_pred + overall_labels).gt(0).sum(dim=1)
        overall_iou = (overall_intersection * 1.0 / overall_union + 1e-5).mean()

        coarse_outputs = out["coarse_logits"]
        coarse_labels = (out["coarse_target"] >= threshold).float()
        coarse_pred = torch.zeros_like(coarse_outputs)
        coarse_pred[coarse_outputs >= threshold] = 1
        coarse_accuracy = ((coarse_pred == coarse_labels).float().sum(dim=1) / coarse_labels.shape[1]).mean()
        coarse_intersection = (coarse_pred * coarse_labels).sum(dim=1)
        coarse_union = (coarse_pred + coarse_labels).gt(0).sum(dim=1)
        coarse_iou = (coarse_intersection * 1.0 / coarse_union + 1e-5).mean()

        sharp_outputs = out["sharp_logits"]
        sharp_labels = (out["sharp_target"] >= threshold).float()
        sharp_pred = torch.zeros_like(sharp_outputs)
        sharp_pred[sharp_outputs >= threshold] = 1
        sharp_accuracy = ((sharp_pred == sharp_labels).float().sum(dim=1) / sharp_labels.shape[1]).mean()
        sharp_intersection = (sharp_pred * sharp_labels).sum(dim=1)
        sharp_union = (sharp_pred + sharp_labels).gt(0).sum(dim=1)
        sharp_iou = (sharp_intersection * 1.0 / sharp_union + 1e-5).mean()

        self.log("test/shape_overall_accuracy", overall_accuracy, sync_dist=True, on_epoch=True)
        self.log("test/shape_overall_iou", overall_iou, sync_dist=True, on_epoch=True)
        self.log("test/shape_coarse_accuracy", coarse_accuracy, sync_dist=True, on_epoch=True)
        self.log("test/shape_coarse_iou", coarse_iou, sync_dist=True, on_epoch=True)
        self.log("test/shape_sharp_accuracy", sharp_accuracy, sync_dist=True, on_epoch=True)
        self.log("test/shape_sharp_iou", sharp_iou, sync_dist=True, on_epoch=True)

    def _log_physics_test_metrics(self, out: Dict[str, Any], batch: Dict[str, Any]):
        if "physics_pred" not in out:
            return

        y_pred = out["physics_pred"]
        y = out["physics_target"]

        sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'physdec'))
        from physdec.utils.eval import calculate_metrics

        metrics = calculate_metrics(y, y_pred)

        for metric_name, metric_value in metrics.items():
            self.log(f"test/phys_{metric_name}", metric_value, sync_dist=True, on_epoch=True)

    def _log_drag_test_metrics(self, out: Dict[str, Any], batch: Dict[str, Any]):
        if "drag_pred" not in out:
            return

        y_pred = out["drag_pred"]
        y = out["drag_target"]

        if y.dim() == 1:
            y = y.unsqueeze(-1)

        mse = nn.functional.mse_loss(y_pred.squeeze(), y.squeeze())
        mae = nn.functional.l1_loss(y_pred.squeeze(), y.squeeze())
        max_ae = torch.max(torch.abs(y_pred.squeeze() - y.squeeze()))

        self.log("test/drag_mse", mse.item(), sync_dist=True, on_epoch=True)
        self.log("test/drag_mae", mae.item(), sync_dist=True, on_epoch=True)
        self.log("test/drag_max_mae", max_ae.item(), sync_dist=True, on_epoch=True, reduce_fx="max")

    def configure_optimizers(self):
        params_to_optimize = []

        if self.cfg.current_task == "shape":
            params_to_optimize.extend(list(self.shape_model.parameters()))

        elif self.cfg.current_task == "physics":
            if not self.cfg.freeze_encoder:
                params_to_optimize.extend(list(self.shape_model.encoder.parameters()))
                params_to_optimize.extend(list(self.shape_model.pre_kl.parameters()))

            if hasattr(self, 'physics_model'):
                params_to_optimize.extend(list(self.physics_model.post_kl.parameters()))
                params_to_optimize.extend(list(self.physics_model.transformer.parameters()))
                params_to_optimize.extend(list(self.physics_model.decoder.parameters()))

        elif self.cfg.current_task == "drag":
            if not self.cfg.freeze_encoder:
                params_to_optimize.extend(list(self.shape_model.encoder.parameters()))
                params_to_optimize.extend(list(self.shape_model.pre_kl.parameters()))

            if hasattr(self, 'drag_model'):
                params_to_optimize.extend(list(self.drag_model.post_kl.parameters()))
                params_to_optimize.extend(list(self.drag_model.transformer.parameters()))
                params_to_optimize.extend(list(self.drag_model.decoder.parameters()))

        elif self.cfg.current_task == "joint":
            params_to_optimize.extend(list(self.shape_model.parameters()))

            if hasattr(self, 'physics_model'):
                params_to_optimize.extend(list(self.physics_model.post_kl.parameters()))
                params_to_optimize.extend(list(self.physics_model.transformer.parameters()))
                params_to_optimize.extend(list(self.physics_model.decoder.parameters()))

            if hasattr(self, 'drag_model'):
                params_to_optimize.extend(list(self.drag_model.post_kl.parameters()))
                params_to_optimize.extend(list(self.drag_model.transformer.parameters()))
                params_to_optimize.extend(list(self.drag_model.decoder.parameters()))

        params_to_optimize = [p for p in params_to_optimize if p.requires_grad]

        if len(params_to_optimize) == 0:
            raise ValueError(f"No parameters to optimize for task: {self.cfg.current_task}")

        optimizer_config = self.cfg.get("optimizer", {})
        lr = optimizer_config.get("args", {}).get("lr", 1e-4)
        weight_decay = optimizer_config.get("args", {}).get("weight_decay", 1e-6)

        optimizer = torch.optim.AdamW(
            params_to_optimize,
            lr=lr,
            weight_decay=weight_decay
        )

        scheduler_config = self.cfg.get("scheduler", None)
        if scheduler_config is not None:
            scheduler_dict = {
                "scheduler": self._create_scheduler(optimizer, scheduler_config),
                "interval": scheduler_config.get("interval", "epoch"),
                "frequency": scheduler_config.get("frequency", 1),
            }
            return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}

        return optimizer

    def _create_scheduler(self, optimizer, scheduler_config):
        scheduler_name = scheduler_config.get("name", "cosine")
        scheduler_args = scheduler_config.get("args", {})

        if scheduler_name in ("CosineAnnealingLR", "cosine"):
            return CosineAnnealingLR(
                optimizer,
                T_max=scheduler_args.get("T_max", 100),
                eta_min=scheduler_args.get("eta_min", 1e-6)
            )
        elif scheduler_name in ("StepLR", "step"):
            return StepLR(
                optimizer,
                step_size=scheduler_args.get("step_size", 100),
                gamma=scheduler_args.get("gamma", 0.1)
            )
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_name}")

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        self.eval()
        out = self(batch, 'val')

        if self.cfg.current_task in ["shape", "joint"]:
            self._log_shape_test_metrics(out, batch)

        if self.cfg.current_task in ["physics", "joint"]:
            self._log_physics_test_metrics(out, batch)

        if self.cfg.current_task in ["drag", "joint"]:
            self._log_drag_test_metrics(out, batch)

        return out

    def on_test_epoch_end(self):
        pass