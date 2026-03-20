import os
import traceback
from contextlib import contextmanager
from typing import List, Tuple, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info
from pytorch_lightning.utilities import rank_zero_only

from ...utils.ema import LitEma
from ...utils.misc import instantiate_from_config, instantiate_non_trainable_model

import numpy as np
try:
    import pyvista as pv
    PYVISTA_AVAILABLE = True
except ImportError:
    print("Warning: pyvista not available, meshes with physics data cannot be saved")
    PYVISTA_AVAILABLE = False

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    print("Warning: trimesh not available, mesh processing may be limited")
    TRIMESH_AVAILABLE = False
except ImportError:
    PYVISTA_AVAILABLE = False
    print("Warning: pyvista not available, physics visualization will be disabled")


class Diffuser(pl.LightningModule):
    def __init__(
        self,
        *,
        first_stage_config,
        denoiser_cfg,
        scheduler_cfg,
        optimizer_cfg,
        pipeline_cfg=None,
        image_processor_cfg=None,
        lora_config=None,
        ema_config=None,
        physics_decoder_config=None,
        drag_decoder_config=None,
        noisy_drag_decoder_config=None,
        test_config=None,
        first_stage_key: str = "surface",
        scale_by_std: bool = False,
        z_scale_factor: float = 1.0,
        ckpt_path: Optional[str] = None,
        ignore_keys: Union[Tuple[str], List[str]] = (),
        torch_compile: bool = False,
        overfitting_debug: bool = False,
        # Deprecated parameters - kept for backward compatibility but ignored
        cond_stage_config=None,
        cond_stage_key: str = "image",
        **kwargs,
    ):
        super().__init__()
        self.first_stage_key = first_stage_key
        self.overfitting_debug = overfitting_debug
        self.physics_decoder_config = physics_decoder_config
        self.drag_decoder_config = drag_decoder_config
        self.noisy_drag_decoder_config = noisy_drag_decoder_config
        self.test_config = test_config
        
        if self.overfitting_debug:
            print("🔥 OVERFITTING DEBUG MODE ENABLED - Using deterministic noise and VAE encoding")
        
        # Warn about deprecated parameters
        if cond_stage_config is not None:
            print("Warning: cond_stage_config parameter is deprecated and ignored in unconditional model")
        if kwargs:
            print(f"Warning: Ignoring unexpected parameters: {list(kwargs.keys())}")

        self.optimizer_cfg = optimizer_cfg

        # ========= init diffusion scheduler ========= #
        self.scheduler_cfg = scheduler_cfg
        self.sampler = None
        if 'transport' in scheduler_cfg:
            self.transport = instantiate_from_config(scheduler_cfg.transport)
            self.sampler = instantiate_from_config(scheduler_cfg.sampler, transport=self.transport)
            self.sample_fn = self.sampler.sample_ode(**scheduler_cfg.sampler.ode_params)

        self.denoiser_cfg = denoiser_cfg
        self.model = instantiate_from_config(denoiser_cfg, device=None, dtype=torch.float32)
        # Force diffusion model to use FP32 for numerical stability
        self.ckpt_path = ckpt_path
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        if lora_config is not None:
            from peft import LoraConfig, get_peft_model
            loraconfig = LoraConfig(
                r=lora_config.rank,
                lora_alpha=lora_config.rank,
                target_modules=lora_config.get('target_modules')
            )
            self.model = get_peft_model(self.model, loraconfig)

        self.ema_config = ema_config
        if self.ema_config is not None:
            if self.ema_config.ema_model == 'DSEma':
                from ..utils.ema_deepspeed import DSEma
                self.model_ema = DSEma(self.model, decay=self.ema_config.ema_decay)
            else:
                self.model_ema = LitEma(self.model, decay=self.ema_config.ema_decay)
            #do not initilize EMA weight from ckpt path, since I need to change moe layers
            if ckpt_path is not None:
                self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        # ========= init vae at last to prevent it is overridden by loaded ckpt ========= #
        self.first_stage_model = instantiate_non_trainable_model(first_stage_config)
        # Convert VAE to bfloat16 for memory efficiency

        self.physics_decoder = None
        self.physics_enabled = False
        try:
            self._init_physics_decoder()
        except Exception as e:
            print(f"Warning: Failed to initialize physics decoder: {e}")
            print("Physics prediction will be disabled for mesh generation")

        self.drag_decoder = None
        self.drag_enabled = False
        try:
            self._init_drag_decoder()
        except Exception as e:
            print(f"Warning: Failed to initialize drag decoder: {e}")
            print("Drag prediction will be disabled for mesh generation")

        self.noisy_drag_decoder = None
        self.noisy_drag_enabled = False
        try:
            self._init_noisy_drag_decoder()
        except Exception as e:
            print(f"Warning: Failed to initialize noisy drag decoder: {e}")
            print("Noisy drag prediction will be disabled for mesh generation")

        self.scale_by_std = scale_by_std
        if scale_by_std:
            self.register_buffer("z_scale_factor", torch.tensor(z_scale_factor))
        else:
            self.z_scale_factor = z_scale_factor

        # ========= init pipeline for inference ========= #
        self.image_processor_cfg = image_processor_cfg
        self.image_processor = None
        if self.image_processor_cfg is not None:
            self.image_processor = instantiate_from_config(self.image_processor_cfg)
        self.pipeline_cfg = pipeline_cfg
        from ...schedulers import FlowMatchEulerDiscreteScheduler
        scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000)
        self.pipeline = instantiate_from_config(
            pipeline_cfg,
            vae=self.first_stage_model,
            model=self.model,
            scheduler=scheduler,
            image_processor=self.image_processor,
            drag_estimator=self.noisy_drag_decoder if self.test_config and self.test_config.get('use_drag_guidance', False) else None,
            phys_decoder=self.physics_decoder if self.test_config and self.test_config.get('use_drag_guidance', False) else None,
        )

        # ========= torch compile to accelerate ========= #
        self.torch_compile = torch_compile
        if self.torch_compile:
            torch.nn.Module.compile(self.model)
            torch.nn.Module.compile(self.first_stage_model)
            print(f'*' * 100)
            print(f'Compile model for acceleration')
            print(f'*' * 100)

    def _init_physics_decoder(self):
        """Initialize physics decoder for pressure prediction"""
        try:
            if self.physics_decoder_config is None:
                print("Physics decoder config not provided, skipping physics decoder initialization")
                self.physics_decoder = None
                self.physics_enabled = False
                return
                
            if not self.physics_decoder_config.get('enabled', False):
                print("Physics decoder disabled in config")
                self.physics_decoder = None
                self.physics_enabled = False
                return
            
            from physdec.pressure_net import PressureEstimator
            from craftsman.models.autoencoders.utils import FourierEmbedder
            
            physics_ckpt_path = self.physics_decoder_config.get('pretrained_model_name_or_path', "")
            
            if not physics_ckpt_path or not os.path.exists(physics_ckpt_path):
                print(f"Warning: Physics checkpoint not found at {physics_ckpt_path}")
                self.physics_decoder = None
                self.physics_enabled = False
                return
            
            # Create physics config from loaded config (no need for OmegaConf.create)
            physics_cfg = self.physics_decoder_config
            
            embedder = FourierEmbedder(
                input_dim=physics_cfg.get('point_feats', 3),
                num_freqs=physics_cfg.get('num_freqs', 8),
                include_pi=physics_cfg.get('include_pi', False)
            )
            
            self.physics_decoder = PressureEstimator(
                encoder=self.first_stage_model.encoder,
                cfg=physics_cfg,
                pre_kl=self.first_stage_model.pre_kl,
                num_latents=physics_cfg.get('num_latents', 256),
                out_dim=physics_cfg.get('out_dim', 1),
                embedder=embedder,
                width=physics_cfg.get('width', 768),
                heads=physics_cfg.get('heads', 12),
                init_scale=physics_cfg.get('init_scale', 0.25),
                qkv_bias=physics_cfg.get('qkv_bias', False),
                use_flash=physics_cfg.get('use_flash', True),
                use_checkpoint=physics_cfg.get('use_checkpoint', True),
                vis_mesh=physics_cfg.get('vis_mesh', False)
            )
            
            # Set to training mode to enable gradients for sampling guidance
            self.physics_decoder.train()

            for param in self.physics_decoder.parameters():
                param.requires_grad_(True)
            
            # Ensure physics decoder uses the same dtype as the main diffusion model
            # Since diffusion model is float32, convert physics decoder to float32 too
            main_model_dtype = next(self.model.parameters()).dtype
            
            self.physics_decoder = self.physics_decoder.to(dtype=main_model_dtype)
            
            # Explicitly convert all sub-components that might have been missed
            if hasattr(self.physics_decoder, 'post_kl') and self.physics_decoder.post_kl is not None:
                self.physics_decoder.post_kl = self.physics_decoder.post_kl.to(dtype=main_model_dtype)
            
            if hasattr(self.physics_decoder, 'transformer') and self.physics_decoder.transformer is not None:
                self.physics_decoder.transformer = self.physics_decoder.transformer.to(dtype=main_model_dtype)
            
            if hasattr(self.physics_decoder, 'decoder') and self.physics_decoder.decoder is not None:
                self.physics_decoder.decoder = self.physics_decoder.decoder.to(dtype=main_model_dtype)
            
            if hasattr(self.physics_decoder, 'embedder') and self.physics_decoder.embedder is not None:
                self.physics_decoder.embedder = self.physics_decoder.embedder.to(dtype=main_model_dtype)
            
            physics_params = list(self.physics_decoder.parameters())
            if physics_params:
                final_dtype = physics_params[0].dtype
                dtype_inconsistent = False
                for i, param in enumerate(physics_params[:10]):
                    if param.dtype != final_dtype:
                        dtype_inconsistent = True
            
            self.physics_enabled = True
                
        except Exception as e:
            print(f"Failed to initialize physics decoder: {e}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            self.physics_decoder = None
            self.physics_enabled = False

    def _init_drag_decoder(self):
        """Initialize drag decoder for drag coefficient prediction"""
        try:
            if self.drag_decoder_config is None:
                self.drag_decoder = None
                self.drag_enabled = False
                return
                
            if not self.drag_decoder_config.get('enabled', False):
                self.drag_decoder = None
                self.drag_enabled = False
                return
            
            from dragdec.drag_net import DragEstimator
            from craftsman.models.autoencoders.utils import FourierEmbedder
            
            drag_ckpt_path = self.drag_decoder_config.get('pretrained_model_name_or_path', "")
            
            if not drag_ckpt_path or not os.path.exists(drag_ckpt_path):
                self.drag_decoder = None
                self.drag_enabled = False
                return
            
            drag_cfg = self.drag_decoder_config
            
            embedder = FourierEmbedder(
                input_dim=drag_cfg.get('point_feats', 3),
                num_freqs=drag_cfg.get('num_freqs', 8),
                include_pi=drag_cfg.get('include_pi', False)
            )
            
            self.drag_decoder = DragEstimator(
                encoder=self.first_stage_model.encoder,
                cfg=drag_cfg,
                pre_kl=self.first_stage_model.pre_kl,
                num_latents=drag_cfg.get('num_latents', 256),
                out_dim=drag_cfg.get('out_dim', 1),
                embedder=embedder,
                width=drag_cfg.get('width', 768),
                heads=drag_cfg.get('heads', 12),
                init_scale=drag_cfg.get('init_scale', 0.25),
                qkv_bias=drag_cfg.get('qkv_bias', False),
                use_flash=drag_cfg.get('use_flash', True),
                use_checkpoint=drag_cfg.get('use_checkpoint', True),
                vis_mesh=drag_cfg.get('vis_mesh', False)
            )
            
            self.drag_decoder.eval()
            self.drag_enabled = True
                
        except Exception as e:
            print(f"Failed to initialize drag decoder: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            self.drag_decoder = None
            self.drag_enabled = False

    def _init_noisy_drag_decoder(self):
        """Initialize noisy drag decoder specifically for sampling-time drag estimation"""
        try:
            if self.noisy_drag_decoder_config is None:
                self.noisy_drag_decoder = None
                self.noisy_drag_enabled = False
                return
                
            if not self.noisy_drag_decoder_config.get('enabled', False):
                self.noisy_drag_decoder = None
                self.noisy_drag_enabled = False
                return
            
            from dragdec.drag_net import DragEstimator
            from craftsman.models.autoencoders.utils import FourierEmbedder
            
            noisy_drag_ckpt_path = self.noisy_drag_decoder_config.get('pretrained_model_name_or_path', "")
            
            if not noisy_drag_ckpt_path or not os.path.exists(noisy_drag_ckpt_path):
                self.noisy_drag_decoder = None
                self.noisy_drag_enabled = False
                return
            
            noisy_drag_cfg = self.noisy_drag_decoder_config
            
            embedder = FourierEmbedder(
                input_dim=noisy_drag_cfg.get('point_feats', 3),
                num_freqs=noisy_drag_cfg.get('num_freqs', 8),
                include_pi=noisy_drag_cfg.get('include_pi', False)
            )
            
            self.noisy_drag_decoder = DragEstimator(
                encoder=self.first_stage_model.encoder,
                cfg=noisy_drag_cfg,
                pre_kl=self.first_stage_model.pre_kl,
                num_latents=noisy_drag_cfg.get('num_latents', 256),
                out_dim=noisy_drag_cfg.get('out_dim', 1),
                embedder=embedder,
                width=noisy_drag_cfg.get('width', 768),
                heads=noisy_drag_cfg.get('heads', 12),
                init_scale=noisy_drag_cfg.get('init_scale', 0.25),
                qkv_bias=noisy_drag_cfg.get('qkv_bias', False),
                use_flash=noisy_drag_cfg.get('use_flash', True),
                use_checkpoint=noisy_drag_cfg.get('use_checkpoint', True),
                vis_mesh=noisy_drag_cfg.get('vis_mesh', False)
            )
            
            # Set to training mode to enable gradients for sampling guidance
            self.noisy_drag_decoder.train()
            
            # Important: Enable gradients for all noisy drag decoder parameters
            for param in self.noisy_drag_decoder.parameters():
                param.requires_grad_(True)
                
            self.noisy_drag_enabled = True
                
        except Exception as e:
            print(f"Failed to initialize noisy drag decoder: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            self.noisy_drag_decoder = None
            self.noisy_drag_enabled = False

    @contextmanager
    def ema_scope(self, context=None):
        if self.ema_config is not None and self.ema_config.get('ema_inference', False):
            self.model_ema.store(self.model)
            self.model_ema.copy_to(self.model)
        try:
            yield None
        finally:
            if self.ema_config is not None and self.ema_config.get('ema_inference', False):
                self.model_ema.restore(self.model)

    def init_from_ckpt(self, path, ignore_keys=()):
        ckpt = torch.load(path, map_location="cpu")
        if 'state_dict' not in ckpt:
            state_dict = {}
            for k in ckpt.keys():
                new_k = k.replace('_forward_module.', '')
                state_dict[new_k] = ckpt[k]
        else:
            state_dict = ckpt["state_dict"]

        keys = list(state_dict.keys())
        for k in keys:
            for ik in ignore_keys:
                if ik in k:
                    del state_dict[k]

        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
            print(f"Unexpected Keys: {unexpected}")

    def on_load_checkpoint(self, checkpoint):
        """
        The pt_model is trained separately, so we already have access to its
        checkpoint and load it separately with `self.set_pt_model`.

        However, the PL Trainer is strict about
        checkpoint loading (not configurable), so it expects the loaded state_dict
        to match exactly the keys in the model state_dict.

        So, when loading the checkpoint, before matching keys, we add all pt_model keys
        from self.state_dict() to the checkpoint state dict, so that they match
        """
        for key in self.state_dict().keys():
            if key.startswith("model_ema") and key not in checkpoint["state_dict"]:
                checkpoint["state_dict"][key] = self.state_dict()[key]
                
    def configure_optimizers(self) -> Tuple[List, List]:
        lr = self.learning_rate

        params_list = []
        trainable_parameters = list(self.model.parameters())
        params_list.append({'params': trainable_parameters, 'lr': lr})

        optimizer = instantiate_from_config(self.optimizer_cfg.optimizer, params=params_list, lr=lr)
        if hasattr(self.optimizer_cfg, 'scheduler'):
            scheduler_func = instantiate_from_config(
                self.optimizer_cfg.scheduler,
                max_decay_steps=self.trainer.max_steps,
                lr_max=lr
            )
            scheduler = {
                "scheduler": lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler_func.schedule),
                "interval": "step",
                "frequency": 1
            }
            schedulers = [scheduler]
        else:
            schedulers = []
        optimizers = [optimizer]

        return optimizers, schedulers

    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_start(self, batch, batch_idx):
        if self.scale_by_std and self.current_epoch == 0 and self.global_step == 0 \
            and batch_idx == 0 and self.ckpt_path is None:

            if 'coarse_surface' in batch and 'sharp_surface' in batch:
                coarse_surface = batch['coarse_surface']
                sharp_surface = batch['sharp_surface']
            else:
                # Fallback: try to use first_stage_key if it contains the surface data
                surface_key = self.first_stage_key
                if surface_key in batch:
                    surface_data = batch[surface_key]
                    if isinstance(surface_data, dict):
                        coarse_surface = surface_data['coarse_surface']
                        sharp_surface = surface_data['sharp_surface']
                    else:
                        raise ValueError("Surface data format not supported for std rescaling")
                else:
                    raise ValueError("Cannot find surface data for std rescaling")
                
            # Set split attribute for VAE encoder - always use 'val' for consistency  
            self.first_stage_model.split = 'val'
                
            shape_latents, z_q, posterior = self.first_stage_model.encode(
                coarse_surface=coarse_surface,
                sharp_surface=sharp_surface,
                sample_posterior=True
            )
            z = z_q.detach().float()

            del self.z_scale_factor
            self.register_buffer("z_scale_factor", 1. / z.flatten().std())

    def on_train_batch_end(self, *args, **kwargs):
        if self.ema_config is not None:
            self.model_ema(self.model)

    #     # Ensure model is always float32 at the start of each epoch

    #     """Ensure consistent setup for validation"""
    #     # Ensure models are in correct mode and dtype
    def _ensure_model_float32(self):
        """Utility method to ensure model is float32"""
        if hasattr(self, 'model'):
            self.model = self.model.float()

    def forward(self, batch, deterministic_for_overfitting=False):
        with torch.no_grad():
            if 'coarse_surface' in batch and 'sharp_surface' in batch:
                coarse_surface = batch['coarse_surface']
                sharp_surface = batch['sharp_surface']
                    
            else:
                raise ValueError(f"Cannot find coarse_surface and sharp_surface in batch. Available keys: {list(batch.keys())}")
                
            self.first_stage_model.split = 'val'
            
            if deterministic_for_overfitting:
                shape_latents, kl_embed, posterior = self.first_stage_model.encode(
                    coarse_surface=coarse_surface.half(),
                    sharp_surface=sharp_surface.half(),
                    sample_posterior=False
                )
            else:
                shape_latents, kl_embed, posterior = self.first_stage_model.encode(
                    coarse_surface=coarse_surface.half(),
                    sharp_surface=sharp_surface.half(),
                    sample_posterior=False
                )
            latents = kl_embed.float()
            
            latents = self.z_scale_factor * latents
            
        self.model = self.model.float()
        
        if deterministic_for_overfitting:
            current_seed = torch.get_rng_state()
            torch.manual_seed(42)
            loss = self.transport.training_losses(self.model, latents.float())["loss"].mean()
            torch.set_rng_state(current_seed)
        else:
            loss = self.transport.training_losses(self.model, latents.float())["loss"].mean()
            
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.forward(batch, deterministic_for_overfitting=self.overfitting_debug)
        split = 'train'
        loss_dict = {
            f"{split}/simple": loss.detach(),
            f"{split}/total_loss": loss.detach(),
            f"{split}/lr_abs": self.optimizers().param_groups[0]['lr'],
        }
            
        self.log_dict(loss_dict, prog_bar=True, logger=True, sync_dist=False, rank_zero_only=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.forward(batch, deterministic_for_overfitting=self.overfitting_debug)
        split = 'val'
        loss_dict = {
            f"{split}/simple": loss.detach(),
            f"{split}/total_loss": loss.detach(),
            f"{split}/lr_abs": self.optimizers().param_groups[0]['lr'],
        }
            
        self.log_dict(loss_dict, prog_bar=True, logger=True, sync_dist=False, rank_zero_only=True)

        # Generate meshes from flow matching samples for first few batches
        if batch_idx < 3:
            try:
                try:
                    batch_size = min(2, batch["coarse_surface"].shape[0])  # Limit batch size
                    sample_batch = {
                        "image": batch.get("image", None),
                        "mask": batch.get("mask", None)
                    }
                    
                    outputs = self.sample(sample_batch, output_type='latent')
                    if outputs and outputs[0] is not None:
                        generated_latents = outputs[0]
                    else:
                        raise ValueError("Pipeline returned None")
                        
                except Exception as pipeline_error:
                    print(f"   Pipeline method failed: {pipeline_error}")
                    print(f"   Trying direct flow matching sampling...")
                    
                    try:
                        batch_size = min(2, batch["coarse_surface"].shape[0])
                        device = next(self.model.parameters()).device
                        
                        with torch.no_grad():
                            sample_coarse = batch["coarse_surface"][:1].to(torch.bfloat16)
                            sample_sharp = batch["sharp_surface"][:1].to(torch.bfloat16)
                            self.first_stage_model.split = 'val'
                            _, sample_latents, _ = self.first_stage_model.encode(
                                coarse_surface=sample_coarse.half(),
                                sharp_surface=sample_sharp.half(),
                                sample_posterior=False
                            )
                            latent_shape = sample_latents.shape[1:]
                        
                        noise_shape = (batch_size,) + latent_shape
                        noise = torch.randn(noise_shape, device=device, dtype=torch.float32)
                        
                        contexts = {
                            'main': torch.zeros(batch_size, 1, 768, device=device, dtype=torch.float32)
                        }
                        
                        if hasattr(self, 'sample_fn') and self.sample_fn is not None:
                            generated_latents = self.sample_fn(
                                noise, 
                                self.model, 
                                contexts=contexts
                            )
                            # Take the last step (final result) from the sampling trajectory
                            generated_latents = generated_latents[-1]
                        else:
                            generated_latents = noise
                        
                    except Exception as direct_error:
                        print(f"   Direct sampling failed: {direct_error}")
                        print(f"   Using encoded latents from batch as fallback...")
                        
                        # Method C: Use encoded latents from current batch as fallback
                        with torch.no_grad():
                            coarse_surface = batch["coarse_surface"][:2].to(torch.bfloat16)
                            sharp_surface = batch["sharp_surface"][:2].to(torch.bfloat16)
                            self.first_stage_model.split = 'val'
                            _, generated_latents, _ = self.first_stage_model.encode(
                                coarse_surface=coarse_surface.half(),
                                sharp_surface=sharp_surface.half(),
                                sample_posterior=True
                            )
                            generated_latents = generated_latents.float()
                
                if generated_latents is not None:
                    meshes = self.decode_latents_to_mesh(
                        generated_latents, 
                        resolution=64,
                        batch_size=10000,  # Smaller batch size for memory efficiency
                        verbose=False
                    )
                    
                    if meshes and any(mesh is not None for mesh in meshes):
                        save_dir = self.get_mesh_save_dir("val_meshes")
                        saved_paths = self.save_meshes(meshes, save_dir, f"val_batch{batch_idx}")
                        successful_saves = len([p for p in saved_paths if p])
                        
                        self.log("val/meshes_generated", successful_saves, prog_bar=False, logger=True)
                    else:
                        print("No valid meshes generated")
                        self.log("val/meshes_generated", 0, prog_bar=False, logger=True)
                else:
                    print("Failed to generate latents")
                    self.log("val/meshes_generated", 0, prog_bar=False, logger=True)
                            
            except Exception as e:
                import traceback
                print(f"Mesh generation failed in validation: {e}")
                print(f"Traceback: {traceback.format_exc()}")
                self.log("val/meshes_generated", 0, prog_bar=False, logger=True)

        return loss

    def sample(self, batch, output_type='latent', **kwargs):
        batch_idx = batch.get('batch_idx', 0)
        seed = batch_idx
        generator = torch.Generator().manual_seed(seed)

        with self.ema_scope("Sample"):
            with torch.amp.autocast(device_type='cuda'):
                try:
                    self.pipeline.device = self.device
                    self.pipeline.dtype = self.dtype
                    additional_params = {'output_type': output_type}
                    
                    uid = None
                    if 'uid' in batch:
                        uid_data = batch['uid']
                        if isinstance(uid_data, list) and len(uid_data) > 0:
                            raw_uid = uid_data[0]
                            uid = os.path.basename(str(raw_uid))
                            for ext in ['.npz', '.obj', '.stl', '.ply']:
                                if uid.endswith(ext):
                                    uid = uid[:-len(ext)]
                                    break
                        elif isinstance(uid_data, str):
                            raw_uid = uid_data
                            uid = os.path.basename(str(raw_uid))
                            for ext in ['.npz', '.obj', '.stl', '.ply']:
                                if uid.endswith(ext):
                                    uid = uid[:-len(ext)]
                                    break

                    initial_latents = batch.get('initial_latents', None)

                    outputs = self.pipeline(
                        generator=generator, 
                        drag_guidance_scale=self.test_config.drag_guidance_scale, 
                        tgt_drag=batch.get('initial_drag_values', self.test_config.target_drag) - 0.03, 
                        init_drag=batch.get('initial_drag_values', None),
                        start_time_step=self.test_config.start_time_step, 
                        num_inference_steps=self.test_config.num_inference_steps,
                        num_optimization_loops=getattr(self.test_config, 'num_optimization_loops', 100),
                        num_optimization_steps=getattr(self.test_config, 'num_optimization_steps', 10),
                        learning_rate=getattr(self.test_config, 'learning_rate', 0.01),
                        angle_threshold=getattr(self.test_config, 'angle_threshold', 10.0),
                        scale_factor=self.z_scale_factor, 
                        test_save_dir=getattr(self.test_config, 'save_dir', 'save'),
                        uid=uid,
                        initial_latents=initial_latents,
                        **additional_params
                    )

                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print(f"Unexpected {e=}, {type(e)=}")
                    with open("error.txt", "a") as f:
                        f.write(str(e))
                        f.write(traceback.format_exc())
                        f.write("\n")
                    outputs = [None]

        return [outputs]


    def decode_latents_to_mesh(self, latents, resolution=128, batch_size=20000, verbose=False):
        """
        Decode latents to mesh using the VAE decoder and marching cubes
        Adapted from mesh_utils.generate_mesh_from_latents for flow matching
        
        Args:
            latents: (B, ...) latent tensors
            resolution: resolution for marching cubes
            batch_size: batch size for query processing  
            verbose: whether to print detailed debugging information
            
        Returns:
            meshes: list of trimesh objects or None if failed
        """
        try:
            try:
                import trimesh
                from skimage import measure
            except ImportError as e:
                print(f"Mesh generation dependencies not available: {e}")
                print("Please install: pip install trimesh scikit-image")
                return [None] * latents.shape[0]
            
            self.first_stage_model.eval()
            self.first_stage_model.split = 'val'
            
            device = latents.device
            B = latents.shape[0]
            
            with torch.no_grad():
                unscaled_latents = latents / self.z_scale_factor
                
                unscaled_latents = self.first_stage_model.post_kl(unscaled_latents.half())
                        
                unscaled_latents = self.first_stage_model.transformer(unscaled_latents)
                
                meshes = []
                for b in range(B):
                    if verbose:
                        print(f"Generating mesh {b+1}/{B}...")
                    
                    try:
                        single_latent = unscaled_latents[b:b+1]
                        
                        x = torch.linspace(-1, 1, resolution, device=device)
                        y = torch.linspace(-1, 1, resolution, device=device)
                        z = torch.linspace(-1, 1, resolution, device=device)
                        
                        grid = torch.meshgrid(x, y, z, indexing='ij')
                        queries = torch.stack(grid, dim=-1).reshape(-1, 3)
                        
                        # Process queries in batches to avoid memory issues
                        logits_list = []
                        num_queries = queries.shape[0]
                        
                        for i in range(0, num_queries, batch_size):
                            end_idx = min(i + batch_size, num_queries)
                            query_batch = queries[i:end_idx].unsqueeze(0).half()
                            
                            try:
                                if hasattr(self.first_stage_model, 'decoder') and hasattr(self.first_stage_model.decoder, '__call__'):
                                    logits_batch = self.first_stage_model.decoder(query_batch, single_latent)
                                else:
                                    logits_batch = self._fallback_sdf_query(query_batch, single_latent)
                            except Exception as decode_error:
                                logits_batch = self._fallback_sdf_query(query_batch, single_latent)
                            
                            logits_list.append(logits_batch.squeeze(0).squeeze(-1))
                        
                        logits = torch.cat(logits_list, dim=0)
                        
                        mesh = self._logits_to_mesh(logits, resolution, verbose, b)
                        meshes.append(mesh)
                        
                    except Exception as e:
                        print(f"Error generating mesh {b}: {e}")
                        meshes.append(None)
                
                return meshes
                
        except Exception as e:
            import traceback
            print(f"decode_latents_to_mesh failed: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            return [None] * latents.shape[0]
    
    def extract_geometry_by_diffdmc(self, latents, resolution=128, batch_size=10000, verbose=False, 
                                    bounds=(-1, -1, -1, 1, 1, 1), save_slice_dir=''):
        """
        Extract geometry using DiffDMC (Differentiable Dual Marching Cubes)
        Alternative to decode_latents_to_mesh with potentially better mesh quality
        
        Args:
            latents: (B, ...) latent tensors
            resolution: grid resolution (corresponds to octree_depth=log2(resolution))
            batch_size: batch size for query processing (num_chunks)
            verbose: whether to print detailed debugging information
            bounds: bounding box for mesh extraction
            save_slice_dir: directory to save slice visualization (optional)
            
        Returns:
            meshes: list of trimesh objects or None if failed
            
        Example usage (as drop-in replacement for decode_latents_to_mesh):
            # Original call:
            # meshes = self.decode_latents_to_mesh(latents, resolution=64, batch_size=10000)
            
            # Alternative with DiffDMC:
            # meshes = self.extract_geometry_by_diffdmc(latents, resolution=64, batch_size=10000)
        """
        try:
            try:
                import trimesh
                import numpy as np
                from einops import repeat
                from tqdm import trange
                import matplotlib.pyplot as plt
                from diso import DiffDMC
                from craftsman.utils.ops import generate_dense_grid_points
            except ImportError as e:
                print(f"DiffDMC dependencies not available: {e}")
                print("Please install: pip install trimesh einops tqdm matplotlib")
                return [None] * latents.shape[0]
            
            # Ensure VAE is in correct mode and dtype
            self.first_stage_model.eval()
            self.first_stage_model.split = 'val'
            
            device = latents.device
            B = latents.shape[0]
            
            with torch.no_grad():
                # Unscale latents first (same as decode_latents_to_mesh)
                unscaled_latents = latents / self.z_scale_factor
                if verbose:
                    print(f"Unscaled latents shape: {unscaled_latents.shape}")
                
                unscaled_latents = self.first_stage_model.post_kl(unscaled_latents.half())
                unscaled_latents = self.first_stage_model.transformer(unscaled_latents)
                
                if isinstance(bounds, float):
                    bounds = [-bounds, -bounds, -bounds, bounds, bounds, bounds]
                
                bbox_min = np.array(bounds[0:3])
                bbox_max = np.array(bounds[3:6])
                bbox_size = bbox_max - bbox_min
                
                octree_depth = int(np.log2(resolution)) if resolution > 0 else 8
                if verbose:
                    print(f"Using octree_depth={octree_depth} for resolution={resolution}")
                
                xyz_samples, grid_size, length, xs, ys, zs = generate_dense_grid_points(
                    bbox_min=bbox_min,
                    bbox_max=bbox_max,
                    octree_depth=octree_depth,
                    indexing="ij"
                )
                xyz_samples = torch.FloatTensor(xyz_samples).to(device)
                
                if verbose:
                    print(f"Generated {xyz_samples.shape[0]} grid points with grid_size {grid_size}")
                
                batch_logits = []
                for start in trange(0, xyz_samples.shape[0], batch_size, desc="Querying occupancy", disable=True):
                    queries = xyz_samples[start: start + batch_size, :]
                    batch_queries = repeat(queries, "p c -> b p c", b=B)
                    
                    # Use VAE decoder to query occupancy (similar to decode_latents_to_mesh)
                    try:
                        if hasattr(self.first_stage_model, 'decoder') and hasattr(self.first_stage_model.decoder, '__call__'):
                            logits = self.first_stage_model.decoder(batch_queries.half(), unscaled_latents)
                            if logits.dim() > 2:
                                logits = logits.squeeze(-1)
                        else:
                            logits = self._fallback_sdf_query(batch_queries.half(), unscaled_latents).squeeze(-1)
                    except Exception as e:
                        if verbose:
                            print(f"Query method failed, using fallback: {e}")
                        logits = self._fallback_sdf_query(batch_queries.half(), unscaled_latents).squeeze(-1)
                    
                    batch_logits.append(logits.float())
                
                grid_logits = torch.cat(batch_logits, dim=1).view((B, grid_size[0], grid_size[1], grid_size[2])).float()
                
                if verbose:
                    print(f"Grid logits shape: {grid_logits.shape}")
                
                if save_slice_dir:
                    self._save_slice_visualization(grid_logits, grid_size, save_slice_dir, verbose)
                
                meshes = []
                diffdmc = DiffDMC(dtype=torch.float32).to(device)
                
                for i in range(B):
                    if verbose:
                        print(f"Extracting mesh {i+1}/{B} using DiffDMC...")
                    
                    try:
                        logits_stats = grid_logits[i]
                        if verbose:
                            print(f"  Input logits - min: {logits_stats.min().item():.6f}, max: {logits_stats.max().item():.6f}, mean: {logits_stats.mean().item():.6f}")
                            print(f"  Positive logits count: {(logits_stats > 0).sum().item()}/{logits_stats.numel()}")
                        
                        if torch.all(logits_stats <= 0) or torch.all(logits_stats >= 0):
                            if verbose:
                                print(f"  Warning: All logits have same sign - may not produce valid mesh")
                        
                        diffdmc_result = diffdmc(-grid_logits[i], isovalue=0, normalize=False)
                        if verbose:
                            print(f"  DiffDMC returned {len(diffdmc_result) if hasattr(diffdmc_result, '__len__') else 'unknown'} values")
                        
                        # Extract vertices and faces (take first two values regardless of how many are returned)
                        if isinstance(diffdmc_result, (tuple, list)) and len(diffdmc_result) >= 2:
                            vertices, faces = diffdmc_result[0], diffdmc_result[1]
                        else:
                            print(f"  Warning: Unexpected DiffDMC result format: {type(diffdmc_result)}")
                            vertices, faces = diffdmc_result
                        
                        if verbose:
                            print(f"  Raw DiffDMC output - vertices: {vertices.shape if hasattr(vertices, 'shape') else 'N/A'}, faces: {faces.shape if hasattr(faces, 'shape') else 'N/A'}")
                        
                        if vertices is None or faces is None:
                            if verbose:
                                print(f"  DiffDMC returned None vertices or faces")
                            meshes.append(None)
                            continue
                            
                        vertices = (vertices + 1) / grid_size[0] * bbox_size[0] + bbox_min[0]
                        
                        if vertices.numel() > 0 and faces.numel() > 0:
                            vertices_np = vertices.detach().cpu().numpy().astype(np.float32)
                            faces_np = faces.detach().cpu().numpy().astype(np.int32)
                            
                            if verbose:
                                print(f"  Creating trimesh with vertices: {vertices_np.shape}, faces: {faces_np.shape}")
                            
                            if faces_np.max() >= len(vertices_np):
                                if verbose:
                                    print(f"  Warning: Invalid face indices - max index: {faces_np.max()}, num vertices: {len(vertices_np)}")
                                meshes.append(None)
                                continue
                            
                            try:
                                mesh = trimesh.Trimesh(vertices=vertices_np, faces=faces_np, validate=True)
                                
                                #         print(f"  Warning: Created mesh is invalid, attempting repair...")
                                if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
                                    if verbose:
                                        print(f"  Mesh became empty after processing")
                                    meshes.append(None)
                                    continue
                                
                                meshes.append(mesh)
                                if verbose:
                                    print(f"  Successfully created mesh with {len(vertices_np)} vertices, {len(faces_np)} faces")
                                    print(f"    Mesh area: {mesh.area:.6f}, watertight: {mesh.is_watertight}")
                                    
                            except Exception as mesh_error:
                                if verbose:
                                    print(f"  Error creating trimesh: {mesh_error}")
                                meshes.append(None)
                                
                        else:
                            meshes.append(None)
                            if verbose:
                                print(f"  No surface found for batch {i} - empty vertices ({vertices.numel()}) or faces ({faces.numel()})")
                                
                    except Exception as e:
                        meshes.append(None)
                        if verbose:
                            print(f"  DiffDMC failed for batch {i}: {e}")
                            print(f"  Traceback: {traceback.format_exc()}")
                
                if verbose:
                    successful_meshes = sum(1 for m in meshes if m is not None)
                    print(f"Successfully generated {successful_meshes}/{B} meshes using DiffDMC")
                
                return meshes
                
        except Exception as e:
            import traceback
            print(f"extract_geometry_by_diffdmc failed: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            return [None] * latents.shape[0]
    
    def _query_vae_occupancy(self, queries, decoded_features):
        """
        Query VAE for occupancy values at given 3D points
        This is a fallback method that needs customization based on your VAE structure
        """
        # This is a placeholder - you need to implement based on your VAE's architecture
        batch_size, num_queries, _ = queries.shape
        return torch.randn(batch_size, num_queries, 1, device=queries.device) * 0.1

    def _interpolate_decoded_features(self, queries, decoded_features):
        """
        Interpolate decoded features to get occupancy at query points
        This method assumes decoded_features contains spatial information
        """
        # This is a simplified version - you may need to adapt based on your VAE's output format
        batch_size, num_queries, _ = queries.shape
        
        # Simple distance-based interpolation as fallback
        # In practice, you'd want to use the actual decoded spatial representation
        distances = torch.norm(queries, dim=-1, keepdim=True)
        occupancy = torch.exp(-distances * 2.0)
        
        return occupancy

    def _fallback_sdf_query(self, queries, latents):
        """
        Fallback SDF query method using simple geometric shapes
        This creates a basic shape as a placeholder when no proper decoder is available
        """
        batch_size, num_queries, _ = queries.shape
        
        distances = torch.norm(queries, dim=-1, keepdim=True)
        radius = 0.8
        sdf = distances - radius
        
        # Convert SDF to occupancy logits (negative SDF means inside -> positive logits)
        logits = -sdf * 5.0
        
        return logits

    def _logits_to_mesh(self, logits, resolution, verbose=False, batch_idx=0):
        """
        Convert logits to mesh using marching cubes
        Adapted from mesh_utils.generate_mesh_from_latents
        """
        try:
            import trimesh
            from skimage import measure
            
            raw_min = logits.min().item()
            raw_max = logits.max().item()
            raw_mean = logits.mean().item()
            if verbose:
                print(f"   Batch {batch_idx}: raw logits range [{raw_min:.4f}, {raw_max:.4f}], mean={raw_mean:.4f}")
            
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                if verbose:
                    print(f"   Batch {batch_idx}: Found NaN or Inf in logits, replacing with zeros")
                logits = torch.where(torch.isnan(logits) | torch.isinf(logits), torch.zeros_like(logits), logits)
            
            # Clamp extreme values to prevent sigmoid saturation issues
            logits = torch.clamp(logits, min=-10.0, max=10.0)
            
            occupancy_grid = logits.reshape(resolution, resolution, resolution)
            occupancy_grid = torch.sigmoid(occupancy_grid)
            
            occupancy_np = occupancy_grid.cpu().numpy()
            
            min_val = occupancy_np.min()
            max_val = occupancy_np.max()
            mean_val = occupancy_np.mean()
            if verbose:
                print(f"   Batch {batch_idx}: occupancy range [{min_val:.4f}, {max_val:.4f}], mean={mean_val:.4f}")
            
            try:
                if max_val <= min_val:
                    if verbose:
                        print(f"   Batch {batch_idx}: Constant occupancy values, skipping mesh generation")
                    return None
                
                if min_val >= 0.5:
                    level = min_val + (max_val - min_val) * 0.1
                    if verbose:
                        print(f"   Batch {batch_idx}: Using adaptive level {level:.4f} (data above 0.5)")
                elif max_val <= 0.5:
                    level = max_val - (max_val - min_val) * 0.1
                    if verbose:
                        print(f"   Batch {batch_idx}: Using adaptive level {level:.4f} (data below 0.5)")
                else:
                    level = 0.5
                
                vertices, faces, _, _ = measure.marching_cubes(
                    occupancy_np, level=level, spacing=(2.0/resolution, 2.0/resolution, 2.0/resolution)
                )
                
                vertices = vertices - 1.0
                
                if len(vertices) > 0 and len(faces) > 0:
                    if TRIMESH_AVAILABLE:
                        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
                        mesh.remove_duplicate_faces()
                        mesh.remove_unreferenced_vertices()
                        
                        # Fix normals to ensure proper orientation
                        try:
                            mesh.fix_normals()
                            if mesh.is_watertight:
                                try:
                                    centroid = mesh.centroid
                                    face_centers = mesh.triangles_center
                                    face_normals = mesh.face_normals
                                    centroid_to_face = face_centers - centroid
                                    dot_products = np.sum(face_normals * centroid_to_face, axis=1)
                                    outward_facing = np.sum(dot_products > 0)
                                    inward_facing = np.sum(dot_products <= 0)
                                    
                                    if inward_facing > outward_facing:
                                        mesh.faces = np.fliplr(mesh.faces)
                                        mesh.fix_normals()
                                        if verbose:
                                            print(f"   Batch {batch_idx}: Flipped face orientation for outward normals")
                                except Exception as normal_error:
                                    if verbose:
                                        print(f"   Batch {batch_idx}: Normal orientation check failed: {normal_error}")
                        except Exception as fix_error:
                            if verbose:
                                print(f"   Batch {batch_idx}: Normal fixing failed: {fix_error}")
                        
                        if verbose:
                            print(f"   Batch {batch_idx}: Generated mesh with {len(vertices)} vertices, {len(faces)} faces")
                        return mesh
                    else:
                        print("Warning: trimesh not available, cannot create mesh object")
                        return None
                else:
                    if verbose:
                        print(f"   Empty mesh generated for batch {batch_idx}")
                    return None
                    
            except Exception as e:
                if verbose:
                    print(f"   Marching cubes failed for batch {batch_idx}: {e}")
                    print(f"   Trying with normalized occupancy values...")
                
                try:
                    if max_val > min_val:
                        occupancy_normalized = (occupancy_np - min_val) / (max_val - min_val)
                        vertices, faces, _, _ = measure.marching_cubes(
                            occupancy_normalized, level=0.5, spacing=(2.0/resolution, 2.0/resolution, 2.0/resolution)
                        )
                        vertices = vertices - 1.0
                        
                        if len(vertices) > 0 and len(faces) > 0:
                            if TRIMESH_AVAILABLE:
                                mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
                                mesh.remove_duplicate_faces()
                                mesh.remove_unreferenced_vertices()
                                
                                try:
                                    mesh.fix_normals()
                                    if mesh.is_watertight:
                                        try:
                                            centroid = mesh.centroid
                                            face_centers = mesh.triangles_center
                                            face_normals = mesh.face_normals
                                            centroid_to_face = face_centers - centroid
                                            dot_products = np.sum(face_normals * centroid_to_face, axis=1)
                                            outward_facing = np.sum(dot_products > 0)
                                            inward_facing = np.sum(dot_products <= 0)
                                            
                                            if inward_facing > outward_facing:
                                                mesh.faces = np.fliplr(mesh.faces)
                                                mesh.fix_normals()
                                                if verbose:
                                                    print(f"   Batch {batch_idx}: Flipped normalized mesh face orientation")
                                        except Exception as normal_error:
                                            if verbose:
                                                print(f"   Batch {batch_idx}: Normalized mesh normal check failed: {normal_error}")
                                except Exception as fix_error:
                                    if verbose:
                                        print(f"   Batch {batch_idx}: Normalized mesh normal fixing failed: {fix_error}")
                                
                                if verbose:
                                    print(f"   Batch {batch_idx}: Generated mesh with normalized values: {len(vertices)} vertices, {len(faces)} faces")
                                return mesh
                            else:
                                print("Warning: trimesh not available, cannot create normalized mesh")
                                return None
                        else:
                            if verbose:
                                print(f"   Empty normalized mesh for batch {batch_idx}")
                            return None
                    else:
                        if verbose:
                            print(f"   Cannot normalize constant values for batch {batch_idx}")
                        return None
                except Exception as e2:
                    if verbose:
                        print(f"   Normalized marching cubes also failed for batch {batch_idx}: {e2}")
                    return None
                    
        except Exception as e:
            print(f"_logits_to_mesh failed for batch {batch_idx}: {e}")
            return None

    def get_mesh_save_dir(self, mesh_type="val_meshes", test_save_dir=None, drag_guidance_scale=None, start_time_step=None, 
                         num_inference_steps=None, num_optimization_loops=None, num_optimization_steps=None, learning_rate=None):
        """
        Get the directory for saving meshes, using test config save_dir if available
        
        Args:
            mesh_type: type of mesh ("val_meshes", "test_meshes", etc.)
            test_save_dir: save directory from test config
            drag_guidance_scale: drag guidance scale value
            start_time_step: start time step value
            num_inference_steps: number of inference steps
            num_optimization_loops: number of optimization loops
            num_optimization_steps: number of optimization steps per loop
            learning_rate: learning rate for optimization
            
        Returns:
            Path to mesh save directory
        """
        import os
        import time
        from pathlib import Path
        
        if test_save_dir:
            save_dir = Path(test_save_dir)
            
            final_save_dir = save_dir / f"{mesh_type}"
            
            os.makedirs(final_save_dir, exist_ok=True)
            print(f"Using test config save_dir: {final_save_dir}")
            return str(final_save_dir.absolute())
        
        # Check if we have test config with save_dir for test meshes (backward compatibility)
        if mesh_type == "test_meshes" and self.test_config is not None:
            test_save_dir_config = self.test_config.get('save_dir')
            if test_save_dir_config:
                save_dir = Path(test_save_dir_config)
                
                final_save_dir = save_dir / f"{mesh_type}"
                
                os.makedirs(final_save_dir, exist_ok=True)
                print(f"Using test config save_dir: {final_save_dir}")
                return str(final_save_dir.absolute())
        
        output_dir = Path.cwd()
        save_base_dir = output_dir / "save"
        mesh_dir = save_base_dir / f"{mesh_type}"
        os.makedirs(mesh_dir, exist_ok=True)
        return str(mesh_dir.absolute())

    def _add_physics_to_meshes(self, meshes, generated_latents, dummy_coarse, dummy_sharp, save_dir, prefix):
        """
        Add physics prediction to meshes and save them with pressure data
        Based on pressure_net.py test_step implementation (lines 306-320)
        
        Args:
            meshes: list of trimesh objects
            generated_latents: diffusion-generated latents for physics decoder
            dummy_coarse: dummy coarse surface for physics decoder
            dummy_sharp: dummy sharp surface for physics decoder  
            save_dir: directory to save meshes with physics
            prefix: filename prefix (can be string or list of strings for each mesh)
            
        Returns:
            saved_paths: list of saved file paths with physics data
        """
        if not self.physics_enabled or self.physics_decoder is None:
            return None
            
        if not PYVISTA_AVAILABLE:
            print("Warning: pyvista not available, cannot save meshes with physics data")
            return self.save_meshes(meshes, save_dir, prefix)
            
        try:
            saved_paths = []
            os.makedirs(save_dir, exist_ok=True)
            
            if isinstance(prefix, list):
                prefixes = prefix[:len(meshes)]
                while len(prefixes) < len(meshes):
                    prefixes.append("mesh")
            else:
                prefixes = [prefix] * len(meshes)
            
            generated_latents = generated_latents / self.z_scale_factor

            for i, mesh in enumerate(meshes):
                current_prefix = prefixes[i]
                if mesh is not None:
                    try:
                        vertices = mesh.vertices.astype(np.float32)
                        faces = mesh.faces.astype(np.int32)
                        
                        phys_points = torch.from_numpy(vertices).unsqueeze(0).to(
                            generated_latents.device, dtype=torch.float32
                        )
                        
                        # Use physics decoder's query method with diffusion-generated latents
                        with torch.no_grad():
                            # First decode the generated latents using physics decoder
                            physics_latents = self.physics_decoder.decode(generated_latents[i:i+1])
                            
                            pressure_pred = self.physics_decoder.query(phys_points, physics_latents)
                            
                            # Denormalize pressure (following pressure_net.py line 301)
                            pressure_pred = (self.physics_decoder.cfg.get('PRESSURE_STD', 117.25) * pressure_pred + 
                                           self.physics_decoder.cfg.get('PRESSURE_MEAN', -94.5))
                            
                            pressures = pressure_pred.squeeze().cpu().numpy().astype(np.float32)
                        
                        # Build PyVista faces format (following pressure_net.py lines 308-312)
                        faces_pv = np.hstack([np.full((faces.shape[0], 1), 3), faces]).astype(np.int32)
                        faces_flat = faces_pv.flatten()
                        
                        pv_mesh = pv.PolyData(vertices, faces_flat)
                        pv_mesh.point_data["p"] = pressures
                        
                        # Save mesh with physics data (.vtp format for PyVista)
                        save_path = os.path.join(save_dir, f"{current_prefix}_{i:03d}.vtp")
                        pv_mesh.save(save_path)
                        saved_paths.append(save_path)
                        print(f"Saved pressure: {save_path}")
                        
                        # ply_path = os.path.join(save_dir, f"{current_prefix}_{i:03d}.ply")
                        # mesh.export(ply_path)
                        # print(f"Saved regular mesh: {ply_path}")
                        
                    except Exception as e:
                        print(f"Failed to add physics to mesh {i}: {e}")
                        save_path = os.path.join(save_dir, f"{current_prefix}_physics_fallback_{i:03d}.ply")
                        try:
                            mesh.export(save_path)
                            saved_paths.append(save_path)
                            print(f"Saved fallback mesh: {save_path}")
                        except Exception as e2:
                            print(f"Failed to save fallback mesh {i}: {e2}")
                            saved_paths.append(None)
                else:
                    saved_paths.append(None)
                    
            return saved_paths
            
        except Exception as e:
            print(f"_add_physics_to_meshes failed: {e}")
            return self.save_meshes(meshes, save_dir, prefix)
        
    def _add_drag_to_meshes(self, meshes, generated_latents, save_dir, prefix, return_drag=False):
        """
        Add drag prediction to meshes and save drag values as txt files
        Based on DragEstimator's predict_drag method
        
        Args:
            meshes: list of trimesh objects
            generated_latents: diffusion-generated latents for drag decoder
            save_dir: directory to save meshes and drag values
            prefix: filename prefix (can be string or list of strings for each mesh)
            
        Returns:
            saved_paths: list of saved file paths (meshes and drag values)
        """
        if not self.drag_enabled or self.drag_decoder is None:
            return None
            
        try:
            import traceback
            saved_paths = []
            os.makedirs(save_dir, exist_ok=True)
            
            if isinstance(prefix, list):
                prefixes = prefix[:len(meshes)]
                while len(prefixes) < len(meshes):
                    prefixes.append("mesh")
            else:
                prefixes = [prefix] * len(meshes)
            
            drag_params = list(self.drag_decoder.parameters())
            if drag_params:
                drag_dtype = drag_params[0].dtype
                drag_input_latents = generated_latents.to(dtype=drag_dtype)
            else:
                raise ValueError("Drag decoder has no parameters - cannot determine target dtype")
            
            try:
                self.drag_decoder = self.drag_decoder.to(dtype=drag_dtype)
                
                if hasattr(self.drag_decoder, 'post_kl') and self.drag_decoder.post_kl is not None:
                    self.drag_decoder.post_kl = self.drag_decoder.post_kl.to(dtype=drag_dtype)
                    for name, param in self.drag_decoder.post_kl.named_parameters():
                        if param.dtype != drag_dtype:
                            print(f"Warning: post_kl.{name} still has dtype {param.dtype}, manually converting")
                            param.data = param.data.to(dtype=drag_dtype)
                            
                if hasattr(self.drag_decoder, 'transformer') and self.drag_decoder.transformer is not None:
                    self.drag_decoder.transformer = self.drag_decoder.transformer.to(dtype=drag_dtype)
                    
                if hasattr(self.drag_decoder, 'decoder') and self.drag_decoder.decoder is not None:
                    self.drag_decoder.decoder = self.drag_decoder.decoder.to(dtype=drag_dtype)
                    
                if hasattr(self.drag_decoder, 'embedder') and self.drag_decoder.embedder is not None:
                    self.drag_decoder.embedder = self.drag_decoder.embedder.to(dtype=drag_dtype)
                if hasattr(self.drag_decoder, 'time_proj') and self.drag_decoder.time_proj is not None:
                    self.drag_decoder.time_proj = self.drag_decoder.time_proj.to(dtype=drag_dtype)
                    # Ensure all time_proj submodules are also converted
                    for name, param in self.drag_decoder.time_proj.named_parameters():
                        if param.dtype != drag_dtype:
                            print(f"Warning: time_proj.{name} still has dtype {param.dtype}, manually converting")
                            param.data = param.data.to(dtype=drag_dtype)

                inconsistent_params = []
                for name, param in self.drag_decoder.named_parameters():
                    if param.dtype != drag_dtype:
                        inconsistent_params.append((name, param.dtype))
                        param.data = param.data.to(dtype=drag_dtype)
                
                post_kl_dtype = next(self.drag_decoder.post_kl.parameters()).dtype if hasattr(self.drag_decoder, 'post_kl') and self.drag_decoder.post_kl is not None else "N/A"
                
                # Create dummy timestep with drag decoder's dtype for compatibility
                dummy_timestep = torch.tensor([1.0], device=drag_input_latents.device, dtype=drag_dtype)
                dummy_timestep = dummy_timestep.expand(drag_input_latents.shape[0])
                
                drag_latents = self.drag_decoder.decode(drag_input_latents)
                
                emb = self.drag_decoder._sinusoidal_time_embedding(dummy_timestep[:1]).to(dtype=drag_dtype)
                t_cond = self.drag_decoder.time_proj(emb).unsqueeze(1)
                
                # Ensure t_cond and drag_latents have compatible dtypes before addition
                if t_cond.dtype != drag_latents.dtype:
                    t_cond = t_cond.to(dtype=drag_latents.dtype)
                
                drag_latents = drag_latents + t_cond
                
                drag_predictions = self.drag_decoder.predict_drag(drag_latents)
                
            except Exception as drag_error:
                print(f"Drag decoder inference failed: {drag_error}")
                print(f"Traceback: {traceback.format_exc()}")
                
                raise drag_error
            
            # Denormalize drag coefficients (following DragEstimator pattern)
            drag_values = drag_predictions.squeeze().cpu().numpy().astype(np.float32)
            
            for i, mesh in enumerate(meshes):
                current_prefix = prefixes[i]
                if mesh is not None:
                    try:
                        mesh_path = os.path.join(save_dir, f"{current_prefix}_{i:03d}.ply")
                        
                        drag_path = os.path.join(save_dir, f"{current_prefix}_{i:03d}.txt")
                        
                        if len(drag_values.shape) == 0:
                            drag_value = float(drag_values)
                        else:
                            drag_value = float(drag_values[i]) if i < len(drag_values) else float(drag_values[0])
                        
                        with open(drag_path, 'w') as f:
                            f.write(f"# Drag coefficient prediction for mesh {i}\n")
                            f.write(f"# Generated using DragEstimator\n")
                            f.write(f"{drag_value:.6f}\n")
                        
                        saved_paths.append((mesh_path, drag_path))
                        print(f"Saved drag coefficient -> {drag_path}")

                    except Exception as e:
                        print(f"Failed to save mesh {i} with drag: {e}")
                        try:
                            mesh_path = os.path.join(save_dir, f"{current_prefix}_drag_{i:03d}_fallback.ply")
                            saved_paths.append((mesh_path, None))
                        except Exception as e2:
                            saved_paths.append((None, None))
                else:
                    saved_paths.append((None, None))

            if not return_drag:
                return saved_paths
            else:
                return saved_paths, drag_values

        except Exception as e:
            print(f"_add_drag_to_meshes failed: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            return self.save_meshes(meshes, save_dir, prefix)

    def save_meshes(self, meshes, save_dir, prefix="mesh"):
        """
        Save meshes to PLY files
        
        Args:
            meshes: list of trimesh objects
            save_dir: directory to save meshes
            prefix: filename prefix
            
        Returns:
            saved_paths: list of saved file paths
        """
        import os
        
        os.makedirs(save_dir, exist_ok=True)
        saved_paths = []
        
        for i, mesh in enumerate(meshes):
            if mesh is not None:
                if isinstance(mesh, list):
                    for j, m in enumerate(mesh):
                        save_path = os.path.join(save_dir, f"{prefix}_{i:03d}_{j:03d}.ply")
                        try:
                            m.export(save_path)
                            saved_paths.append(save_path)
                            print(f"Saved mesh: {save_path}")
                        except Exception as e:
                            print(f"Failed to save mesh {i}_{j}: {e}")
                            saved_paths.append(None)
                else:
                    save_path = os.path.join(save_dir, f"{prefix}_{i:03d}.ply")
                    try:
                        mesh.export(save_path)
                        saved_paths.append(save_path)
                        print(f"Saved mesh: {save_path}")
                    except Exception as e:
                        print(f"Failed to save mesh {i}: {e}")
                        saved_paths.append(None)
            else:
                saved_paths.append(None)
                
        return saved_paths

    def save_meshes_with_fixed_denormalization(self, meshes, save_dir, prefix="mesh"):
        """
        Save meshes with fixed denormalization parameters to original scale, in STL format
        Also compute and save frontal area for each mesh
        
        Fixed parameters:
        - scale_factor: 2.340483
        - x_translation: 1.531090  
        - z_min: -0.3 (lowest point on z-axis)
        
        Args:
            meshes: list of trimesh objects
            save_dir: directory to save meshes
            prefix: filename prefix (can be string or list of strings for each mesh)
            
        Returns:
            saved_paths: list of saved file paths
        """
        import os
        import numpy as np
        
        os.makedirs(save_dir, exist_ok=True)
        saved_paths = []
        
        scale_factor = 2.340483
        x_translation = 1.531090
        z_min_target = -0.318497
        
        if isinstance(prefix, list):
            prefixes = prefix[:len(meshes)]
            while len(prefixes) < len(meshes):
                prefixes.append("mesh")
        else:
            prefixes = [prefix] * len(meshes)
        
        for i, mesh in enumerate(meshes):
            current_prefix = prefixes[i]
            if mesh is not None:
                try:
                    denormalized_mesh = mesh.copy()
                    
                    denormalized_mesh.apply_scale(scale_factor)
                    
                    x_transform = np.array([x_translation, 0.0, 0.0])
                    denormalized_mesh.apply_translation(x_transform)
                    
                    current_z_min = denormalized_mesh.bounds[0][2]
                    z_adjustment = z_min_target - current_z_min
                    z_transform = np.array([0.0, 0.0, z_adjustment])
                    denormalized_mesh.apply_translation(z_transform)
                    
                    final_bounds = denormalized_mesh.bounds
                    
                    try:
                        if hasattr(denormalized_mesh, 'is_winding_consistent'):
                            winding_consistent = denormalized_mesh.is_winding_consistent
                        else:
                            winding_consistent = False
                        
                        # Always fix normals to ensure outward pointing
                        denormalized_mesh.fix_normals()
                        
                        if denormalized_mesh.is_watertight:
                            try:
                                centroid = denormalized_mesh.centroid
                                
                                # Sample a few face normals and check their direction relative to centroid
                                face_centers = denormalized_mesh.triangles_center
                                face_normals = denormalized_mesh.face_normals
                                
                                centroid_to_face = face_centers - centroid
                                
                                # Dot product should be positive if normals point outward
                                dot_products = np.sum(face_normals * centroid_to_face, axis=1)
                                outward_facing = np.sum(dot_products > 0)
                                inward_facing = np.sum(dot_products < 0)
                                
                                if inward_facing > outward_facing:
                                    denormalized_mesh.faces = np.fliplr(denormalized_mesh.faces)
                                    
                                    denormalized_mesh.fix_normals()
                                    
                                    face_normals_new = denormalized_mesh.face_normals
                                    dot_products_new = np.sum(face_normals_new * centroid_to_face, axis=1)
                                    outward_new = np.sum(dot_products_new > 0)
                                    inward_new = np.sum(dot_products_new < 0)
                                
                            except Exception as normal_check_error:
                                print(f"    Advanced normal checking failed: {normal_check_error}")
                        
                        if hasattr(denormalized_mesh, 'is_winding_consistent'):
                            final_winding = denormalized_mesh.is_winding_consistent
                            
                    except Exception as normal_fix_error:
                        print(f"    Normal fixing failed: {normal_fix_error}")
                        print(f"    Continuing with original mesh normals")
                                        

                    is_watertight = denormalized_mesh.is_watertight
                    
                    try:
                        if hasattr(denormalized_mesh, 'edges_boundary'):
                            boundary_edges = denormalized_mesh.edges_boundary
                            is_closed = len(boundary_edges) == 0
                        elif hasattr(denormalized_mesh, 'outline'):
                            outline = denormalized_mesh.outline()
                            is_closed = len(outline) == 0
                        else:
                            is_closed = is_watertight
                    except:
                        is_closed = is_watertight
                    
                    # Check for singly connected (genus 0) - Euler characteristic should be 2
                    V = len(denormalized_mesh.vertices)
                    F = len(denormalized_mesh.faces)
                    
                    try:
                        if hasattr(denormalized_mesh, 'edges'):
                            E = len(denormalized_mesh.edges)
                        elif hasattr(denormalized_mesh, 'edges_unique'):
                            E = len(denormalized_mesh.edges_unique)
                        else:
                            E = int(3 * F / 2)
                    except:
                        E = int(3 * F / 2)
                    
                    euler_char = V - E + F
                    is_singly_connected = (euler_char == 2) and is_closed
                    genus = (2 - euler_char) // 2 if is_closed else "N/A (not closed)"
                    
                    # Try to fix mesh if not watertight or not singly connected
                    if not is_watertight or not is_singly_connected:
                        try:
                            denormalized_mesh.fill_holes()
                            
                            denormalized_mesh.remove_duplicate_faces()
                            denormalized_mesh.remove_degenerate_faces()
                            
                            denormalized_mesh.fix_normals()
                            
                            if denormalized_mesh.is_watertight:
                                try:
                                    centroid = denormalized_mesh.centroid
                                    face_centers = denormalized_mesh.triangles_center
                                    face_normals = denormalized_mesh.face_normals
                                    centroid_to_face = face_centers - centroid
                                    dot_products = np.sum(face_normals * centroid_to_face, axis=1)
                                    outward_facing = np.sum(dot_products > 0)
                                    inward_facing = np.sum(dot_products < 0)
                                    
                                    if inward_facing > outward_facing:
                                        denormalized_mesh.faces = np.fliplr(denormalized_mesh.faces)
                                        denormalized_mesh.fix_normals()
                                except Exception as repair_normal_error:
                                    print(f"      Post-repair normal fixing failed: {repair_normal_error}")
                            
                            V_after = len(denormalized_mesh.vertices)
                            F_after = len(denormalized_mesh.faces)
                            
                            try:
                                if hasattr(denormalized_mesh, 'edges'):
                                    E_after = len(denormalized_mesh.edges)
                                elif hasattr(denormalized_mesh, 'edges_unique'):
                                    E_after = len(denormalized_mesh.edges_unique)
                                else:
                                    E_after = int(3 * F_after / 2)
                            except:
                                E_after = int(3 * F_after / 2)
                            
                            euler_char_after = V_after - E_after + F_after
                            is_watertight_after = denormalized_mesh.is_watertight
                            
                            try:
                                if hasattr(denormalized_mesh, 'edges_boundary'):
                                    boundary_edges = denormalized_mesh.edges_boundary
                                    is_closed_after = len(boundary_edges) == 0
                                elif hasattr(denormalized_mesh, 'outline'):
                                    outline = denormalized_mesh.outline()
                                    is_closed_after = len(outline) == 0
                                else:
                                    is_closed_after = is_watertight_after
                            except:
                                is_closed_after = is_watertight_after
                            
                            is_singly_connected_after = (euler_char_after == 2) and is_closed_after
                            genus_after = (2 - euler_char_after) // 2 if is_closed_after else "N/A"
                            
                        except Exception as fix_error:
                            print(f"    Error fixing mesh: {fix_error}")
                    
                    try:
                        if not hasattr(denormalized_mesh, 'face_normals') or len(denormalized_mesh.face_normals) == 0:
                            denormalized_mesh.fix_normals()
                        
                        if denormalized_mesh.is_watertight:
                            try:
                                centroid = denormalized_mesh.centroid
                                face_centers = denormalized_mesh.triangles_center
                                face_normals = denormalized_mesh.face_normals
                                
                                n_samples = min(100, len(face_centers))
                                sample_indices = np.random.choice(len(face_centers), n_samples, replace=False)
                                
                                sample_centers = face_centers[sample_indices]
                                sample_normals = face_normals[sample_indices]
                                centroid_to_sample = sample_centers - centroid
                                
                                dot_products = np.sum(sample_normals * centroid_to_sample, axis=1)
                                outward_count = np.sum(dot_products > 0)
                                inward_count = np.sum(dot_products <= 0)
                                
                                outward_ratio = outward_count / n_samples
                                
                                if outward_ratio < 0.7:
                                    denormalized_mesh.faces = np.fliplr(denormalized_mesh.faces)
                                    denormalized_mesh.fix_normals()
                                    
                            except Exception as final_check_error:
                                print(f"    Final normal check failed: {final_check_error}")
                        
                        # Always ensure normals are properly computed
                        denormalized_mesh.fix_normals()
                        
                    except Exception as final_normal_error:
                        print(f"    Final normal verification failed: {final_normal_error}")
                    
                    save_path = os.path.join(save_dir, f"{current_prefix}_denorm.stl")
                    denormalized_mesh.export(save_path)
                    saved_paths.append(save_path)
                    print(f"Saved denormalized mesh: {save_path}")
                    
                    # Calculate frontal area for denormalized mesh (if available)
                    try:
                        frontal_area = self.find_Aref(denormalized_mesh, ray_per_dim=256) if hasattr(self, 'find_Aref') else None
                    except:
                        frontal_area = None
                    
                    ply_path = os.path.join(save_dir, f"{current_prefix}_denorm.ply")
                    denormalized_mesh.export(ply_path)
                    
                except Exception as e:
                    import traceback
                    print(f"Traceback: {traceback.format_exc()}")
                    save_path = os.path.join(save_dir, f"{current_prefix}_{i:03d}_fallback.ply")
                    try:
                        mesh.export(save_path)
                        saved_paths.append(save_path)
                        print(f"Saved fallback mesh: {save_path}")
                    except Exception as e2:
                        print(f"Failed to save fallback mesh {i}: {e2}")
                        saved_paths.append(None)
            else:
                print(f"Skipping mesh {i}: mesh is None")
                saved_paths.append(None)
                
        return saved_paths

    def test_step(self, batch, batch_idx):
        """
        Test step for mesh generation using pretrained VAE and diffusion models.
        Can be called from main.py using test config file.
        Includes fixed denormalization with specified parameters.
        """
        
        try:
            batch_size = 4
            device = next(self.model.parameters()).device
            num_points = 32768
            dummy_coarse = batch['coarse_surface']
            dummy_sharp = batch['sharp_surface']

            self.first_stage_model.split = 'val'
            _, sample_latents, _ = self.first_stage_model.encode(
                coarse_surface=dummy_coarse.half(),
                sharp_surface=dummy_sharp.half(),
                sample_posterior=False
            )
            sample_latents = sample_latents.float()  # Ensure latents are float32
            latent_shape = sample_latents.shape[1:]
            
            # Set the random seed for reproducible noise generation
            torch.manual_seed(batch_idx)
            noise_shape = (batch_size,) + latent_shape
            noise = torch.randn(noise_shape, device=device, dtype=torch.float32)
            
            contexts = {
                'main': torch.zeros(batch_size, 1, 768, device=device, dtype=torch.float32)
            }
            
            sample_batch = {}
            sample_batch['batch_idx'] = batch_idx
            
            # Add uid information to sample_batch if available
            if 'uid' in batch:
                raw_uid = batch['uid']
                # Extract basename and remove extension for clean uid
                if isinstance(raw_uid, list) and len(raw_uid) > 0:
                    raw_uid = raw_uid[0]
                elif isinstance(raw_uid, str):
                    pass
                else:
                    try:
                        if hasattr(raw_uid, 'tolist'):
                            raw_uid_list = raw_uid.tolist()
                            raw_uid = raw_uid_list[0] if len(raw_uid_list) > 0 else str(raw_uid)
                        else:
                            raw_uid = str(raw_uid)
                    except:
                        raw_uid = f"batch_{batch_idx}"
                
                raw_uid_str = str(raw_uid).strip("[]'\"")
                
                uid = os.path.basename(raw_uid_str)
                for ext in ['.npz', '.obj', '.stl', '.ply']:
                    if uid.endswith(ext):
                        uid = uid[:-len(ext)]
                        break
            else:
                # Generate deterministic uid based on batch_idx for reproducibility
                uid = f"batch_{batch_idx:04d}_reproducible"

            sample_batch['uid'] = uid

            # Add sample_latents to sample_batch to use as initial latents in pipeline
            # Ensure sample_latents matches batch_size for consistency
            sample_batch['initial_latents'] = sample_latents
            
            # Add drag prediction if available (only for original meshes)  
            uids = []
            if 'uid' in batch:
                uid_data = batch['uid']
                if isinstance(uid_data, list):
                    uids = uid_data[:len(sample_latents)]
                elif isinstance(uid_data, str):
                    uids = [uid_data]
                else:
                    if hasattr(uid_data, 'tolist'):
                        uids = uid_data.tolist()[:len(sample_latents)]
                    else:
                        uids = [str(uid_data)]
            
            # Ensure we have the right number of UIDs
            while len(uids) < len(sample_latents):
                uids.append(f"missing_{batch_idx}_{len(uids)}")
            uids = uids[:len(sample_latents)]
            
            uid_prefixes = []
            for uid in uids:
                if uid:
                    clean_uid = str(uid)
                    
                    # Remove brackets and quotes if present (in case uid is a string representation of a list)
                    clean_uid = clean_uid.strip("[]'\"")
                    
                    # Extract basename and remove extension for prefix
                    clean_uid = os.path.basename(clean_uid)
                    
                    for ext in ['.npz', '.obj', '.stl', '.ply']:
                        if clean_uid.endswith(ext):
                            clean_uid = clean_uid[:-len(ext)]
                            break
                    
                    uid_prefixes.append(clean_uid)
                else:
                    uid_prefixes.append(f"unknown_{batch_idx}")
            
            # Get parameters from test_config for consistent directory naming
            test_save_dir = getattr(self.test_config, 'save_dir', None) if self.test_config else None
            drag_guidance_scale = getattr(self.test_config, 'drag_guidance_scale', None) if self.test_config else None
            start_time_step = getattr(self.test_config, 'start_time_step', None) if self.test_config else None
            num_inference_steps = getattr(self.test_config, 'num_inference_steps', None) if self.test_config else None
            num_optimization_loops = getattr(self.test_config, 'num_optimization_loops', None) if self.test_config else None
            num_optimization_steps = getattr(self.test_config, 'num_optimization_steps', None) if self.test_config else None
            learning_rate = getattr(self.test_config, 'learning_rate', None) if self.test_config else None
            
            save_dir = self.get_mesh_save_dir(
                "test_meshes",
                test_save_dir=test_save_dir,
                drag_guidance_scale=drag_guidance_scale,
                start_time_step=start_time_step,
                num_inference_steps=num_inference_steps,
                num_optimization_loops=num_optimization_loops,
                num_optimization_steps=num_optimization_steps,
                learning_rate=learning_rate
            )
            
            if self.drag_enabled and self.drag_decoder is not None:
                try:
                    original_prefixes_drag = [f"{prefix}_original_drag" for prefix in uid_prefixes[:len(sample_latents)]]
                    meshes_with_drag, drag_values = self._add_drag_to_meshes(
                        sample_latents, sample_latents, save_dir, original_prefixes_drag, return_drag=True
                    )
                    sample_batch['initial_drag_values'] = drag_values
                    print("Initial drag coefficient:", drag_values)
                    if meshes_with_drag:
                        drag_saves = len([p for p in meshes_with_drag if p and p[0] is not None])
                    else:
                        print("Drag prediction failed")
                except Exception as drag_error:
                    print(f"Drag prediction failed: {drag_error}")

            self.model = self.model.float()
            generated_latents = None
            
            try:
                outputs = self.sample(sample_batch, output_type='latent')
                if outputs is not None:
                    if hasattr(outputs, 'shape'):
                        generated_latents = outputs[:batch_size]
                    elif isinstance(outputs, (list, tuple)) and len(outputs) > 0:
                        # outputs is a list, try to get the first element
                        first_output = outputs[0]
                        if first_output is not None and hasattr(first_output, 'shape'):
                            generated_latents = first_output[:batch_size]
                        elif first_output is not None:
                            generated_latents = first_output
                        else:
                            print("Warning: First output is None")
                    else:
                        print(f"Warning: Unexpected outputs format: {type(outputs)}")
                else:
                    print("Warning: sample() returned None")
            except Exception as e:
                print(f"Error during sampling: {e}")
                print(f"Traceback: {traceback.format_exc()}")
            
            try:
                # Ensure sample_latents matches batch_size for consistency
                sample_latents_expanded = sample_latents
                
                original_meshes = self.extract_geometry_by_diffdmc(
                    sample_latents_expanded,
                    resolution=getattr(self.test_config, 'resolution', 128) if self.test_config else 128,
                    batch_size=10000,
                    verbose=False
                )
                
                valid_original_count = 0
                for i, mesh in enumerate(original_meshes):
                    if mesh is not None:
                        valid_original_count += 1
                    else:
                        print(f"  Original mesh {i}: None (generation failed)")
                
                if valid_original_count == 0:
                    try:
                        original_meshes_fallback = self.decode_latents_to_mesh(
                            sample_latents_expanded, 
                            resolution=getattr(self.test_config, 'resolution', 64) if self.test_config else 64,
                            batch_size=5000,
                            verbose=False
                        )
                        
                        fallback_valid_count = sum(1 for m in original_meshes_fallback if m is not None)
                        if fallback_valid_count > valid_original_count:
                            original_meshes = original_meshes_fallback
                            valid_original_count = fallback_valid_count
                        else:
                            print(f"Fallback method only generated {fallback_valid_count} original meshes, keeping original results")
                            
                    except Exception as fallback_error:
                        print(f"Fallback original mesh generation also failed: {fallback_error}")
                
            except Exception as e:
                print(f"Failed to generate original meshes: {e}")
                print(f"Traceback: {traceback.format_exc()}")
                original_meshes = [None] * batch_size
            
            try:
                generated_meshes = self.extract_geometry_by_diffdmc(
                    generated_latents, 
                    resolution=getattr(self.test_config, 'resolution', 128) if self.test_config else 128,
                    batch_size=10000,
                    verbose=False
                )
                
                valid_generated_count = 0
                for i, mesh in enumerate(generated_meshes):
                    if mesh is not None:
                        valid_generated_count += 1
                    else:
                        print(f"  Generated mesh {i}: None (generation failed)")
                
                if valid_generated_count == 0:
                    try:
                        generated_meshes_fallback = self.decode_latents_to_mesh(
                            generated_latents, 
                            resolution=getattr(self.test_config, 'resolution', 64) if self.test_config else 64,
                            batch_size=5000,
                            verbose=False
                        )
                        
                        fallback_valid_count = sum(1 for m in generated_meshes_fallback if m is not None)
                        if fallback_valid_count > valid_generated_count:
                            generated_meshes = generated_meshes_fallback
                            valid_generated_count = fallback_valid_count
                        else:
                            print(f"Fallback method only generated {fallback_valid_count} meshes, keeping original results")
                            
                    except Exception as fallback_error:
                        print(f"Fallback mesh generation also failed: {fallback_error}")
                
            except Exception as e:
                print(f"Failed to generate meshes from generated_latents: {e}")
                print(f"Traceback: {traceback.format_exc()}")
                generated_meshes = [None] * batch_size
                
                try:
                    generated_meshes = self.decode_latents_to_mesh(
                        generated_latents, 
                        resolution=64,
                        batch_size=2000,
                        verbose=False
                    )
                    fallback_count = sum(1 for m in generated_meshes if m is not None)
                except Exception as emergency_error:
                    generated_meshes = [None] * batch_size
            
            if (generated_meshes and any(mesh is not None for mesh in generated_meshes)) or (original_meshes and any(mesh is not None for mesh in original_meshes)):
                # Use generated_meshes for UID extraction (as it's the primary output)
                meshes_for_uid = generated_meshes if generated_meshes else original_meshes
                
                if original_meshes and any(mesh is not None for mesh in original_meshes):
                    try:
                        original_prefixes = [f"{prefix}_original" for prefix in uid_prefixes[:len(original_meshes)]]
                        original_denorm_saved_paths = self.save_meshes_with_fixed_denormalization(
                            original_meshes, save_dir, original_prefixes[0]
                        )
                        successful_original_saves = len([p for p in original_denorm_saved_paths if p])
                    except Exception as original_error:
                        print(f"Original mesh saving failed: {original_error}")
                        print(f"Traceback: {traceback.format_exc()}")
                        successful_original_saves = 0
                else:
                    successful_original_saves = 0
                    print("No original meshes to save")
                
                # Save generated meshes (from generated_latents)
                if generated_meshes and any(mesh is not None for mesh in generated_meshes):
                    try:
                        generated_prefixes = [f"{prefix}_generated" for prefix in uid_prefixes[:len(generated_meshes)]]
                        generated_denorm_saved_paths = self.save_meshes_with_fixed_denormalization(
                            generated_meshes, save_dir, generated_prefixes[0]
                        )
                        successful_generated_saves = len([p for p in generated_denorm_saved_paths if p])
                    except Exception as generated_error:
                        print(f"Generated mesh saving failed: {generated_error}")
                        print(f"Traceback: {traceback.format_exc()}")
                        try:
                            generated_prefixes = [f"{prefix}_generated" for prefix in uid_prefixes[:len(generated_meshes)]]
                            saved_paths = self.save_meshes(generated_meshes, save_dir, generated_prefixes[0])
                            successful_generated_saves = len([p for p in saved_paths if p])
                        except Exception as fallback_error:
                            print(f"Fallback mesh saving also failed: {fallback_error}")
                            successful_generated_saves = 0
                else:
                    successful_generated_saves = 0
                    print("No generated meshes to save")
                
                try:
                    if generated_meshes:
                        regular_generated_prefixes = [f"{prefix}_generated_regular" for prefix in uid_prefixes[:len(generated_meshes)]]
                        regular_generated_saved_paths = self.save_meshes(generated_meshes, save_dir, regular_generated_prefixes[0])
                        successful_regular_generated_saves = len([p for p in regular_generated_saved_paths if p])
                    else:
                        successful_regular_generated_saves = 0
                        
                    if original_meshes:
                        regular_original_prefixes = [f"{prefix}_original_regular" for prefix in uid_prefixes[:len(original_meshes)]]
                        regular_original_saved_paths = self.save_meshes(original_meshes, save_dir, regular_original_prefixes[0])
                        successful_regular_original_saves = len([p for p in regular_original_saved_paths if p])
                    else:
                        successful_regular_original_saves = 0
                        
                except Exception as regular_error:
                    print(f"Regular mesh saving failed: {regular_error}")
                    successful_regular_generated_saves = 0
                    successful_regular_original_saves = 0
                
                # Add physics prediction if available (only for generated meshes)
                if self.physics_enabled and self.physics_decoder is not None and generated_meshes:
                    try:
                        generated_prefixes_physics = [f"{prefix}_generated_physics" for prefix in uid_prefixes[:len(generated_meshes)]]
                        meshes_with_physics = self._add_physics_to_meshes(
                            generated_meshes, generated_latents, dummy_coarse, dummy_sharp, save_dir, generated_prefixes_physics
                        )
                        if meshes_with_physics:
                            physics_saves = len([p for p in meshes_with_physics if p])
                    except Exception as physics_error:
                        print(f"Physics prediction failed: {physics_error}")
                
                # Add drag prediction if available (only for generated meshes)  
                if self.drag_enabled and self.drag_decoder is not None and generated_meshes:
                    try:
                        generated_prefixes_drag = [f"{prefix}_generated_drag" for prefix in uid_prefixes[:len(generated_meshes)]]
                        meshes_with_drag, drag_values = self._add_drag_to_meshes(
                            generated_meshes, generated_latents, save_dir, generated_prefixes_drag, return_drag=True
                        )
                        print("Optimized drag coefficient", drag_values)
                        if meshes_with_drag:
                            drag_saves = len([p for p in meshes_with_drag if p and p[0] is not None])
                        else:
                            print("Drag prediction failed")
                    except Exception as drag_error:
                        print(f"Drag prediction failed: {drag_error}")
                
                total_successful_saves = successful_original_saves + successful_generated_saves
                self.log("test/meshes_generated", total_successful_saves, prog_bar=True, logger=True)
                self.log("test/original_meshes", successful_original_saves, prog_bar=True, logger=True)
                self.log("test/generated_meshes", successful_generated_saves, prog_bar=True, logger=True)
                
                print("Test step completed successfully!")
                return {
                    "test_meshes_generated": total_successful_saves,
                    "original_meshes_saved": successful_original_saves,
                    "generated_meshes_saved": successful_generated_saves,
                    "save_dir": save_dir
                }
            else:
                print("Error: No valid meshes generated (neither original nor generated)")
                self.log("test/meshes_generated", 0, prog_bar=True, logger=True)
                self.log("test/original_meshes", 0, prog_bar=True, logger=True)
                self.log("test/generated_meshes", 0, prog_bar=True, logger=True)
                return {"test_meshes_generated": 0, "original_meshes_saved": 0, "generated_meshes_saved": 0}
                
        except Exception as e:
            print(f"Test step failed: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            self.log("test/meshes_generated", 0, prog_bar=True, logger=True)
            return {"error": str(e)}
