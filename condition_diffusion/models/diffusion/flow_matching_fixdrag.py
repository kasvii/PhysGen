import os,sys
import traceback
from contextlib import contextmanager
from typing import List, Tuple, Optional, Union

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info
from pytorch_lightning.utilities import rank_zero_only

from ...utils.ema import LitEma
from ...utils.misc import instantiate_from_config, instantiate_non_trainable_model

import numpy as np
try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False
    print("Warning: trimesh not available, Chamfer distance calculation will be disabled")

try:
    import pyvista as pv
    PYVISTA_AVAILABLE = True
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
        cond_stage_config=None,
        cond_stage_key: str = "condition_image",
        scale_by_std: bool = False,
        z_scale_factor: float = 1.0,
        ckpt_path: Optional[str] = None,
        ignore_keys: Union[Tuple[str], List[str]] = (),
        torch_compile: bool = False,
        overfitting_debug: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.first_stage_key = first_stage_key
        self.cond_stage_key = cond_stage_key
        self.overfitting_debug = overfitting_debug
        self.physics_decoder_config = physics_decoder_config
        self.drag_decoder_config = drag_decoder_config
        self.noisy_drag_decoder_config = noisy_drag_decoder_config
        self.test_config = test_config
        self.test_step_outputs = []
        
        
        self.cond_stage_model = None
        if cond_stage_config is not None:
            self.cond_stage_model = instantiate_from_config(cond_stage_config)
            self.cond_stage_model.eval()
            self.cond_stage_model.requires_grad_(False)
        

        self.optimizer_cfg = optimizer_cfg


        self.scheduler_cfg = scheduler_cfg
        self.sampler = None
        if 'transport' in scheduler_cfg:
            self.transport = instantiate_from_config(scheduler_cfg.transport)
            self.sampler = instantiate_from_config(scheduler_cfg.sampler, transport=self.transport)
            self.sample_fn = self.sampler.sample_ode(**scheduler_cfg.sampler.ode_params)

        self.denoiser_cfg = denoiser_cfg
        self.model = instantiate_from_config(denoiser_cfg, device=None, dtype=torch.float32)


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

            if ckpt_path is not None:
                self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)


        self.first_stage_model = instantiate_non_trainable_model(first_stage_config)

        

        self.physics_decoder = None
        self.physics_enabled = False
        try:
            self._init_physics_decoder()
        except Exception as e:
            print(f"Warning: Failed to initialize physics decoder: {e}")

        self.drag_decoder = None
        self.drag_enabled = False
        try:
            self._init_drag_decoder()
        except Exception as e:
            print(f"Warning: Failed to initialize drag decoder: {e}")


        self.noisy_drag_decoder = None
        self.noisy_drag_enabled = False
        try:
            self._init_noisy_drag_decoder()
        except Exception as e:
            print(f"Warning: Failed to initialize noisy drag decoder: {e}")

        self.scale_by_std = scale_by_std
        if scale_by_std:
            self.register_buffer("z_scale_factor", torch.tensor(z_scale_factor))
        else:
            self.z_scale_factor = z_scale_factor


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
            drag_estimator=self.noisy_drag_decoder if self.test_config.use_drag_guidance else None,
            phys_decoder=self.physics_decoder if self.test_config.use_drag_guidance else None,
            conditioner=self.cond_stage_model,
        )


        self.torch_compile = torch_compile
        if self.torch_compile:
            torch.nn.Module.compile(self.model)
            torch.nn.Module.compile(self.first_stage_model)

        self.test_results = []

    def _init_physics_decoder(self):
        """Initialize physics decoder for pressure prediction"""
        try:
            if self.physics_decoder_config is None:
                self.physics_decoder = None
                self.physics_enabled = False
                return
                
            if not self.physics_decoder_config.get('enabled', False):
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
            


                




                

            self.physics_decoder.train()


            for param in self.physics_decoder.parameters():
                param.requires_grad_(True)
            


            main_model_dtype = next(self.model.parameters()).dtype
            

            self.physics_decoder = self.physics_decoder.to(dtype=main_model_dtype)
            

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
        import traceback
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
                print(f"Warning: Drag checkpoint not found at {drag_ckpt_path}")
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
                print(f"Warning: Noisy drag checkpoint not found at {noisy_drag_ckpt_path}")
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
            


                




                

            self.noisy_drag_decoder.train()
            

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
            
            context = None
            if self.cond_stage_model is not None and self.cond_stage_key in batch:
                condition_data = batch[self.cond_stage_key]
                context = self.cond_stage_model(condition_data).float()
            else:
                if self.cond_stage_model is not None:
                    batch_size = coarse_surface.shape[0]
                    context = self.cond_stage_model.unconditional_embedding(batch_size)
                
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
            if context is not None:
                contexts = {'main': context}
                loss = self.transport.training_losses(
                    lambda x, t: self.model(x, t, contexts), 
                    latents.float()
                )["loss"].mean()
            else:
                batch_size = latents.shape[0]
                device = latents.device
                dtype = latents.dtype
                text_len = getattr(self.model, 'text_len', 256)
                context_dim = getattr(self.model, 'context_dim', 768)
                dummy_context = torch.zeros(batch_size, text_len, context_dim, device=device, dtype=dtype)
                contexts = {'main': dummy_context}
                loss = self.transport.training_losses(
                    lambda x, t: self.model(x, t, contexts), 
                    latents.float()
                )["loss"].mean()
            torch.set_rng_state(current_seed)
        else:
            if context is not None:
                contexts = {'main': context}
                loss = self.transport.training_losses(
                    lambda x, t: self.model(x, t, contexts), 
                    latents.float()
                )["loss"].mean()
            else:
                batch_size = latents.shape[0]
                device = latents.device
                dtype = latents.dtype
                text_len = getattr(self.model, 'text_len', 256)
                context_dim = getattr(self.model, 'context_dim', 768)
                dummy_context = torch.zeros(batch_size, text_len, context_dim, device=device, dtype=dtype)
                contexts = {'main': dummy_context}
                loss = self.transport.training_losses(
                    lambda x, t: self.model(x, t, contexts), 
                    latents.float()
                )["loss"].mean()
            
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

        if batch_idx < 3:
            try:
                
                batch_size = min(2, batch["coarse_surface"].shape[0])
                
                sample_batch = {}
                if self.cond_stage_key in batch:
                    sample_batch[self.cond_stage_key] = batch[self.cond_stage_key][:batch_size]
                

                if 'drag_coefficient' in batch:
                    sample_batch['drag_coefficient'] = batch['drag_coefficient'][:batch_size]
                

                generated_latents = self.sample(sample_batch, output_type='latent')
                if generated_latents is None or not hasattr(generated_latents, 'shape'):
                    raise ValueError("Pipeline returned None or invalid latents")
                
                if generated_latents is not None:
                    meshes = self.decode_latents_to_mesh(
                        generated_latents, 
                        resolution=64,
                        batch_size=10000,
                        verbose=True
                    )
                    
                    if meshes and any(mesh is not None for mesh in meshes):
                        save_dir = self.get_mesh_save_dir("val_meshes")
                        saved_paths = self.save_meshes(meshes, save_dir, f"val_batch{batch_idx}")
                        successful_saves = len([p for p in saved_paths if p])
                        

                        if self.drag_enabled and self.drag_decoder is not None:
                            try:
                                meshes_with_drag = self._add_drag_to_meshes(
                                    meshes, generated_latents, save_dir, f"val_batch{batch_idx}_drag"
                                )
                                if meshes_with_drag:
                                    drag_saves = len([p for p in meshes_with_drag if p and p[0] is not None])
                                else:
                                    print("Drag prediction failed for validation")
                            except Exception as drag_error:
                                print(f"Drag prediction failed for validation: {drag_error}")
                        
                        if self.cond_stage_key in sample_batch and sample_batch[self.cond_stage_key] is not None:
                            condition_images = sample_batch[self.cond_stage_key]
                            saved_condition_paths = self.save_condition_images(
                                condition_images, save_dir, "condition", batch_idx
                            )
                            condition_saves = len([p for p in saved_condition_paths if p])
                        
                        self.log("val/meshes_generated", successful_saves, prog_bar=False, logger=True)
                    else:
                        self.log("val/meshes_generated", 0, prog_bar=False, logger=True)
                else:
                    self.log("val/meshes_generated", 0, prog_bar=False, logger=True)
                            
            except Exception as e:
                import traceback
                print(f"Mesh generation failed in validation: {e}")
                print(f"Traceback: {traceback.format_exc()}")
                self.log("val/meshes_generated", 0, prog_bar=False, logger=True)

        return loss

    def sample(self, batch, output_type='latent', **kwargs):
        generator = torch.Generator().manual_seed(0)

        with self.ema_scope("Sample"):
            with torch.amp.autocast(device_type='cuda'):
                try:
                    self.pipeline.device = self.device
                    self.pipeline.dtype = self.dtype
                    additional_params = {'output_type': output_type}

                    if self.cond_stage_key in batch and batch[self.cond_stage_key] is not None:
                        condition_data = batch[self.cond_stage_key]
                        
                        uid = None
                        if 'uid' in batch:
                            uid_data = batch['uid']
                            if isinstance(uid_data, list) and len(uid_data) > 0:
                                uid = uid_data[0]
                            elif isinstance(uid_data, str):
                                uid = uid_data
                            else:
                                try:
                                    if hasattr(uid_data, 'tolist'):
                                        uid_list = uid_data.tolist()
                                        uid = uid_list[0] if uid_list else None
                                    else:
                                        uid = str(uid_data)
                                except:
                                    uid = None
                        
                        if uid is None:
                            from datetime import datetime
                            import pytz
                            zurich_tz = pytz.timezone('Europe/Zurich')
                            current_time = datetime.now(zurich_tz)
                            uid = f"sample_{current_time.strftime('%Y%m%d_%H%M%S_%f')[:19]}"
                        

                        additional_params[self.cond_stage_key] = condition_data
                        
                        tgt_drag_value = self.test_config.target_drag
                        if 'drag_coefficient' in batch:
                            drag_coeff = batch['drag_coefficient']
                            if hasattr(drag_coeff, 'item'):
                                tgt_drag_value = drag_coeff.item()
                            elif isinstance(drag_coeff, (list, tuple)) and len(drag_coeff) > 0:
                                tgt_drag_value = float(drag_coeff[0])
                            elif isinstance(drag_coeff, (int, float)):
                                tgt_drag_value = float(drag_coeff)
                            else:
                                try:
                                    tgt_drag_value = float(drag_coeff)
                                except:
                                    tgt_drag_value = self.test_config.target_drag
                                    
                        print("Target drag coefficient:", tgt_drag_value)
                        
                        outputs = self.pipeline(
                            generator=generator, 
                            drag_guidance_scale=self.test_config.drag_guidance_scale, 
                            tgt_drag=tgt_drag_value,
                            init_drag=None,
                            start_time_step=self.test_config.start_time_step, 
                            num_inference_steps=self.test_config.num_inference_steps,
                            num_optimization_loops=getattr(self.test_config, 'num_optimization_loops', 100),
                            num_optimization_steps=getattr(self.test_config, 'num_optimization_steps', 10),
                            learning_rate=getattr(self.test_config, 'learning_rate', 0.01),
                            angle_threshold=getattr(self.test_config, 'angle_threshold', 10.0),
                            scale_factor=self.z_scale_factor, 
                            test_save_dir=getattr(self.test_config, 'save_dir', 'save'),
                            uid=uid,
                            **additional_params
                        )
                    else:
                        uid = None
                        if 'uid' in batch:
                            uid_data = batch['uid']
                            if isinstance(uid_data, list) and len(uid_data) > 0:
                                uid = uid_data[0]
                            elif isinstance(uid_data, str):
                                uid = uid_data
                            else:
                                try:
                                    if hasattr(uid_data, 'tolist'):
                                        uid_list = uid_data.tolist()
                                        uid = uid_list[0] if uid_list else None
                                    else:
                                        uid = str(uid_data)
                                except:
                                    uid = None
                        
                        if uid is None:
                            from datetime import datetime
                            import pytz
                            zurich_tz = pytz.timezone('Europe/Zurich')
                            current_time = datetime.now(zurich_tz)
                            uid = f"unconditional_{current_time.strftime('%Y%m%d_%H%M%S_%f')[:23]}"
                        
                        tgt_drag_value = self.test_config.target_drag
                        if 'drag_coefficient' in batch:
                            drag_coeff = batch['drag_coefficient']
                            if hasattr(drag_coeff, 'item'):
                                tgt_drag_value = drag_coeff.item()
                            elif isinstance(drag_coeff, (list, tuple)) and len(drag_coeff) > 0:
                                tgt_drag_value = float(drag_coeff[0])
                            elif isinstance(drag_coeff, (int, float)):
                                tgt_drag_value = float(drag_coeff)
                            else:
                                try:
                                    tgt_drag_value = float(drag_coeff)
                                except:
                                    tgt_drag_value = self.test_config.target_drag
                        
                        outputs = self.pipeline(
                            generator=generator, 
                            drag_guidance_scale=self.test_config.drag_guidance_scale, 
                            tgt_drag=0,
                            start_time_step=self.test_config.start_time_step, 
                            num_inference_steps=self.test_config.num_inference_steps,
                            num_optimization_loops=getattr(self.test_config, 'num_optimization_loops', 100),
                            num_optimization_steps=getattr(self.test_config, 'num_optimization_steps', 10),
                            learning_rate=getattr(self.test_config, 'learning_rate', 0.01),
                            angle_threshold=getattr(self.test_config, 'angle_threshold', 10.0),
                            scale_factor=self.z_scale_factor, 
                            test_save_dir=getattr(self.test_config, 'save_dir', 'save'),
                            uid=uid,
                            **additional_params
                        )

                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    with open("error.txt", "a") as f:
                        f.write(str(e))
                        f.write(traceback.format_exc())
                        f.write("\n")
                    outputs = None

        if output_type == 'latent':
            return outputs
        else:
            return [outputs] if outputs is not None else [None]


    def decode_latents_to_mesh(self, latents, resolution=128, batch_size=20000, verbose=True):
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
                    
                    try:
                        single_latent = unscaled_latents[b:b+1]
                        
                        x = torch.linspace(-1, 1, resolution, device=device)
                        y = torch.linspace(-1, 1, resolution, device=device)
                        z = torch.linspace(-1, 1, resolution, device=device)
                        
                        grid = torch.meshgrid(x, y, z, indexing='ij')
                        queries = torch.stack(grid, dim=-1).reshape(-1, 3)
                        

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
                                print(f"   Decode method failed: {decode_error}, using fallback")
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

    def extract_geometry_by_diffdmc(self, latents, resolution=128, batch_size=10000, verbose=True, 
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
            meshes = self.extract_geometry_by_diffdmc(
                latents,
                resolution=64,
                batch_size=10000,
            )
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
            

            self.first_stage_model.eval()
            self.first_stage_model.split = 'val'
            
            device = latents.device
            B = latents.shape[0]
            
            with torch.no_grad():
                unscaled_latents = latents / self.z_scale_factor
                
                unscaled_latents = self.first_stage_model.post_kl(unscaled_latents.half())
                unscaled_latents = self.first_stage_model.transformer(unscaled_latents)
                
                if isinstance(bounds, float):
                    bounds = [-bounds, -bounds, -bounds, bounds, bounds, bounds]
                
                bbox_min = np.array(bounds[0:3])
                bbox_max = np.array(bounds[3:6])
                bbox_size = bbox_max - bbox_min
                
                octree_depth = int(np.log2(resolution)) if resolution > 0 else 8
                
                xyz_samples, grid_size, length, xs, ys, zs = generate_dense_grid_points(
                    bbox_min=bbox_min,
                    bbox_max=bbox_max,
                    octree_depth=octree_depth,
                    indexing="ij"
                )
                xyz_samples = torch.FloatTensor(xyz_samples).to(device)
                
                
                batch_logits = []
                for start in trange(0, xyz_samples.shape[0], batch_size, desc="Querying occupancy", disable=True):
                    queries = xyz_samples[start: start + batch_size, :]
                    batch_queries = repeat(queries, "p c -> b p c", b=B)
                    
                    try:
                        if hasattr(self.first_stage_model, 'decoder') and hasattr(self.first_stage_model.decoder, '__call__'):
                            logits = self.first_stage_model.decoder(batch_queries.half(), unscaled_latents)
                            if logits.dim() > 2:
                                logits = logits.squeeze(-1)
                        else:

                            logits = self._fallback_sdf_query(batch_queries.half(), unscaled_latents).squeeze(-1)
                    except Exception as e:
                        print(f"Query method failed: {e}")
                        logits = self._fallback_sdf_query(batch_queries.half(), unscaled_latents).squeeze(-1)
                    
                    batch_logits.append(logits.float())
                
                grid_logits = torch.cat(batch_logits, dim=1).view((B, grid_size[0], grid_size[1], grid_size[2])).float()
                
                
                if save_slice_dir:
                    self._save_slice_visualization(grid_logits, grid_size, save_slice_dir, verbose)
                
                meshes = []
                diffdmc = DiffDMC(dtype=torch.float32).to(device)
                
                for i in range(B):
                    
                    try:
                        vertices, faces = diffdmc(-grid_logits[i], isovalue=0, normalize=False)
                        

                        vertices = (vertices + 1) / grid_size[0] * bbox_size[0] + bbox_min[0]
                        
                        if vertices.numel() > 0 and faces.numel() > 0:
                            vertices_np = vertices.detach().cpu().numpy().astype(np.float32)
                            faces_np = faces.detach().cpu().numpy().astype(np.int32)
                            
                            mesh = trimesh.Trimesh(vertices=vertices_np, faces=faces_np)
                            
                            mesh.remove_duplicate_faces()
                            mesh.remove_unreferenced_vertices()
                            
                            meshes.append(mesh)
                        else:
                            meshes.append(None)
                                
                    except Exception as e:
                        meshes.append(None)
                        if verbose:
                            print(f"  DiffDMC failed for batch {i}: {e}")
                
                if verbose:
                    successful_meshes = sum(1 for m in meshes if m is not None)
                
                return meshes
                
        except Exception as e:
            import traceback
            print(f"extract_geometry_by_diffdmc failed: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            return [None] * latents.shape[0]

    def _save_slice_visualization(self, grid_logits, grid_size, save_slice_dir, verbose=True):
        """
        Save slice visualization for debugging (adapted from utils.py)
        """
        try:
            import matplotlib.pyplot as plt
            
            slice_grid = grid_logits[0, (grid_size[0]-1)//2].cpu().numpy()
            color_values = np.where(slice_grid > 0, 1, 0)
            
            y_coords = np.arange(0, grid_size[0]-1, 1)
            z_coords = np.arange(0, grid_size[0]-1, 1)
            y_grid, z_grid = np.meshgrid(y_coords, z_coords)
            color_values = color_values[y_grid, z_grid].T
            
            plt.figure(figsize=(8, 8))
            plt.scatter(y_grid, z_grid, s=1, c=color_values, cmap='gray', marker='o')
            plt.gca().set_facecolor((0.6, 0.6, 0.6))
            plt.axis([0, grid_size[0], 0, grid_size[0]])
            plt.gca().invert_xaxis()
            
            plt.savefig(save_slice_dir + '.png', dpi=150, bbox_inches='tight')
            plt.close()
            
                
        except Exception:
            pass

    def _query_vae_occupancy(self, queries, decoded_features):
        """
        Query VAE for occupancy values at given 3D points
        This is a fallback method that needs customization based on your VAE structure
        """


        batch_size, num_queries, _ = queries.shape
        return torch.randn(batch_size, num_queries, 1, device=queries.device) * 0.1

    def _interpolate_decoded_features(self, queries, decoded_features):
        """
        Interpolate decoded features to get occupancy at query points
        This method assumes decoded_features contains spatial information
        """

        batch_size, num_queries, _ = queries.shape
        


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
        

        logits = -sdf * 5.0
        
        return logits

    def _logits_to_mesh(self, logits, resolution, verbose=True, batch_idx=0):
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
            
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                logits = torch.where(torch.isnan(logits) | torch.isinf(logits), torch.zeros_like(logits), logits)
            

            logits = torch.clamp(logits, min=-10.0, max=10.0)
            
            occupancy_grid = logits.reshape(resolution, resolution, resolution)
            occupancy_grid = torch.sigmoid(occupancy_grid)
            
            occupancy_np = occupancy_grid.cpu().numpy()
            
            min_val = occupancy_np.min()
            max_val = occupancy_np.max()
            mean_val = occupancy_np.mean()
            

            try:

                if max_val <= min_val:
                    return None
                

                if min_val >= 0.5:
                    level = min_val + (max_val - min_val) * 0.1
                elif max_val <= 0.5:
                    level = max_val - (max_val - min_val) * 0.1
                else:
                    level = 0.5
                
                vertices, faces, _, _ = measure.marching_cubes(
                    occupancy_np, level=level, spacing=(2.0/resolution, 2.0/resolution, 2.0/resolution)
                )
                
                vertices = vertices - 1.0
                
                if len(vertices) > 0 and len(faces) > 0:
                    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
                    mesh.remove_duplicate_faces()
                    mesh.remove_unreferenced_vertices()
                    return mesh
                else:
                    return None
                    
            except Exception as e:
                if verbose:
                    print(f"   Marching cubes failed for batch {batch_idx}: {e}")
                

                try:
                    if max_val > min_val:
                        occupancy_normalized = (occupancy_np - min_val) / (max_val - min_val)
                        vertices, faces, _, _ = measure.marching_cubes(
                            occupancy_normalized, level=0.5, spacing=(2.0/resolution, 2.0/resolution, 2.0/resolution)
                        )
                        vertices = vertices - 1.0
                        
                        if len(vertices) > 0 and len(faces) > 0:
                            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
                            mesh.remove_duplicate_faces()
                            mesh.remove_unreferenced_vertices()
                            return mesh
                        else:
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

    def get_mesh_save_dir(self, mesh_type="val_meshes", drag_guidance_scale=None, start_time_step=None, 
                         num_inference_steps=None, num_optimization_loops=None, num_optimization_steps=None, learning_rate=None):
        """
        Get the directory for saving meshes, using test config save_dir if available
        
        Args:
            mesh_type: type of mesh ("val_meshes", "test_meshes", etc.)
            drag_guidance_scale: drag guidance scale value
            start_time_step: start time step value
            num_inference_steps: number of inference steps
            num_optimization_loops: number of optimization loops
            num_optimization_steps: number of optimization steps per loop
            
        Returns:
            Path to mesh save directory
        """
        import time
        from pathlib import Path
        
        if mesh_type == "test_meshes" and self.test_config is not None:
            test_save_dir = None
            if hasattr(self.test_config, 'get'):
                test_save_dir = self.test_config.get('save_dir')
            elif hasattr(self.test_config, 'save_dir'):
                test_save_dir = self.test_config.save_dir
            
            if test_save_dir:
                save_dir = Path(test_save_dir)
                
                
                final_save_dir = save_dir / f"{mesh_type}"
                
                os.makedirs(final_save_dir, exist_ok=True)
                print(f"Using test config save_dir: {final_save_dir}")
                return str(final_save_dir.absolute())
        


        try:
            trainer = getattr(self, '_trainer', None)
            if trainer is None:

                try:
                    trainer = self.trainer
                except RuntimeError:
                    trainer = None
            
            if trainer is not None and hasattr(trainer, 'logger') and hasattr(trainer.logger, 'save_dir'):
                log_dir = Path(trainer.logger.save_dir)
                output_dir = log_dir.parent
            else:

                output_dir = Path.cwd()
        except Exception:

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
                        
                        with torch.no_grad():

                            physics_latents = self.physics_decoder.decode(generated_latents[i:i+1])
                            
                            pressure_pred = self.physics_decoder.query(phys_points, physics_latents)
                            
                            pressure_pred = (self.physics_decoder.cfg.get('PRESSURE_STD', 117.25) * pressure_pred + 
                                           self.physics_decoder.cfg.get('PRESSURE_MEAN', -94.5))
                            
                            pressures = pressure_pred.squeeze().cpu().numpy().astype(np.float32)
                        
                        faces_pv = np.hstack([np.full_like(faces[:,0:1], 3), faces]).astype(np.int32)
                        faces_flat = faces_pv.flatten()
                        
                        pv_mesh = pv.PolyData(vertices, faces_flat)
                        pv_mesh.point_data["p"] = pressures
                        
                        save_path = os.path.join(save_dir, f"{current_prefix}_physics_{i:03d}.vtp")
                        pv_mesh.save(save_path)
                        saved_paths.append(save_path)
                        print(f"Saved pressure: {save_path}")
                        
                    except Exception as e:
                        print(f"Failed to add physics to mesh {i}: {e}")

                        save_path = os.path.join(save_dir, f"{current_prefix}_physics_fallback_{i:03d}.ply")
                        try:
                            mesh.export(save_path)
                            saved_paths.append(save_path)
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
                            param.data = param.data.to(dtype=drag_dtype)
                            
                if hasattr(self.drag_decoder, 'transformer') and self.drag_decoder.transformer is not None:
                    self.drag_decoder.transformer = self.drag_decoder.transformer.to(dtype=drag_dtype)
                    
                if hasattr(self.drag_decoder, 'decoder') and self.drag_decoder.decoder is not None:
                    self.drag_decoder.decoder = self.drag_decoder.decoder.to(dtype=drag_dtype)
                    
                if hasattr(self.drag_decoder, 'embedder') and self.drag_decoder.embedder is not None:
                    self.drag_decoder.embedder = self.drag_decoder.embedder.to(dtype=drag_dtype)
                if hasattr(self.drag_decoder, 'time_proj') and self.drag_decoder.time_proj is not None:
                    self.drag_decoder.time_proj = self.drag_decoder.time_proj.to(dtype=drag_dtype)

                    for name, param in self.drag_decoder.time_proj.named_parameters():
                        if param.dtype != drag_dtype:
                            param.data = param.data.to(dtype=drag_dtype)


                inconsistent_params = []
                for name, param in self.drag_decoder.named_parameters():
                    if param.dtype != drag_dtype:
                        inconsistent_params.append((name, param.dtype))
                        param.data = param.data.to(dtype=drag_dtype)
                

                post_kl_dtype = next(self.drag_decoder.post_kl.parameters()).dtype if hasattr(self.drag_decoder, 'post_kl') and self.drag_decoder.post_kl is not None else "N/A"
                

                dummy_timestep = torch.tensor([1.0], device=drag_input_latents.device, dtype=drag_dtype)
                dummy_timestep = dummy_timestep.expand(drag_input_latents.shape[0])
                
                drag_latents = self.drag_decoder.decode(drag_input_latents)
                
                emb = self.drag_decoder._sinusoidal_time_embedding(dummy_timestep[:1]).to(dtype=drag_dtype)
                t_cond = self.drag_decoder.time_proj(emb).unsqueeze(1)
                

                if t_cond.dtype != drag_latents.dtype:
                    t_cond = t_cond.to(dtype=drag_latents.dtype)
                
                drag_latents = drag_latents + t_cond
                
                drag_predictions = self.drag_decoder.predict_drag(drag_latents)
                
            except Exception as drag_error:
                print(f"Drag decoder inference failed: {drag_error}")
                print(f"Traceback: {traceback.format_exc()}")
                
                raise drag_error
            
            drag_values = drag_predictions.squeeze().cpu().numpy().astype(np.float32)
            
            for i, mesh in enumerate(meshes):
                current_prefix = prefixes[i]
                if mesh is not None:
                    try:
                        mesh_path = os.path.join(save_dir, f"{current_prefix}_drag_{i:03d}.ply")
                        drag_path = os.path.join(save_dir, f"{current_prefix}_drag_{i:03d}.txt")
                        
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
                            mesh.export(mesh_path)
                            saved_paths.append((mesh_path, None))
                        except Exception as e2:
                            print(f"Failed to save fallback mesh {i}: {e2}")
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

    def save_meshes(self, meshes, save_dir, prefix):
        """
        Save meshes to PLY files
        
        Args:
            meshes: list of trimesh objects
            save_dir: directory to save meshes
            prefix: filename prefix (can be string or list of strings for each mesh)
            
        Returns:
            saved_paths: list of saved file paths
        """
        import os
        
        os.makedirs(save_dir, exist_ok=True)
        saved_paths = []
        
        if isinstance(prefix, list):
            prefixes = prefix[:len(meshes)]
            while len(prefixes) < len(meshes):
                prefixes.append("mesh")
        else:
            prefixes = [prefix] * len(meshes)
        
        for i, mesh in enumerate(meshes):
            current_prefix = prefixes[i]
            if mesh is not None:

                if isinstance(mesh, list):
                    for j, m in enumerate(mesh):
                        save_path = os.path.join(save_dir, f"{current_prefix}_mesh_{i:03d}_{j:03d}.ply")
                        try:
                            m.export(save_path)
                            saved_paths.append(save_path)
                            print(f"Saved mesh: {save_path}")
                        except Exception as e:
                            print(f"Failed to save mesh {i}_{j}: {e}")
                            saved_paths.append(None)
                else:
                    save_path = os.path.join(save_dir, f"{current_prefix}_mesh_{i:03d}.ply")
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

    def save_condition_images(self, condition_images, save_dir, prefix, batch_idx=0):
        """
        Save condition images to files
        
        Args:
            condition_images: tensor of condition images (B, C, H, W)
            save_dir: directory to save images
            prefix: filename prefix (can be string or list of strings for each image)
            batch_idx: batch index for filename
            
        Returns:
            saved_paths: list of saved image file paths
        """
        if condition_images is None:
            return []
            
        import os
        from PIL import Image
        import numpy as np
        
        os.makedirs(save_dir, exist_ok=True)
        saved_paths = []
        
        try:
            if isinstance(condition_images, torch.Tensor):
                images_np = condition_images.detach().cpu().numpy()
                
                if len(images_np.shape) == 4:
                    batch_size = images_np.shape[0]
                elif len(images_np.shape) == 3:
                    images_np = images_np[None, ...]
                    batch_size = 1
                else:
                    print(f"Unsupported condition image shape: {images_np.shape}")
                    return []
                
                if isinstance(prefix, list):
                    prefixes = prefix[:batch_size]
                    while len(prefixes) < batch_size:
                        prefixes.append(f"condition_batch{batch_idx}")
                else:
                    prefixes = [prefix] * batch_size
                
                for i in range(batch_size):
                    img_data = images_np[i]
                    current_prefix = prefixes[i]
                    
                    if img_data.shape[0] in [1, 3, 4]:
                        img_data = np.transpose(img_data, (1, 2, 0))
                    
                    if img_data.shape[-1] == 1:
                        img_data = img_data.squeeze(-1)
                        mode = 'L'
                    elif img_data.shape[-1] == 3:
                        mode = 'RGB'
                    elif img_data.shape[-1] == 4:
                        mode = 'RGBA'
                    else:
                        print(f"Unsupported number of channels: {img_data.shape[-1]}")
                        saved_paths.append(None)
                        continue
                    
                    if img_data.dtype == np.float32 or img_data.dtype == np.float64:
                        if img_data.min() >= -1.0 and img_data.max() <= 1.0:
                            img_data = (img_data + 1.0) / 2.0
                        elif img_data.min() >= 0.0 and img_data.max() <= 1.0:
                            pass
                        else:
                            img_min, img_max = img_data.min(), img_data.max()
                            if img_max > img_min:
                                img_data = (img_data - img_min) / (img_max - img_min)
                        
                        img_data = (img_data * 255).astype(np.uint8)
                    
                    try:
                        pil_img = Image.fromarray(img_data, mode=mode)
                        save_path = os.path.join(save_dir, f"{current_prefix}_condition_batch{batch_idx:03d}_{i:03d}.png")
                        pil_img.save(save_path)
                        saved_paths.append(save_path)
                    except Exception as e:
                        print(f"Failed to save condition image {i}: {e}")
                        saved_paths.append(None)
                        
            else:
                print(f"Unsupported condition image type: {type(condition_images)}")
                return []
                
        except Exception as e:
            print(f"Error saving condition images: {e}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return []
            
        return saved_paths

    def test_step(self, batch, batch_idx):
        """
        Test step for mesh generation using pretrained VAE and diffusion models.
        Can be called from main.py using test config file.
        """
        
        try:
            batch_size = min(4, batch.get("coarse_surface", torch.tensor([1, 1, 1, 1])).shape[0])
            device = next(self.model.parameters()).device
            
            sample_batch = {}
            dummy_coarse = None
            dummy_sharp = None
            
            if self.cond_stage_key in batch:
                condition_data = batch[self.cond_stage_key][:batch_size]
                sample_batch[self.cond_stage_key] = condition_data
                

                if 'uid' in batch:
                    uid_data = batch['uid']
                    if isinstance(uid_data, list):
                        clean_uid = str(uid_data[0])
                    else:
                        clean_uid = str(uid_data)
                    

                    clean_uid = os.path.basename(clean_uid)
                    if clean_uid.endswith('.npz'):
                        clean_uid = clean_uid[:-4]
                    
                    sample_batch['uid'] = clean_uid
                

                if 'drag_coefficient' in batch:
                    sample_batch['drag_coefficient'] = batch['drag_coefficient'][:batch_size]

                num_points = 32768
                dummy_coarse = torch.randn(batch_size, num_points, 6, device=device, dtype=torch.float16)
                dummy_sharp = torch.randn(batch_size, num_points, 6, device=device, dtype=torch.float16)
                
                outputs = self.sample(sample_batch, output_type='latent')
                if outputs is not None and hasattr(outputs, 'shape'):
                    generated_latents = outputs[:batch_size]
                else:
                    print("Error: Failed to generate conditional latents")
                    return {"error": "Failed to generate conditional latents"}
                    
            else:
                return {"error": "Failed to generate latents"}
            
            test_resolution = 512
            if self.test_config is not None and hasattr(self.test_config, 'resolution'):
                test_resolution = self.test_config.resolution
            

            save_dir = self.get_mesh_save_dir(
                "test_meshes",
                drag_guidance_scale=getattr(self.test_config, 'drag_guidance_scale', None),
                start_time_step=getattr(self.test_config, 'start_time_step', None),
                num_inference_steps=getattr(self.test_config, 'num_inference_steps', None),
                num_optimization_loops=getattr(self.test_config, 'num_optimization_loops', None),
                num_optimization_steps=getattr(self.test_config, 'num_optimization_steps', None),
                learning_rate=getattr(self.test_config, 'learning_rate', None)
            )
            
            uids = []
            if 'uid' in batch:
                uid_data = batch['uid']
                if isinstance(uid_data, list):
                    uids = uid_data[:batch_size]
                elif isinstance(uid_data, str):
                    uids = [uid_data]
                else:
                    try:
                        if hasattr(uid_data, 'tolist'):
                            uids = uid_data.tolist()[:batch_size]
                        else:
                            uids = [str(uid_data)]
                    except:
                        print("Warning: Could not extract UIDs from batch")
                        uids = [None] * batch_size
            else:
                print("Warning: No 'uid' found in batch, using default UIDs")
                uids = [f"unknown_{batch_idx}_{i}" for i in range(batch_size)]
            
            while len(uids) < batch_size:
                uids.append(f"missing_{batch_idx}_{len(uids)}")
            uids = uids[:batch_size]
            
            meshes = self.extract_geometry_by_diffdmc(
                generated_latents, 
                resolution=test_resolution,
                batch_size=10000,
                verbose=True
            )
            

            if meshes and any(mesh is not None for mesh in meshes):

                mesh_uids = uids[:len(meshes)]
                while len(mesh_uids) < len(meshes):
                    mesh_uids.append(f"missing_{batch_idx}_{len(mesh_uids)}")
                mesh_uids = mesh_uids[:len(meshes)]
                
                uid_prefixes = []
                for uid in mesh_uids:
                    if uid:
                        clean_uid = os.path.basename(str(uid))
                        if clean_uid.endswith('.npz'):
                            clean_uid = clean_uid[:-4]
                        elif clean_uid.endswith('.obj'):
                            clean_uid = clean_uid[:-4]
                        elif clean_uid.endswith('.stl'):
                            clean_uid = clean_uid[:-4]
                        uid_prefixes.append(clean_uid)
                    else:
                        uid_prefixes.append(f"unknown_{batch_idx}")
                saved_paths = self.save_meshes(meshes, save_dir, uid_prefixes)
                


                if self.cond_stage_key in sample_batch and sample_batch[self.cond_stage_key] is not None:
                    condition_images = sample_batch[self.cond_stage_key]
                    saved_condition_paths = self.save_condition_images(
                        condition_images, save_dir, uid_prefixes, batch_idx
                    )
                    condition_saves = len([p for p in saved_condition_paths if p])
                
                if self.physics_enabled and self.physics_decoder is not None:
                    try:
                        meshes_with_physics = self._add_physics_to_meshes(
                            meshes, generated_latents, dummy_coarse, dummy_sharp, save_dir, uid_prefixes
                        )
                        if meshes_with_physics:
                            successful_saves = len([p for p in meshes_with_physics if p])
                        else:
                            print("Physics prediction failed")
                    except Exception as physics_error:
                        print(f"Physics prediction failed: {physics_error}")
                
                if self.drag_enabled and self.drag_decoder is not None:
                    try:
                        meshes_with_drag, drag_values = self._add_drag_to_meshes(
                            meshes, generated_latents, save_dir, uid_prefixes, return_drag=True
                        )
                        print("Optimized drag coefficient", drag_values)
                        if meshes_with_drag:
                            drag_saves = len([p for p in meshes_with_drag if p and p[0] is not None])
                        else:
                            print("Drag prediction failed")
                    except Exception as drag_error:
                        print(f"Drag prediction failed: {drag_error}")
                
        except Exception as e:
            import traceback
            print(f"Test step failed: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            self.log("test/meshes_generated", 0, prog_bar=True, logger=True)
            return {"error": str(e)}

    # def on_test_epoch_end(self):
    #     result = self._aggregate_test_metrics()
    #     if result:
    #         self.test_step_outputs = []
            
    # def on_test_end(self):
    #     """
    #     Alternative hook that might be called instead of on_test_epoch_end
    #     """
    #     result = self._aggregate_test_metrics()
    #     if result:
    #         self.test_step_outputs = []
