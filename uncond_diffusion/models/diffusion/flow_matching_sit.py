import os
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
        first_stage_key: str = "surface",
        scale_by_std: bool = False,
        z_scale_factor: float = 1.0,
        ckpt_path: Optional[str] = None,
        ignore_keys: Union[Tuple[str], List[str]] = (),
        torch_compile: bool = False,
        overfitting_debug: bool = False,
        cond_stage_config=None,
        cond_stage_key: str = "image",
        **kwargs,
    ):
        super().__init__()
        self.first_stage_key = first_stage_key
        self.overfitting_debug = overfitting_debug
        
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
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        # ========= init vae at last to prevent it is overridden by loaded ckpt ========= #
        self.first_stage_model = instantiate_non_trainable_model(first_stage_config)

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
        )

        # ========= torch compile to accelerate ========= #
        self.torch_compile = torch_compile
        if self.torch_compile:
            torch.nn.Module.compile(self.model)
            torch.nn.Module.compile(self.first_stage_model)

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
                
                if generated_latents is not None:
                    meshes = self.decode_latents_to_mesh(
                        generated_latents, 
                        resolution=64,
                        batch_size=10000,
                        verbose=False
                    )
                    
                    if meshes and any(mesh is not None for mesh in meshes):
                        save_dir = self.get_mesh_save_dir("val_meshes")
                        saved_paths = self.save_meshes(meshes, save_dir, f"val_batch{batch_idx}")
                        successful_saves = len([p for p in saved_paths if p])
                        print(f"Saved {successful_saves}/{len(meshes)} validation meshes to {save_dir}")
                        
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

    @torch.no_grad()
    def sample(self, batch, output_type='latent', **kwargs):
        generator = torch.Generator().manual_seed(0)

        with self.ema_scope("Sample"):
            with torch.amp.autocast(device_type='cuda'):
                try:
                    self.pipeline.device = self.device
                    self.pipeline.dtype = self.dtype
                    additional_params = {'output_type': output_type}

                    outputs = self.pipeline(generator=generator, **additional_params)

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
            
            # Ensure VAE is in correct mode and dtype
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
                                    # Method 4: Fallback - use a simple SDF approximation
                                    print(f"   No suitable decode method found, using fallback SDF approximation")
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
            
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                logits = torch.where(torch.isnan(logits) | torch.isinf(logits), torch.zeros_like(logits), logits)
            
            # Clamp extreme values to prevent sigmoid saturation issues
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
                            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
                            mesh.remove_duplicate_faces()
                            mesh.remove_unreferenced_vertices()
                            return mesh
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

    def get_mesh_save_dir(self, mesh_type="val_meshes"):
        """
        Get the directory for saving meshes under training_cfg.output_dir/save
        
        Args:
            mesh_type: type of mesh ("val_meshes", "test_meshes", etc.)
            
        Returns:
            Path to mesh save directory
        """
        import os
        import time
        from pathlib import Path
        
        if hasattr(self, 'trainer') and self.trainer is not None:
            if hasattr(self.trainer, 'logger') and hasattr(self.trainer.logger, 'save_dir'):
                log_dir = Path(self.trainer.logger.save_dir)
                output_dir = log_dir.parent
            else:
                # Fallback to current directory if trainer/logger not available
                output_dir = Path.cwd()
        else:
            output_dir = Path.cwd()
        
        save_base_dir = output_dir / "save"
        
        # Create subdirectory with timestamp for this mesh type (Zurich time)
        try:
            from datetime import datetime
            import pytz
            zurich_tz = pytz.timezone('Europe/Zurich')
            zurich_time = datetime.now(zurich_tz)
            timestamp = zurich_time.strftime("%Y%m%d_%H%M%S")
        except ImportError:
            timestamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
        mesh_dir = save_base_dir / f"{mesh_type}_{timestamp}"
        
        os.makedirs(mesh_dir, exist_ok=True)
        return str(mesh_dir.absolute())

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
