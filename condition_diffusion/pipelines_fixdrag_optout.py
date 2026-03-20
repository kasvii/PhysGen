# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party components and must ensure that the usage of the third party components adheres to


# For avoidance of doubts, Hunyuan 3D means the large language models and



# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

import copy
import importlib
import inspect
import os, sys
import time
import pytz
from typing import List, Optional, Union

import numpy as np
import torch
import trimesh
import yaml
from PIL import Image
from diffusers.utils.torch_utils import randn_tensor
from diffusers.utils.import_utils import is_accelerate_version, is_accelerate_available
from tqdm import tqdm

from .models.autoencoders import ShapeVAE
from .models.autoencoders import SurfaceExtractors
from .utils import logger, synchronize_timer, smart_load_model


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


@synchronize_timer('Export to trimesh')
def export_to_trimesh(mesh_output):
    if isinstance(mesh_output, list):
        outputs = []
        for mesh in mesh_output:
            if mesh is None:
                outputs.append(None)
            else:
                mesh.mesh_f = mesh.mesh_f[:, ::-1]
                mesh_output = trimesh.Trimesh(mesh.mesh_v, mesh.mesh_f)
                outputs.append(mesh_output)
        return outputs
    else:
        mesh_output.mesh_f = mesh_output.mesh_f[:, ::-1]
        mesh_output = trimesh.Trimesh(mesh_output.mesh_v, mesh_output.mesh_f)
        return mesh_output


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config, **kwargs):
    if "target" not in config:
        raise KeyError("Expected key `target` to instantiate.")
    cls = get_obj_from_str(config["target"])
    params = config.get("params", dict())
    kwargs.update(params)
    instance = cls(**kwargs)
    return instance


class Hunyuan3DDiTPipeline:
    model_cpu_offload_seq = "conditioner->model->vae"
    _exclude_from_cpu_offload = []

    @classmethod
    @synchronize_timer('Hunyuan3DDiTPipeline Model Loading')
    def from_single_file(
        cls,
        ckpt_path,
        config_path,
        device='cuda',
        dtype=torch.float16,
        use_safetensors=None,
        **kwargs,
    ):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        if use_safetensors:
            ckpt_path = ckpt_path.replace('.ckpt', '.safetensors')
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Model file {ckpt_path} not found")
        logger.info(f"Loading model from {ckpt_path}")

        if use_safetensors:
            import safetensors.torch
            safetensors_ckpt = safetensors.torch.load_file(ckpt_path, device='cpu')
            ckpt = {}
            for key, value in safetensors_ckpt.items():
                model_name = key.split('.')[0]
                new_key = key[len(model_name) + 1:]
                if model_name not in ckpt:
                    ckpt[model_name] = {}
                ckpt[model_name][new_key] = value
        else:
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        model = instantiate_from_config(config['model'])
        model.load_state_dict(ckpt['model'])
        vae = instantiate_from_config(config['vae'])
        vae.load_state_dict(ckpt['vae'], strict=False)
        
        conditioner = None
        if 'conditioner' in config:
            conditioner = instantiate_from_config(config['conditioner'])
            if 'conditioner' in ckpt:
                conditioner.load_state_dict(ckpt['conditioner'])
            conditioner.eval()
            conditioner.requires_grad_(False)
        
        image_processor = None
        if 'image_processor' in config:
            image_processor = instantiate_from_config(config['image_processor'])
            
        scheduler = instantiate_from_config(config['scheduler'])

        model_kwargs = dict(
            vae=vae,
            model=model,
            scheduler=scheduler,
            conditioner=conditioner,
            image_processor=image_processor,
            device=device,
            dtype=dtype,
        )
        model_kwargs.update(kwargs)

        return cls(
            **model_kwargs
        )

    @classmethod
    def from_pretrained(
        cls,
        model_path,
        device='cuda',
        dtype=torch.float16,
        use_safetensors=False,
        variant='fp16',
        subfolder='hunyuan3d-dit-v2-1',
        **kwargs,
    ):
        kwargs['from_pretrained_kwargs'] = dict(
            model_path=model_path,
            subfolder=subfolder,
            use_safetensors=use_safetensors,
            variant=variant,
            dtype=dtype,
            device=device,
        )
        config_path, ckpt_path = smart_load_model(
            model_path,
            subfolder=subfolder,
            use_safetensors=use_safetensors,
            variant=variant
        )
        return cls.from_single_file(
            ckpt_path,
            config_path,
            device=device,
            dtype=dtype,
            use_safetensors=use_safetensors,
            **kwargs
        )

    def __init__(
        self,
        vae,
        model,
        scheduler,
        conditioner=None,
        image_processor=None,
        drag_estimator=None,
        phys_decoder=None,
        device='cuda',
        dtype=torch.float16,
        scale_factor=1.0,
        **kwargs
    ):
        self.vae = vae
        self.vae.eval()
        self.vae.split = 'val'
        
        self.model = model
        self.scheduler = scheduler
        self.conditioner = conditioner
        self.image_processor = image_processor
        self.drag_estimator = drag_estimator.to(torch.float32) if drag_estimator is not None else None
        self.phys_decoder = phys_decoder.to(torch.float32) if phys_decoder is not None else None
        self.kwargs = kwargs
        self.scale_factor = scale_factor
        
        self.x_dir = torch.tensor([1.,0.,0.], device=device)
        self.y_dir = torch.tensor([0.,1.,0.], device=device)
        self.z_dir = torch.tensor([0.,0.,1.], device=device)
        

        if self.conditioner is not None:
            self.conditioner.eval()
            self.conditioner.requires_grad_(False)
            
        self.to(device, dtype)

    def compile(self):
        self.vae = torch.compile(self.vae)
        self.model = torch.compile(self.model)
        if self.conditioner is not None:
            self.conditioner = torch.compile(self.conditioner)

    def enable_flashvdm(
        self,
        enabled: bool = True,
        adaptive_kv_selection=True,
        topk_mode='mean',
        mc_algo='mc',
        replace_vae=True,
    ):
        if enabled:
            model_path = self.kwargs['from_pretrained_kwargs']['model_path']
            turbo_vae_mapping = {
                'Hunyuan3D-2': ('tencent/Hunyuan3D-2', 'hunyuan3d-vae-v2-0-turbo'),
                'Hunyuan3D-2mv': ('tencent/Hunyuan3D-2', 'hunyuan3d-vae-v2-0-turbo'),
                'Hunyuan3D-2mini': ('tencent/Hunyuan3D-2mini', 'hunyuan3d-vae-v2-mini-turbo'),
            }
            model_name = model_path.split('/')[-1]
            if replace_vae and model_name in turbo_vae_mapping:
                model_path, subfolder = turbo_vae_mapping[model_name]
                self.vae = ShapeVAE.from_pretrained(
                    model_path, subfolder=subfolder,
                    use_safetensors=self.kwargs['from_pretrained_kwargs']['use_safetensors'],
                    device=self.device,
                )
            self.vae.enable_flashvdm_decoder(
                enabled=enabled,
                adaptive_kv_selection=adaptive_kv_selection,
                topk_mode=topk_mode,
                mc_algo=mc_algo
            )
        else:
            model_path = self.kwargs['from_pretrained_kwargs']['model_path']
            vae_mapping = {
                'Hunyuan3D-2': ('tencent/Hunyuan3D-2', 'hunyuan3d-vae-v2-0'),
                'Hunyuan3D-2mv': ('tencent/Hunyuan3D-2', 'hunyuan3d-vae-v2-0'),
                'Hunyuan3D-2mini': ('tencent/Hunyuan3D-2mini', 'hunyuan3d-vae-v2-mini'),
            }
            model_name = model_path.split('/')[-1]
            if model_name in vae_mapping:
                model_path, subfolder = vae_mapping[model_name]
                self.vae = ShapeVAE.from_pretrained(model_path, subfolder=subfolder)
            self.vae.enable_flashvdm_decoder(enabled=False)

    def to(self, device=None, dtype=None):
        if dtype is not None:
            self.dtype = dtype
            self.vae.to(dtype=dtype)
            self.model.to(dtype=dtype)
            if self.conditioner is not None:
                self.conditioner.to(dtype=dtype)
        if device is not None:
            self.device = torch.device(device)
            self.vae.to(device)
            self.model.to(device)
            if self.conditioner is not None:
                self.conditioner.to(device)

    @property
    def _execution_device(self):
        r"""
        Returns the device on which the pipeline's models will be executed. After calling
        [`~DiffusionPipeline.enable_sequential_cpu_offload`] the execution device can only be inferred from
        Accelerate's module hooks.
        """
        for name, model in self.components.items():
            if not isinstance(model, torch.nn.Module) or name in self._exclude_from_cpu_offload:
                continue

            if not hasattr(model, "_hf_hook"):
                return self.device
            for module in model.modules():
                if (
                    hasattr(module, "_hf_hook")
                    and hasattr(module._hf_hook, "execution_device")
                    and module._hf_hook.execution_device is not None
                ):
                    return torch.device(module._hf_hook.execution_device)
        return self.device

    def enable_model_cpu_offload(self, gpu_id: Optional[int] = None, device: Union[torch.device, str] = "cuda"):
        r"""
        Offloads all models to CPU using accelerate, reducing memory usage with a low impact on performance. Compared
        to `enable_sequential_cpu_offload`, this method moves one whole model at a time to the GPU when its `forward`
        method is called, and the model remains in GPU until the next model runs. Memory savings are lower than with
        `enable_sequential_cpu_offload`, but performance is much better due to the iterative execution of the `unet`.

        Arguments:
            gpu_id (`int`, *optional*):
                The ID of the accelerator that shall be used in inference. If not specified, it will default to 0.
            device (`torch.Device` or `str`, *optional*, defaults to "cuda"):
                The PyTorch device type of the accelerator that shall be used in inference. If not specified, it will
                default to "cuda".
        """
        if self.model_cpu_offload_seq is None:
            raise ValueError(
                "Model CPU offload cannot be enabled because no `model_cpu_offload_seq` class attribute is set."
            )

        if is_accelerate_available() and is_accelerate_version(">=", "0.17.0.dev0"):
            from accelerate import cpu_offload_with_hook
        else:
            raise ImportError("`enable_model_cpu_offload` requires `accelerate v0.17.0` or higher.")

        torch_device = torch.device(device)
        device_index = torch_device.index

        if gpu_id is not None and device_index is not None:
            raise ValueError(
                f"You have passed both `gpu_id`={gpu_id} and an index as part of the passed device `device`={device}"
                f"Cannot pass both. Please make sure to either not define `gpu_id` or not pass the index as part of "
                f"the device: `device`={torch_device.type}"
            )



        self._offload_gpu_id = gpu_id or torch_device.index or getattr(self, "_offload_gpu_id", 0)

        device_type = torch_device.type
        device = torch.device(f"{device_type}:{self._offload_gpu_id}")

        if self.device.type != "cpu":
            self.to("cpu")
            device_mod = getattr(torch, self.device.type, None)
            if hasattr(device_mod, "empty_cache") and device_mod.is_available():
                device_mod.empty_cache()  


        all_model_components = {k: v for k, v in self.components.items() if isinstance(v, torch.nn.Module)}

        self._all_hooks = []
        hook = None
        for model_str in self.model_cpu_offload_seq.split("->"):
            model = all_model_components.pop(model_str, None)
            if not isinstance(model, torch.nn.Module):
                continue

            _, hook = cpu_offload_with_hook(model, device, prev_module_hook=hook)
            self._all_hooks.append(hook)




        for name, model in all_model_components.items():
            if not isinstance(model, torch.nn.Module):
                continue

            if name in self._exclude_from_cpu_offload:
                model.to(device)
            else:
                _, hook = cpu_offload_with_hook(model, device)
                self._all_hooks.append(hook)

    def maybe_free_model_hooks(self):
        r"""
        Function that offloads all components, removes all model hooks that were added when using
        `enable_model_cpu_offload` and then applies them again. In case the model has not been offloaded this function
        is a no-op. Make sure to add this function to the end of the `__call__` function of your pipeline so that it
        functions correctly when applying enable_model_cpu_offload.
        """
        if not hasattr(self, "_all_hooks") or len(self._all_hooks) == 0:

            return

        for hook in self._all_hooks:
            hook.offload()
            hook.remove()


        self.enable_model_cpu_offload()

    @synchronize_timer('Encode cond')
    def encode_cond(self, image, additional_cond_inputs, do_classifier_free_guidance, dual_guidance):
        if self.conditioner is None:
            return None
            
        bsz = image.shape[0]
        

        try:
            cond = self.conditioner(image=image, **additional_cond_inputs)
        except Exception as e:
            print(f"Failed to encode condition: {e}")

            if hasattr(self.conditioner, 'unconditional_embedding'):
                cond = self.conditioner.unconditional_embedding(bsz, **additional_cond_inputs)
            else:
                return None

        if do_classifier_free_guidance:
            if hasattr(self.conditioner, 'unconditional_embedding'):
                un_cond = self.conditioner.unconditional_embedding(bsz, **additional_cond_inputs)
            else:
                print("Warning: conditioner has no unconditional_embedding method, CFG may not work properly")
                un_cond = cond

            if dual_guidance:
                un_cond_drop_main = copy.deepcopy(un_cond)
                if isinstance(cond, dict) and 'additional' in cond:
                    un_cond_drop_main['additional'] = cond['additional']

                def cat_recursive(a, b, c):
                    if isinstance(a, torch.Tensor):
                        return torch.cat([a, b, c], dim=0).to(self.dtype)
                    out = {}
                    for k in a.keys():
                        out[k] = cat_recursive(a[k], b[k], c[k])
                    return out

                cond = cat_recursive(cond, un_cond_drop_main, un_cond)
            else:
                def cat_recursive(a, b):
                    if isinstance(a, torch.Tensor):
                        return torch.cat([a, b], dim=0).to(self.dtype)
                    out = {}
                    for k in a.keys():
                        out[k] = cat_recursive(a[k], b[k])
                    return out

                cond = cat_recursive(cond, un_cond)
        return cond

    def prepare_extra_step_kwargs(self, generator, eta):



        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def prepare_latents(self, batch_size, dtype, device, generator, latents=None):
        shape = (batch_size, *self.vae.latent_shape)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        latents = latents * getattr(self.scheduler, 'init_noise_sigma', 1.0)
        return latents

    def prepare_image(self, image, mask=None) -> dict:
        if isinstance(image, torch.Tensor):
            outputs = {'image': image}
            if mask is not None:
                outputs['mask'] = mask
            return outputs
            
        if isinstance(image, str) and not os.path.exists(image):
            raise FileNotFoundError(f"Couldn't find image at path {image}")

        if self.image_processor is None:
            raise ValueError("image_processor is None, cannot process images")

        if not isinstance(image, list):
            image = [image]

        outputs = []
        for img in image:
            output = self.image_processor(img)
            outputs.append(output)

        if len(outputs) > 0 and isinstance(outputs[0], torch.Tensor):
            return {'image': torch.cat(outputs, dim=0) if len(outputs) > 1 else outputs[0]}
        
        cond_input = {k: [] for k in outputs[0].keys()}
        for output in outputs:
            for key, value in output.items():
                cond_input[key].append(value)
        for key, value in cond_input.items():
            if isinstance(value[0], torch.Tensor):
                cond_input[key] = torch.cat(value, dim=0)

        return cond_input

    def get_guidance_scale_embedding(self, w, embedding_dim=512, dtype=torch.float32):
        """
        See https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298

        Args:
            timesteps (`torch.Tensor`):
                generate embedding vectors at these timesteps
            embedding_dim (`int`, *optional*, defaults to 512):
                dimension of the embeddings to generate
            dtype:
                data type of the generated embeddings

        Returns:
            `torch.FloatTensor`: Embedding vectors with shape `(len(timesteps), embedding_dim)`
        """
        assert len(w.shape) == 1
        w = w * 1000.0

        half_dim = embedding_dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
        emb = w.to(dtype)[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1))
        assert emb.shape == (w.shape[0], embedding_dim)
        return emb

    def set_surface_extractor(self, mc_algo):
        if mc_algo is None:
            return
        logger.info('The parameters `mc_algo` is deprecated, and will be removed in future versions.\n'
                    'Please use: \n'
                    'from hy3dshape.models.autoencoders import SurfaceExtractors\n'
                    'pipeline.vae.surface_extractor = SurfaceExtractors[mc_algo]() instead\n')
        if mc_algo not in SurfaceExtractors.keys():
            raise ValueError(f"Unknown mc_algo {mc_algo}")
        self.vae.surface_extractor = SurfaceExtractors[mc_algo]()

    @torch.no_grad()
    def __call__(
        self,
        image: Union[str, List[str], Image.Image] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        eta: float = 0.0,
        guidance_scale: float = 7.5,
        dual_guidance_scale: float = 10.5,
        dual_guidance: bool = True,
        generator=None,
        box_v=1.01,
        octree_resolution=384,
        mc_level=-1 / 512,
        num_chunks=8000,
        mc_algo=None,
        output_type: Optional[str] = "trimesh",
        enable_pbar=True,
        **kwargs,
    ) -> List[List[trimesh.Trimesh]]:
        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        self.set_surface_extractor(mc_algo)

        device = self.device
        dtype = self.dtype
        do_classifier_free_guidance = guidance_scale >= 0 and \
                                      getattr(self.model, 'guidance_cond_proj_dim', None) is None
        dual_guidance = dual_guidance_scale >= 0 and dual_guidance

        if not isinstance(image, torch.Tensor):
            cond_inputs = self.prepare_image(image)
            image = cond_inputs.pop('image')
        
        cond = self.encode_cond(
            image=image,
            additional_cond_inputs=cond_inputs,
            do_classifier_free_guidance=do_classifier_free_guidance,
            dual_guidance=False,
        )
        batch_size = image.shape[0]

        t_dtype = torch.long
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas)

        latents = self.prepare_latents(batch_size, dtype, device, generator)
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        guidance_cond = None
        if getattr(self.model, 'guidance_cond_proj_dim', None) is not None:
            logger.info('Using lcm guidance scale')
            guidance_scale_tensor = torch.tensor(guidance_scale - 1).repeat(batch_size)
            guidance_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.model.guidance_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)
        with synchronize_timer('Diffusion Sampling'):
            for i, t in enumerate(tqdm(timesteps, desc="Diffusion Sampling", leave=False, disable=True)):
                if do_classifier_free_guidance:
                    latent_model_input = torch.cat([latents] * (3 if dual_guidance else 2))
                else:
                    latent_model_input = latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                timestep_tensor = torch.tensor([t], dtype=t_dtype, device=device)
                timestep_tensor = timestep_tensor.expand(latent_model_input.shape[0])
                noise_pred = self.model(latent_model_input, timestep_tensor, cond, guidance_cond=guidance_cond)

                if do_classifier_free_guidance:
                    if dual_guidance:
                        noise_pred_clip, noise_pred_dino, noise_pred_uncond = noise_pred.chunk(3)
                        noise_pred = (
                            noise_pred_uncond
                            + guidance_scale * (noise_pred_clip - noise_pred_dino)
                            + dual_guidance_scale * (noise_pred_dino - noise_pred_uncond)
                        )
                    else:
                        noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

                outputs = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs)
                latents = outputs.prev_sample

                if callback is not None and i % callback_steps == 0:
                    step_idx = i // getattr(self.scheduler, "order", 1)
                    callback(step_idx, t, outputs)

        return self._export(
            latents,
            output_type,
            box_v, mc_level, num_chunks, octree_resolution, mc_algo,
        )

    def _export(
        self,
        latents,
        output_type='trimesh',
        box_v=1.01,
        mc_level=0.0,
        num_chunks=20000,
        octree_resolution=256,
        mc_algo='mc',
        enable_pbar=True
    ):
        if not output_type == "latent":
            latents = 1. / self.scale_factor * latents
            latents = self.vae(latents)
            outputs = self.vae.latents2mesh(
                latents,
                bounds=box_v,
                mc_level=mc_level,
                num_chunks=num_chunks,
                octree_resolution=octree_resolution,
                mc_algo=mc_algo,
                enable_pbar=enable_pbar,
            )
        else:
            outputs = latents

        if output_type == 'trimesh':
            outputs = export_to_trimesh(outputs)

        return outputs
    
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
            
            device = latents.device
            B = latents.shape[0]
            
            with torch.no_grad():

                if hasattr(self, 'vae') and self.vae is not None:
                    unscaled_latents = 1. / self.scale_factor * latents
                    unscaled_latents = self.vae.post_kl(unscaled_latents.half())
                    unscaled_latents = self.vae.transformer(unscaled_latents)
                else:

                    unscaled_latents = latents
                    print("Warning: No VAE found, using latents directly")
                
                
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
                        if hasattr(self.vae, 'decoder') and hasattr(self.vae.decoder, '__call__'):
                            logits_batch = self.vae.decoder(batch_queries, unscaled_latents)
                        else:

                            logits_batch = self._fallback_sdf_query(batch_queries, unscaled_latents)
                    except Exception as decode_error:
                        print(f"   Decode method failed: {decode_error}, using fallback")
                        logits_batch = self._fallback_sdf_query(batch_queries, unscaled_latents)
                    
                    batch_logits.append(logits_batch.squeeze(0).squeeze(-1).float())
                
                grid_logits = torch.cat(batch_logits, dim=0).view(B, grid_size[0], grid_size[1], grid_size[2]).float()
                
                
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

    def _query_vae_pipeline(self, queries, decoded_latents):
        """
        Query VAE for occupancy values in pipeline context
        """
        batch_size, num_queries, _ = queries.shape
        


        if hasattr(self, 'vae') and hasattr(self.vae, 'surface_extractor'):
            try:

                distances = torch.norm(queries, dim=-1, keepdim=True)
                radius = 0.7
                sdf = distances - radius
                logits = -sdf * 3.0
                return logits
            except Exception as e:
                print(f"VAE query failed: {e}")
        

        return self._fallback_sdf_query(queries, decoded_latents)

    def _save_slice_visualization(self, grid_logits, grid_size, save_slice_dir, verbose=True):
        """
        Save slice visualization for debugging
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

    def _fallback_sdf_query(self, queries, latents):
        """
        Fallback SDF query method using simple geometric shapes
        """
        batch_size, num_queries, _ = queries.shape
        

        distances = torch.norm(queries, dim=-1, keepdim=True)
        radius = 0.8
        sdf = distances - radius
        

        logits = -sdf * 5.0
        
        return logits


class Hunyuan3DDiTFlowMatchingGuidanceLoopPipeline(Hunyuan3DDiTPipeline):
    
    def get_mesh_save_dir(self, mesh_type="val_meshes", test_save_dir=None, drag_guidance_scale=None, start_time_step=None, 
                         num_inference_steps=None, num_optimization_loops=None, num_optimization_steps=None, learning_rate=None):
        """
        Get the directory for saving meshes and logs, using test config save_dir if available
        
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
        import os
        import time
        from pathlib import Path
        
        if test_save_dir:
            save_dir = Path(test_save_dir)

            final_save_dir = save_dir / f"{mesh_type}"
            
            os.makedirs(final_save_dir, exist_ok=True)
            return str(final_save_dir.absolute())
        

        try:
            trainer = getattr(self, '_trainer', None)
            if trainer is None:

                try:
                    trainer = self.trainer
                except (RuntimeError, AttributeError):
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


    def __call__(
        self,
        image: Union[str, List[str], Image.Image, dict, List[dict], torch.Tensor] = None,
        condition_image: Union[str, List[str], Image.Image, dict, List[dict], torch.Tensor] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        eta: float = 0.0,
        guidance_scale: float = 5.0,
        generator=None,
        box_v=1.01,
        octree_resolution=384,
        mc_level=0.0,
        mc_algo=None,
        num_chunks=8000,
        output_type: Optional[str] = "latents",
        enable_pbar=True,
        mask = None,
        tgt_drag = 2.0,
        init_drag = None,
        drag_guidance_scale = 0.0,
        start_time_step = 0.75,
        num_optimization_loops = 100,
        num_optimization_steps = 10,
        learning_rate = 0.01,
        angle_threshold = 10.0,
        test_save_dir = None,
        uid = None,
        **kwargs,
    ) -> List[List[trimesh.Trimesh]]:
        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        self.set_surface_extractor(mc_algo)

        device = self.device
        dtype = self.dtype
        do_classifier_free_guidance = guidance_scale >= 0 and not (
            hasattr(self.model, 'guidance_embed') and
            self.model.guidance_embed is True
        )


        if condition_image is not None:
            image = condition_image
        

        condition_data = None
        for key, value in kwargs.items():
            if key.endswith('_image') or key.startswith('condition'):
                if value is not None:
                    condition_data = value
                    image = condition_data
                    break

        if image is None and mask is None:
            batch_size = 1
            dummy_image = torch.zeros(batch_size, 3, 256, 256, device=device, dtype=dtype)
            image = dummy_image
            cond = None
        else:

            if self.conditioner is None:
                cond = None
            else:
                cond_inputs = self.prepare_image(image, mask)
                image = cond_inputs.pop('image')
                cond = self.encode_cond(
                    image=image,
                    additional_cond_inputs=cond_inputs,
                    do_classifier_free_guidance=do_classifier_free_guidance,
                    dual_guidance=False,
                )

        batch_size = image.shape[0]


        sigmas = np.linspace(0, 1, num_inference_steps) if sigmas is None else sigmas
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
        )
        latents = self.prepare_latents(batch_size, dtype, device, generator)

        guidance = None
        if hasattr(self.model, 'guidance_embed') and \
            self.model.guidance_embed is True:
            guidance = torch.tensor([guidance_scale] * batch_size, device=device, dtype=dtype)

        tgt_drag_denorm = tgt_drag
        tgt_drag = (tgt_drag - 0.26325600) / 0.02007368
            
        target_drag_tensor = torch.full((latents.shape[0], 1), fill_value=tgt_drag, device=device, dtype=dtype)

        import time
        import os
        log_save_dir = self.get_mesh_save_dir(
            "test_meshes", 
            test_save_dir=test_save_dir,
            drag_guidance_scale=drag_guidance_scale,
            start_time_step=start_time_step,
            num_inference_steps=num_inference_steps,
            num_optimization_loops=num_optimization_loops,
            num_optimization_steps=num_optimization_steps,
            learning_rate=learning_rate,
        )
        from datetime import datetime
        import pytz
        zurich_tz = pytz.timezone('Europe/Zurich')
        zurich_time = datetime.now(zurich_tz)
        timestamp = zurich_time.strftime("%Y%m%d_%H%M%S")
        
        if uid is not None:

            clean_uid = str(uid).replace('/', '_').replace('\\', '_').replace(':', '_')
            sampling_log_path = os.path.join(log_save_dir, f"{clean_uid}_sampling_log_{timestamp}.txt")
            optimization_log_path = os.path.join(log_save_dir, f"{clean_uid}_optimization_log_{timestamp}.txt")
        else:
            sampling_log_path = os.path.join(log_save_dir, f"sampling_log_{timestamp}.txt")
            optimization_log_path = os.path.join(log_save_dir, f"optimization_log_{timestamp}.txt")
        
        num_optimization_loops = 1 if self.drag_estimator is None else num_optimization_loops + 1

        with open(sampling_log_path, 'w') as f:
            f.write("=== Sampling Process Log ===\n")
            f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            if uid is not None:
                f.write(f"UID: {uid}\n")
            f.write(f"Batch size: {batch_size}\n")
            f.write(f"Num inference steps: {num_inference_steps}\n")
            f.write(f"Guidance scale: {guidance_scale}\n")
            f.write(f"Target drag: {tgt_drag_denorm}\n")
            f.write(f"Drag guidance scale: {drag_guidance_scale}\n")
            f.write(f"Start time step: {start_time_step}\n")
            f.write(f"Num optimization loops: {num_optimization_loops}\n")
            f.write(f"Num optimization steps: {num_optimization_steps}\n")
            f.write(f"Learning rate: {learning_rate}\n")
            f.write(f"Angle threshold: {angle_threshold}\n")
            f.write("-" * 50 + "\n")
        
        with open(optimization_log_path, 'w') as f:
            f.write("=== Optimization Process Log ===\n")
            f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            if uid is not None:
                f.write(f"UID: {uid}\n")
            f.write(f"Num optimization loops: {num_optimization_loops}\n")
            f.write(f"Num optimization steps per loop: {num_optimization_steps}\n")
            f.write(f"Learning rate: {learning_rate}\n")
            f.write(f"Angle threshold: {angle_threshold} degrees\n")
            f.write("-" * 50 + "\n")
        
        for opt_step in tqdm(range(num_optimization_loops), desc="Alternating Optimization"):
            with open(sampling_log_path, 'a') as f:
                f.write(f"\n=== Optimization Step {opt_step} ===\n")
                f.write(f"Time: {time.strftime('%H:%M:%S')}\n")
            
            if opt_step !=0:
                timesteps, num_inference_steps = retrieve_timesteps(
                    self.scheduler,
                    num_inference_steps,
                    device,
                    sigmas=sigmas,
                )
                timesteps = timesteps[int(len(timesteps) * start_time_step):]
                noise = randn_tensor(latents.shape, generator=generator, device=device, dtype=dtype)
                sigma = sigmas[int(num_inference_steps * start_time_step)]
                with open(sampling_log_path, 'a') as f:
                    f.write(f"Sigma value: {sigma}\n")
                    f.write(f"Timesteps length: {len(timesteps)}\n")
                latents = (1.0 - sigma) * noise + sigma * latents
            with synchronize_timer('Diffusion Sampling'):
                with open(sampling_log_path, 'a') as f:
                    f.write(f"Starting diffusion sampling with {len(timesteps)} timesteps\n")
                for i, t in enumerate(tqdm(timesteps, desc="Diffusion Sampling", disable=True)):
                    if do_classifier_free_guidance:
                        latent_model_input = torch.cat([latents] * 2)
                    else:
                        latent_model_input = latents


                    timestep = t.expand(latent_model_input.shape[0]).to(latents.dtype)
                    timestep = timestep / self.scheduler.config.num_train_timesteps
                    
                    if cond is not None:
                        contexts = {'main': cond}
                    else:
                        batch_size = latent_model_input.shape[0]
                        device = latent_model_input.device
                        dtype = latent_model_input.dtype
                        
                        text_len = getattr(self.model, 'text_len', 256)
                        context_dim = getattr(self.model, 'context_dim', 768)
                        dummy_context = torch.zeros(batch_size, text_len, context_dim, device=device, dtype=dtype)
                        contexts = {'main': dummy_context}
                        
                    
                    noise_pred = self.model(latent_model_input, timestep, contexts, guidance=guidance)

                    if do_classifier_free_guidance:
                        noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

                    outputs = self.scheduler.step(noise_pred, t, latents)
                    
                    if self.drag_estimator is not None and timestep[0] > start_time_step:
                        with open(sampling_log_path, 'a') as f:
                            f.write(f"Step {i}: Using drag estimator for guidance (timestep={timestep[0].item():.4f})\n")

                        with torch.inference_mode(False), torch.cuda.amp.autocast(enabled=False):
                            latents_req = latents.clone().detach().float().requires_grad_(True)
                            
                            latents_drag = self.drag_estimator.decode(latents_req)
                            
                            t_cond = self.drag_estimator._time_condition(timestep[:1]).unsqueeze(1)
                            latents_drag = latents_drag + t_cond
                
                            drag_coeff_denorm = self.drag_estimator.predict_drag(latents_drag)
                            drag_coeff = (drag_coeff_denorm - 0.26325600) / 0.02007368
                            
                            loss = torch.nn.functional.mse_loss(drag_coeff, target_drag_tensor.to(dtype=torch.float32))

                            drag_grad = torch.autograd.grad(loss, latents_req, retain_graph=False, create_graph=False, allow_unused=False)[0]
                            
                        if drag_grad is not None:
                            drag_loss = loss.item()
                            drag_coeff_mean = drag_coeff.mean().item()
                            drag_coeff_denorm_mean = drag_coeff_denorm.mean().item()
                            drag_grad_norm = drag_grad.norm().item()
                            with open(sampling_log_path, 'a') as f:
                                f.write(f"  Drag loss: {drag_loss:.6f}, Drag coeff: {drag_coeff_mean:.6f}, Drag coeff denorm: {drag_coeff_denorm_mean:.6f}, Grad norm: {drag_grad_norm:.6f}\n")
                                f.write(f"  noise_pred norm: {noise_pred.norm().item():.6f}, drag_grad norm: {drag_grad.norm().item():.6f}\n")
                            latents = outputs.prev_sample - drag_guidance_scale * drag_grad
                        else:
                            latents = outputs.prev_sample
                            print("Warning: drag_grad is None, skipping drag guidance for this step")
                            with open(sampling_log_path, 'a') as f:
                                f.write(f"  Warning: drag_grad is None, skipping drag guidance\n")
                    else:  
                        latents = outputs.prev_sample

                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, outputs)
                

                if self.phys_decoder is not None and opt_step < num_optimization_loops - 1:
                    
                    meshes = self.extract_geometry_by_diffdmc(
                        latents, 
                        resolution=128,
                        batch_size=10000,
                        verbose=True
                    )
                    vertices = meshes[0].vertices.astype(np.float32)
                    faces = meshes[0].faces.astype(np.int32)

                    unscale_latents = 1. / self.scale_factor * latents
                    physics_latents = self.phys_decoder.decode(unscale_latents).float()
                    
                    phys_points = torch.from_numpy(vertices).unsqueeze(0).to(physics_latents.device, dtype=torch.float32)
                    
                    pressure_pred = self.phys_decoder.query(phys_points, physics_latents)
                    
                    optimized_latents = self._optimize_aerodynamic_constraints(
                        unscale_latents, vertices, faces, pressure_pred,
                        num_optimization_steps=num_optimization_steps,
                        learning_rate=learning_rate,
                        angle_threshold=angle_threshold,
                        optimization_log_path=optimization_log_path,
                        uid=uid
                    )

                    if optimized_latents is not None:
                        latents = optimized_latents * self.scale_factor
                    else:
                        print("Warning: Optimization returned None, using original latents")
            
        with open(sampling_log_path, 'a') as f:
            f.write(f"\n=== Pipeline Completed ===\n")
            f.write(f"Time: {time.strftime('%H:%M:%S')}\n")
            f.write(f"Total optimization steps: {num_optimization_loops}\n")
            f.write("-" * 50 + "\n")
        
        if self.drag_estimator is not None:
            try:
                with torch.inference_mode(False), torch.cuda.amp.autocast(enabled=False):

                    dummy_timestep = torch.tensor([1.0], device=latents.device, dtype=torch.float32)
                    dummy_timestep = dummy_timestep.expand(latents.shape[0])
                    
                    latents_req = latents.clone().detach().float()
                    latents_drag = self.drag_estimator.decode(latents_req)
                    
                    t_cond = self.drag_estimator._time_condition(dummy_timestep[:1]).unsqueeze(1)
                    latents_drag = latents_drag + t_cond
        
                    final_drag_coeff = self.drag_estimator.predict_drag(latents_drag)
                    final_drag_coeff_norm = (final_drag_coeff - 0.26325600) / 0.02007368

                    with open(sampling_log_path, 'a') as f:
                        f.write(f"\n=== Final Drag Coefficient ===\n")
                        f.write(f"Normalized: {final_drag_coeff.mean().item():.6f}\n")
                        f.write(f"Denormalized: {final_drag_coeff_norm.mean().item():.6f}\n")
                        f.write("-" * 50 + "\n")
                    
            except Exception as e:
                print(f"Warning: Final drag coefficient estimation failed: {e}")
                with open(sampling_log_path, 'a') as f:
                    f.write(f"\n=== Final Drag Coefficient ===\n")
                    f.write(f"Estimation failed: {e}\n")
                    f.write("-" * 50 + "\n")
        else:
            with open(sampling_log_path, 'a') as f:
                f.write(f"\n=== Final Drag Coefficient ===\n")
                f.write(f"Drag estimator not available\n")
                f.write("-" * 50 + "\n")

        return self._export(
            latents,
            output_type,
            box_v, mc_level, num_chunks, octree_resolution, mc_algo,
            enable_pbar=enable_pbar,
        )

    def _calculate_face_normals(self, vertices, faces):
        """
        Calculate face normals for the mesh
        
        Args:
            vertices: [N, 3] vertex positions
            faces: [M, 3] face indices
            
        Returns:
            normals: [M, 3] face normal vectors (unit vectors)
        """
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]
        

        edge1 = v1 - v0
        edge2 = v2 - v0
        normals = torch.cross(edge1, edge2, dim=1)
        
        normals = normals / (torch.norm(normals, dim=1, keepdim=True) + 1e-8)
        
        return normals

    def _calculate_face_areas(self, vertices, faces):
        """
        Calculate area for each face of the mesh
        
        Args:
            vertices: [N, 3] vertex positions
            faces: [M, 3] face indices
            
        Returns:
            areas: [M] face areas
        """
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]
        
        edge1 = v1 - v0
        edge2 = v2 - v0
        cross_product = torch.cross(edge1, edge2, dim=1)
        
        areas = 0.5 * torch.norm(cross_product, dim=1)
        
        return areas

    def _find_faces_by_direction(self, face_normals, target_direction, angle_threshold_deg=10.0):
        """
        Find faces that are oriented within a certain angle of the target direction
        
        Args:
            face_normals: [M, 3] face normal vectors
            target_direction: [3] target direction vector
            angle_threshold_deg: angle threshold in degrees
            
        Returns:
            mask: [M] boolean mask for faces within the angle threshold
        """
        angle_threshold_rad = np.radians(angle_threshold_deg)
        

        cos_angles = torch.sum(face_normals * target_direction.unsqueeze(0), dim=1)
        

        mask = cos_angles > torch.cos(torch.tensor(angle_threshold_rad, device=face_normals.device))
        
        return mask

    def _calculate_face_pressures(self, vertices, faces, pressure_pred):
        """
        Calculate average pressure for each face based on vertex pressures
        
        Args:
            vertices: [N, 3] vertex positions  
            faces: [M, 3] face indices
            pressure_pred: [1, N, ?] pressure predictions at vertices (? could be 1 or 3)
            
        Returns:
            face_pressures: [M, ?] average pressure for each face
        """

        if pressure_pred.dim() == 3:
            pressure = pressure_pred.squeeze(0)
        elif pressure_pred.dim() == 2:
            pressure = pressure_pred if pressure_pred.shape[0] != 1 else pressure_pred.squeeze(0)
        else:
            pressure = pressure_pred.squeeze(0)

        if pressure.dim() == 1:
            pressure = pressure.unsqueeze(-1)


        pressure = pressure.contiguous().to(torch.float32)

        device = pressure.device
        faces = faces.to(device=device, dtype=torch.long).contiguous()

        N, C = pressure.shape
        idx0 = faces[:, 0].unsqueeze(1).expand(-1, C)
        idx1 = faces[:, 1].unsqueeze(1).expand(-1, C)
        idx2 = faces[:, 2].unsqueeze(1).expand(-1, C)


        p0 = torch.gather(pressure, 0, idx0)
        p1 = torch.gather(pressure, 0, idx1)
        p2 = torch.gather(pressure, 0, idx2)

        face_pressures = (p0 + p1 + p2) / 3.0

        if C == 1:
            face_pressures = face_pressures.squeeze(-1)

        return face_pressures

    def _optimize_aerodynamic_constraints(self, latents, vertices, faces, pressure_pred, 
                                        num_optimization_steps=10, learning_rate=0.01, 
                                        angle_threshold=10.0, optimization_log_path=None, uid=None):
        """
        Optimize latents to satisfy aerodynamic constraints
        Args:
            latents: [B, ...] latent representation
            vertices: [N, 3] mesh vertices
            faces: [M, 3] mesh faces
            pressure_pred: [1, N, 3] initial pressure predictions
            prev_opted_latents: [B, ...] previous optimized latents (for warm start)
            init_drag: initial drag coefficient value for relative optimization (float or tensor)
            num_optimization_steps: number of optimization iterations
            learning_rate: optimization learning rate
            angle_threshold: angle threshold in degrees for face selection
            optimization_log_path: path to save optimization logs
            uid: unique identifier for logging
            
        Returns:
            optimized_latents: optimized latent representation
        """
        
        if optimization_log_path:
            import time
            with open(optimization_log_path, 'a') as f:
                f.write(f"\n=== Starting Aerodynamic Constraint Optimization ===\n")
                f.write(f"Time: {time.strftime('%H:%M:%S')}\n")
                f.write(f"Optimization steps: {num_optimization_steps}\n")
                f.write(f"Learning rate: {learning_rate}\n")
                f.write(f"Angle threshold: {angle_threshold} degrees\n")
                f.write(f"Mesh vertices: {len(vertices)}\n")
                f.write(f"Mesh faces: {len(faces)}\n")
                f.write("-" * 50 + "\n")
        
        vertices_tensor = torch.from_numpy(vertices).to(latents.device, dtype=torch.float32)
        faces_tensor = torch.from_numpy(faces).to(latents.device, dtype=torch.long)

        face_normals = self._calculate_face_normals(vertices_tensor, faces_tensor)
        face_areas = self._calculate_face_areas(vertices_tensor, faces_tensor)
        
        optimized_latents = latents.clone().detach().to(torch.float32).requires_grad_(True)
        
        optimizer = torch.optim.Adam([optimized_latents], lr=learning_rate)
        

        if self.phys_decoder is not None:
            self.phys_decoder.train()

            for param in self.phys_decoder.parameters():
                param.requires_grad_(False)
        

        x_dir = torch.tensor([-1.0, 0.0, 0.0], device=latents.device, dtype=torch.float32)
        y_dir = torch.tensor([0.0, 1.0, 0.0], device=latents.device, dtype=torch.float32)
        z_dir = torch.tensor([0.0, 0.0, 1.0], device=latents.device, dtype=torch.float32)
        
        if optimization_log_path:
            with open(optimization_log_path, 'a') as f:
                f.write(f"Mesh statistics - Vertices: {len(vertices)}, Faces: {len(faces)}, Total face area: {face_areas.sum().item():.4f}\n")
        

        ckpt_originals = {}
        if self.phys_decoder is not None:
            try:
                ckpt_originals = self._set_module_attr_recursive(self.phys_decoder, 'use_checkpoint', False)
            except Exception as _:
                ckpt_originals = {}
        
        def compute_force_integral(face_pressures: torch.Tensor, face_normals: torch.Tensor, 
                                  face_areas: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
            """
            Compute integral of force in given direction: ∫ P * (n · d) * dA
            
            Args:
                face_pressures: [M] or [M, 1] pressure on each face
                face_normals: [M, 3] unit normal vectors for each face
                face_areas: [M] area of each face
                direction: [3] direction vector to integrate in
            
            Returns:
                integral: scalar force integral in the given direction
            """
            if face_pressures.dim() == 2:
                face_pressures = face_pressures.squeeze(-1)
            
            normal_component = torch.sum(face_normals * direction.unsqueeze(0), dim=1)
            
            force_integral = torch.sum(face_pressures * normal_component * face_areas)
            
            return force_integral
        
        with torch.enable_grad():
            for step in range(num_optimization_steps):
                optimizer.zero_grad()


                try:
                    with torch.enable_grad():
                        physics_latents = self.phys_decoder.decode(optimized_latents)
                        phys_points = vertices_tensor.unsqueeze(0)
                        current_pressure = self.phys_decoder.query(phys_points, physics_latents)
                        pressure_std = torch.tensor(self.phys_decoder.cfg.get('PRESSURE_STD', 117.25), 
                                                   device=current_pressure.device, dtype=torch.float32)
                        pressure_mean = torch.tensor(self.phys_decoder.cfg.get('PRESSURE_MEAN', -94.5), 
                                                    device=current_pressure.device, dtype=torch.float32)
                        current_pressure = (pressure_std * current_pressure.to(torch.float32)) + pressure_mean
                except Exception as e:
                    print(f"Warning: Physics decoder forward pass failed: {e}")
                    continue

                face_pressures = self._calculate_face_pressures(vertices_tensor, faces_tensor, current_pressure)

                if self.drag_estimator is not None:
                    try:

                        dummy_timestep = torch.tensor([1.0], device=optimized_latents.device, dtype=torch.float32)
                        dummy_timestep = dummy_timestep.expand(optimized_latents.shape[0])
                        
                        latents_scaled = optimized_latents * self.scale_factor
                        latents_drag = self.drag_estimator.decode(latents_scaled)
                        t_cond = self.drag_estimator._time_condition(dummy_timestep[:1]).unsqueeze(1)
                        latents_drag = latents_drag + t_cond
                        
                        predicted_cd = self.drag_estimator.predict_drag(latents_drag) * 100
                        drag_loss = predicted_cd.mean()
                        drag_force = torch.tensor(0.0, device=optimized_latents.device)
                        
                    except Exception as e:
                        print(f"  Drag estimator failed, fallback to pressure integration: {e}")

                        drag_force = torch.tensor(0.0, device=optimized_latents.device)
                        drag_loss = torch.tensor(0.0, device=optimized_latents.device)
                else:
                    drag_force = torch.tensor(0.0, device=optimized_latents.device)
                    drag_loss = torch.tensor(0.0, device=optimized_latents.device)


                frontback_force = compute_force_integral(face_pressures, face_normals, face_areas, x_dir) * 0.2
                frontback_loss = torch.abs(frontback_force)


                lateral_force = compute_force_integral(face_pressures, face_normals, face_areas, y_dir) * 0.1
                lateral_loss = torch.abs(lateral_force)
                

                lift_force = compute_force_integral(face_pressures, face_normals, face_areas, z_dir) * 0.1
                lift_loss = torch.relu(lift_force)
                
                total_loss = frontback_loss + drag_loss + lateral_loss + lift_loss
                loss_components = {
                    'drag': float(drag_loss.detach()),
                    'frontback': float(frontback_loss.detach()),
                    'lateral': float(lateral_loss.detach()), 
                    'lift': float(lift_loss.detach()),
                    'drag_force': float(drag_force.detach()),
                    'frontback_force': float(frontback_force.detach()),
                    'lateral_force': float(lateral_force.detach()),
                    'lift_force': float(lift_force.detach())
                }
                
                if self.drag_estimator is not None and 'predicted_cd' in locals():
                    log_str = f"Direct Cd optimization | Drag Loss (Cd): {drag_loss.item():.6f}"
                    log_str += f", Front/Back: {frontback_force.item():.4f}"
                    log_str += f", Lateral: {lateral_force.item():.4f}, Lift: {lift_force.item():.4f}"
                else:
                    log_str = f"Pressure integration | Forces - Drag: {drag_force.item():.4f}"
                    log_str += f", Front/Back: {frontback_force.item():.4f}"
                    log_str += f", Lateral: {lateral_force.item():.4f}, Lift: {lift_force.item():.4f}"
                
                if optimization_log_path:
                    with open(optimization_log_path, 'a') as f:
                        f.write(f"{log_str}\n")
                        
                reg = optimized_latents - latents.to(torch.float32)
                regularization_loss = 0.001 * (reg * reg).mean()
                total_loss = total_loss + regularization_loss
                loss_components['regularization'] = float(regularization_loss.detach())


                grad = None
                try:
                    grad = torch.autograd.grad(total_loss, optimized_latents, retain_graph=False, allow_unused=True)[0]
                except Exception as e:
                    print(f"Grad wrt latents failed (first try): {e}")
                    grad = None

                if grad is None:
                    try:
                        optimizer.zero_grad()
                        physics_latents2 = self.phys_decoder.decode(optimized_latents)
                        phys_points2 = vertices_tensor.unsqueeze(0)
                        current_pressure2 = self.phys_decoder.query(phys_points2, physics_latents2)
                        pressure_std2 = torch.tensor(self.phys_decoder.cfg.get('PRESSURE_STD', 117.25), 
                                                     device=current_pressure2.device, dtype=torch.float32)
                        pressure_mean2 = torch.tensor(self.phys_decoder.cfg.get('PRESSURE_MEAN', -94.5), 
                                                      device=current_pressure2.device, dtype=torch.float32)
                        current_pressure2 = (pressure_std2 * current_pressure2.to(torch.float32)) + pressure_mean2
                        face_pressures2 = self._calculate_face_pressures(vertices_tensor, faces_tensor, current_pressure2)
                        

                        if self.drag_estimator is not None:
                            try:
                                dummy_timestep = torch.tensor([1.0], device=optimized_latents.device, dtype=torch.float32)
                                dummy_timestep = dummy_timestep.expand(optimized_latents.shape[0])
                                
                                latents_scaled = optimized_latents * self.scale_factor
                                latents_drag = self.drag_estimator.decode(latents_scaled)
                                t_cond = self.drag_estimator._time_condition(dummy_timestep[:1]).unsqueeze(1)
                                latents_drag = latents_drag + t_cond
                                
                                predicted_cd = self.drag_estimator.predict_drag(latents_drag)
                                drag_loss2 = predicted_cd.mean()
                            except Exception:

                                drag_force2 = compute_force_integral(face_pressures2, face_normals, face_areas, x_dir)
                                drag_loss2 = torch.abs(drag_force2)
                        else:
                            drag_force2 = compute_force_integral(face_pressures2, face_normals, face_areas, x_dir)
                            drag_loss2 = torch.abs(drag_force2)
                        
                        lateral_force2 = compute_force_integral(face_pressures2, face_normals, face_areas, y_dir)
                        lateral_loss2 = torch.tensor(0.0)
                        
                        lift_force2 = compute_force_integral(face_pressures2, face_normals, face_areas, z_dir)
                        lift_loss2 = torch.tensor(0.0)
                        
                        total_loss2 = drag_loss2 + lateral_loss2 + lift_loss2
                        
                        reg2 = optimized_latents - latents.to(torch.float32)
                        total_loss2 = total_loss2 + 0.001 * (reg2 * reg2).mean()
                        
                        grad = torch.autograd.grad(total_loss2, optimized_latents, retain_graph=False, allow_unused=True)[0]
                    except Exception as e2:
                        print(f"Second grad attempt failed: {e2}")
                        grad = None

                if grad is None or not torch.isfinite(grad).all():
                    print("Warning: invalid grad; skipping step")
                    continue

                optimized_latents.grad = grad
                optimizer.step()
                


                with torch.no_grad():
                    try:
                        meshes = self.extract_geometry_by_diffdmc(
                            optimized_latents * self.scale_factor, 
                            resolution=128,
                            batch_size=10000,
                            verbose=False
                        )
                        
                        if meshes[0] is not None:

                            new_vertices = meshes[0].vertices.astype(np.float32)
                            new_faces = meshes[0].faces.astype(np.int32)
                            
                            vertices_tensor = torch.from_numpy(new_vertices).to(latents.device, dtype=torch.float32)
                            faces_tensor = torch.from_numpy(new_faces).to(latents.device, dtype=torch.long)
                            

                            face_normals = self._calculate_face_normals(vertices_tensor, faces_tensor)
                            face_areas = self._calculate_face_areas(vertices_tensor, faces_tensor)
                            

                            if step % 2 == 0 and optimization_log_path is not None and False:
                                try:
                                    from datetime import datetime
                                    import pytz
                                    zurich_tz = pytz.timezone('Europe/Zurich')
                                    zurich_time = datetime.now(zurich_tz)
                                    timestamp = zurich_time.strftime("%Y%m%d_%H%M%S")
        
                                    import trimesh
                                    mesh_obj = trimesh.Trimesh(vertices=new_vertices, faces=new_faces)
                                    
                                    log_dir = os.path.dirname(optimization_log_path)
                                    optimization_log_basename = os.path.splitext(os.path.basename(optimization_log_path))[0]
                                    mesh_filename = f"{optimization_log_basename}_{timestamp}_step{step:03d}.obj"
                                    mesh_filepath = os.path.join(log_dir, mesh_filename)
                                    
                                    mesh_obj.export(mesh_filepath)
                                    
                                    with open(optimization_log_path, 'a') as f:
                                        f.write(f"Step {step}: Saved mesh to {mesh_filename} - Vertices: {len(new_vertices)}, Faces: {len(new_faces)}, Total area: {face_areas.sum().item():.4f}\n")
                                        
                                except ImportError:
                                    print(f"  Step {step}: Warning - trimesh not available, cannot save mesh")
                                except Exception as save_error:
                                    print(f"  Step {step}: Warning - Failed to save mesh: {save_error}")
                        else:
                            print(f"  Warning: Step {step}: Failed to extract mesh, keeping previous geometry")
                            
                    except Exception as mesh_error:
                        print(f"  Warning: Step {step}: Mesh extraction failed: {mesh_error}")

                if step % 2 == 0 or step == num_optimization_steps - 1:
                    log_str = f"Step {step:2d}: Total Loss = {float(total_loss.detach()):.6f}"
                    log_str += f", drag_loss = {loss_components['drag']:.6f}"
                    log_str += f", frontback_loss = {loss_components['frontback']:.6f}"
                    log_str += f", lateral_loss = {loss_components['lateral']:.6f}" 
                    log_str += f", lift_loss = {loss_components['lift']:.6f}"
                    log_str += f" | Forces: drag = {loss_components['drag_force']:.4f}"
                    log_str += f", front/back = {loss_components['frontback_force']:.4f}"
                    log_str += f", lateral = {loss_components['lateral_force']:.4f}"
                    log_str += f", lift = {loss_components['lift_force']:.4f}"
                    
                    if self.drag_estimator is not None:
                        try:
                            with torch.inference_mode(False), torch.cuda.amp.autocast(enabled=False):
                                dummy_timestep = torch.tensor([1.0], device=optimized_latents.device, dtype=torch.float32)
                                dummy_timestep = dummy_timestep.expand(optimized_latents.shape[0])
                                
                                latents_req = optimized_latents.clone().detach().float() * self.scale_factor
                                latents_drag = self.drag_estimator.decode(latents_req)
                                
                                t_cond = self.drag_estimator._time_condition(dummy_timestep[:1]).unsqueeze(1)
                                latents_drag = latents_drag + t_cond
                    
                                drag_coeff = self.drag_estimator.predict_drag(latents_drag)
                                log_str += f" | Drag Coeff: {drag_coeff.mean().item():.6f}"
                                
                        except Exception as e:
                            print(f"Warning: Drag coefficient estimation failed: {e}")
                            log_str += " | Drag Coeff: N/A"
                    
                    if optimization_log_path:
                        with open(optimization_log_path, 'a') as f:
                            f.write(f"{log_str}\n")

        if self.phys_decoder is not None and ckpt_originals:
            try:
                self._restore_module_attrs(self.phys_decoder, 'use_checkpoint', ckpt_originals)
            except Exception as _:
                pass
        
        if self.phys_decoder is not None:
            self.phys_decoder.eval()
        
        final_drag_coeff = None
        if self.drag_estimator is not None:
            try:
                with torch.inference_mode(False), torch.cuda.amp.autocast(enabled=False):
                    dummy_timestep = torch.tensor([1.0], device=optimized_latents.device, dtype=torch.float32)
                    dummy_timestep = dummy_timestep.expand(optimized_latents.shape[0])
                    
                    latents_req = optimized_latents.clone().detach().float() * self.scale_factor
                    latents_drag = self.drag_estimator.decode(latents_req)
                    
                    t_cond = self.drag_estimator._time_condition(dummy_timestep[:1]).unsqueeze(1)
                    latents_drag = latents_drag + t_cond
        
                    drag_coeff = self.drag_estimator.predict_drag(latents_drag)
                    final_drag_coeff = drag_coeff
                    
            except Exception as e:
                print(f"Warning: Final drag coefficient estimation failed: {e}")
                final_drag_coeff = None
        
        if optimization_log_path:
            with open(optimization_log_path, 'a') as f:
                f.write(f"\n=== Optimization Completed ===\n")
                f.write(f"Time: {time.strftime('%H:%M:%S')}\n")
                if final_drag_coeff is not None:
                    f.write(f"Final drag coefficient: {final_drag_coeff.mean().item():.6f}\n")
                else:
                    f.write(f"Final drag coefficient: N/A (estimation failed)\n")
                f.write("-" * 50 + "\n")
        
        return optimized_latents.detach()

    def _set_module_attr_recursive(self, module: torch.nn.Module, attr: str, value):
        """Recursively set attribute on module and all submodules if present, return dict of originals."""
        originals = {}
        if hasattr(module, attr):
            originals[id(module)] = getattr(module, attr)
            setattr(module, attr, value)
        for child in module.children():
            originals.update(self._set_module_attr_recursive(child, attr, value))
        return originals

    def _restore_module_attrs(self, module: torch.nn.Module, attr: str, originals: dict):
        """Restore attributes saved by _set_module_attr_recursive."""
        if id(module) in originals:
            setattr(module, attr, originals[id(module)])
        for child in module.children():
            self._restore_module_attrs(child, attr, originals)

    def _save_directional_meshes(self, vertices, faces, direction_masks, optimization_log_path=None, uid=None):
        """
        Save meshes for each directional face group
        
        Args:
            vertices: [N, 3] tensor of vertex positions
            faces: [M, 3] tensor of face indices  
            direction_masks: dict with keys like 'front', 'back', etc. and boolean mask values
            optimization_log_path: path for logging
            uid: unique identifier for filename prefix
        """
        try:
            import trimesh
            import numpy as np
            
            vertices_np = vertices.detach().cpu().numpy()
            faces_np = faces.detach().cpu().numpy()
            
            if optimization_log_path:
                save_dir = os.path.dirname(optimization_log_path)
            else:
                save_dir = "./directional_meshes"
            
            os.makedirs(save_dir, exist_ok=True)
            
            saved_files = []
            
            for direction, mask in direction_masks.items():
                if mask.sum() == 0:
                    print(f"Warning: No faces found for direction '{direction}', skipping...")
                    continue
                
                face_indices = torch.where(mask)[0].detach().cpu().numpy()
                direction_faces = faces_np[face_indices]
                
                unique_vertices = np.unique(direction_faces.flatten())
                
                new_vertices = vertices_np[unique_vertices]
                vertex_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_vertices)}
                
                new_faces = np.array([[vertex_mapping[face[0]], vertex_mapping[face[1]], vertex_mapping[face[2]]] 
                                     for face in direction_faces])
                
                try:
                    mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces)
                    
                    if uid is not None:
                        clean_uid = str(uid).replace('/', '_').replace('\\', '_').replace(':', '_')
                        filename = f"{clean_uid}_{direction}_faces_3.obj"
                    else:
                        filename = f"{direction}_faces_3.obj"
                    
                    filepath = os.path.join(save_dir, filename)
                    mesh.export(filepath)
                    
                    saved_files.append(filepath)
                    
                except Exception as e:
                    print(f"Failed to create/save mesh for direction '{direction}': {e}")
                    continue
            
            if optimization_log_path and saved_files:
                with open(optimization_log_path, 'a') as f:
                    f.write(f"\nSaved directional meshes:\n")
                    for filepath in saved_files:
                        f.write(f"  - {filepath}\n")
            
        except ImportError:
            print("Warning: trimesh not available, cannot save directional meshes")
        except Exception as e:
            print(f"Error saving directional meshes: {e}")
            if optimization_log_path:
                with open(optimization_log_path, 'a') as f:
                    f.write(f"Error saving directional meshes: {e}\n")
