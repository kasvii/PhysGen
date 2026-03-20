# -*- coding: utf-8 -*-

import importlib
from omegaconf import OmegaConf, DictConfig, ListConfig

import torch
import torch.distributed as dist
from typing import Union


def get_config_from_file(config_file: str) -> Union[DictConfig, ListConfig]:
    config_file = OmegaConf.load(config_file)

    if 'base_config' in config_file.keys():
        if config_file['base_config'] == "default_base":
            base_config = OmegaConf.create()
        elif config_file['base_config'].endswith(".yaml"):
            base_config = get_config_from_file(config_file['base_config'])
        else:
            raise ValueError(f"{config_file} must be `.yaml` file or it contains `base_config` key.")

        config_file = {key: value for key, value in config_file if key != "base_config"}

        return OmegaConf.merge(base_config, config_file)

    return config_file


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def get_obj_from_config(config):
    target_key = "target" if "target" in config else "_target_"
    if target_key not in config:
        raise KeyError("Expected key `target` or `_target_` to instantiate.")

    return get_obj_from_str(config[target_key])


def instantiate_from_config(config, **kwargs):
    target_key = "target" if "target" in config else "_target_"
    if target_key not in config:
        raise KeyError("Expected key `target` or `_target_` to instantiate.")

    cls = get_obj_from_str(config[target_key])

    if config.get("from_pretrained", None):
        return cls.from_pretrained(
                    config["from_pretrained"], 
                    use_safetensors=config.get('use_safetensors', False),
                    variant=config.get('variant', 'fp16'),
                    **kwargs)

    # Get parameters - either from 'params' field or all config fields except target
    if "params" in config:
        params = config["params"]
    else:
        params = {k: v for k, v in config.items() if k != target_key}
    
    # Check if the class constructor expects a single 'cfg' parameter
    import inspect
    sig = inspect.signature(cls.__init__)
    param_names = list(sig.parameters.keys())
    
    if len(param_names) == 2 and 'cfg' in param_names:
        instance = cls(cfg=params)
    else:
        kwargs.update(params)
        instance = cls(**kwargs)

    return instance


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


def instantiate_non_trainable_model(config):
    # Special handling for MichelangeloAutoencoder to ensure cfg parameter is passed correctly
    if config.get('_target_') == 'craftsman.models.autoencoders.michelangelo_autoencoder.MichelangeloAutoencoder':
        config_copy = config.copy()
        if 'cfg' not in config_copy:
            cfg_params = {k: v for k, v in config_copy.items() if k != '_target_'}
            config_copy = {'_target_': config_copy['_target_'], 'cfg': cfg_params}
        
        model = instantiate_from_config(config_copy)
    else:
        model = instantiate_from_config(config)
    
    # Convert to FP32 for stability and consistency with diffusion model
    # This must be done AFTER configure() which loads pretrained weights
    model = model.float()
    model = model.eval()
    model.train = disabled_train
    for param in model.parameters():
        param.requires_grad = False
    
    for name, param in model.named_parameters():
        if param.dtype != torch.float32:
            print(f"Warning: Converting {name} from {param.dtype} to float32")
            param.data = param.data.float()

    return model


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def all_gather_batch(tensors):
    """
    Performs all_gather operation on the provided tensors.
    """
    world_size = get_world_size()
    # There is no need for reduction in the single-proc case
    if world_size == 1:
        return tensors
    tensor_list = []
    output_tensor = []
    for tensor in tensors:
        tensor_all = [torch.ones_like(tensor) for _ in range(world_size)]
        dist.all_gather(
            tensor_all,
            tensor,
            async_op=False
        )

        tensor_list.append(tensor_all)

    for tensor_all in tensor_list:
        output_tensor.append(torch.cat(tensor_all, dim=0))
    return output_tensor
