# This code is modified based on Hunyuan 3D. Please comply with the license terms of Hunyuan 3D.
# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

from typing import Tuple, List
import warnings
warnings.filterwarnings("ignore")

import os
import torch
import argparse
import shutil
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
from einops._torch_specific import allow_ops_in_compiled_graph
allow_ops_in_compiled_graph()

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.strategies import DDPStrategy, DeepSpeedStrategy
from pytorch_lightning.loggers import Logger, TensorBoardLogger
from pytorch_lightning.utilities import rank_zero_info

from condition_diffusion.utils import get_config_from_file, instantiate_from_config

class NaNDetectionCallback(Callback):
    """Callback to detect and handle NaN values during training"""
    
    def __init__(self):
        super().__init__()
        self.nan_count = 0
        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Check for NaN in loss after each batch"""
        if outputs is not None and 'loss' in outputs:
            loss = outputs['loss']
            if torch.isnan(loss).any():
                self.nan_count += 1
                rank_zero_info(f"❌ NaN detected in loss at batch {batch_idx}!")
                rank_zero_info(f"Loss value: {loss}")
                
                nan_params = []
                for name, param in pl_module.named_parameters():
                    if param is not None and torch.isnan(param).any():
                        nan_params.append(name)
                
                if nan_params:
                    rank_zero_info(f"❌ Parameters with NaN: {nan_params}")
                
                nan_grads = []
                for name, param in pl_module.named_parameters():
                    if param.grad is not None and torch.isnan(param.grad).any():
                        nan_grads.append(name)
                
                if nan_grads:
                    rank_zero_info(f"❌ Gradients with NaN: {nan_grads}")
                
                if self.nan_count > 5:
                    rank_zero_info("❌ Too many NaN occurrences, stopping training!")
                    trainer.should_stop = True

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'condition_diffusion'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'craftsman'))

import craftsman
craftsman.clear_registry()


class SetupCallback(Callback):
    def __init__(self, config: DictConfig, basedir: Path, logdir: str = "log", ckptdir: str = "ckpt") -> None:
        super().__init__()
        self.logdir = basedir / logdir
        self.ckptdir = basedir / ckptdir
        self.config = config

    def on_fit_start(self, trainer: pl.trainer.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.global_rank == 0:
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)


def setup_callbacks(config: DictConfig) -> Tuple[List[Callback], Logger]:
    training_cfg = config.training
    basedir = Path(training_cfg.output_dir)
    os.makedirs(basedir, exist_ok=True)
    all_callbacks = []

    setup_callback = SetupCallback(config, basedir)
    all_callbacks.append(setup_callback)
    
    nan_callback = NaNDetectionCallback()
    all_callbacks.append(nan_callback)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=setup_callback.ckptdir,
        filename="ckpt-{step:08d}",
        monitor=training_cfg.monitor,
        mode="max",
        save_top_k=-1,
        save_last=True,
        verbose=False,
        every_n_train_steps=training_cfg.every_n_train_steps)
    all_callbacks.append(checkpoint_callback)

    if "callbacks" in config:
        for key, value in config['callbacks'].items():
            custom_callback = instantiate_from_config(value)
            all_callbacks.append(custom_callback)

    logger = TensorBoardLogger(save_dir=str(setup_callback.logdir), name="tensorboard")

    return all_callbacks, logger


def merge_cfg(cfg, arg_cfg):
    for key in arg_cfg.keys():
        if key in cfg.training:
            arg_cfg[key] = cfg.training[key]
    cfg.training = DictConfig(arg_cfg)
    return cfg


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action='store_true')
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument("-nn", "--num_nodes", type=int, default=1)
    parser.add_argument("-ng", "--num_gpus", type=int, default=1)
    parser.add_argument("-u", "--update_every", type=int, default=1)
    parser.add_argument("-st", "--steps", type=int, default=50000000)
    parser.add_argument("-lr", "--base_lr", type=float, default=4.5e-6)
    parser.add_argument("-a", "--use_amp", default=False, action="store_true")
    parser.add_argument("--amp_type", type=str, default="16")
    parser.add_argument("--gradient_clip_val", type=float, default=None)
    parser.add_argument("--gradient_clip_algorithm", type=str, default=None)
    parser.add_argument("--every_n_train_steps", type=int, default=50000)
    parser.add_argument("--log_every_n_steps", type=int, default=50)
    parser.add_argument("--val_check_interval", type=int, default=1024)
    parser.add_argument("--limit_val_batches", type=int, default=64)
    parser.add_argument("--monitor", type=str, default="val/total_loss")
    parser.add_argument("--output_dir", type=str, help="the output directory to save everything.")
    parser.add_argument("--ckpt_path", type=str, default="", help="the restore checkpoints.")
    parser.add_argument("--deepspeed", default=False, action="store_true")
    parser.add_argument("--deepspeed2", default=False, action="store_true")
    parser.add_argument("--scale_lr", type=bool, nargs="?", const=True, default=False,
                        help="scale base-lr by ngpu * batch_size * n_accumulate")
    parser.add_argument("--test", action='store_true', 
                        help="Run test mode to generate meshes using pretrained models")
    parser.add_argument("--test_batches", type=int, default=3, 
                        help="Number of test batches to generate")
    return parser.parse_args()
    

if __name__ == "__main__":
    
    args = get_args()
    
    if args.fast:
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision('medium')
        torch.utils.data._utils.MP_STATUS_CHECK_INTERVAL = 0.05
    
    pl.seed_everything(args.seed, workers=True)

    config = get_config_from_file(args.config)
    config = merge_cfg(config, vars(args))
    training_cfg = config.training

    if args.test:
        rank_zero_info("="*60)
        rank_zero_info("RUNNING IN TEST MODE - GENERATING MESHES")
        rank_zero_info("="*60)
        
        data: pl.LightningDataModule = instantiate_from_config(config.dataset)
        
        model: pl.LightningModule = instantiate_from_config(config.model)
        
        if config.model.get('resume_from_checkpoint'):
            checkpoint_path = config.model.resume_from_checkpoint
            rank_zero_info(f"Loading model from checkpoint: {checkpoint_path}")
            
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            rank_zero_info("Model loaded successfully!")
        
        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()
        data.setup('test')
        test_dataloader = data.test_dataloader()
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_dataloader):
                if batch_idx >= args.test_batches:
                    break
                    
                rank_zero_info(f"\n--- Test Batch {batch_idx + 1}/{args.test_batches} ---")
                
                device = next(model.parameters()).device
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(device)
                
                try:
                    result = model.test_step(batch, batch_idx)
                except Exception as e:
                    import traceback
                    rank_zero_info(f"Test batch {batch_idx} failed: {e}")
                    rank_zero_info(f"Traceback: {traceback.format_exc()}")
        
        rank_zero_info("="*60)
        rank_zero_info("TEST MODE COMPLETED")
        rank_zero_info("="*60)
        exit(0)

    callbacks, loggers = setup_callbacks(config)

    data: pl.LightningDataModule = instantiate_from_config(config.dataset)

    model: pl.LightningModule = instantiate_from_config(config.model)
    
    nodes = args.num_nodes
    ngpus = args.num_gpus
    base_lr = training_cfg.base_lr
    accumulate_grad_batches = training_cfg.update_every
    batch_size = config.dataset.params.batch_size

    if 'NNODES' in os.environ:
        nodes = int(os.environ['NNODES'])
        training_cfg.num_nodes = nodes
        args.num_nodes = nodes

    if args.scale_lr:
        model.learning_rate = accumulate_grad_batches * nodes * ngpus * batch_size * base_lr
    else:
        model.learning_rate = base_lr

    if args.num_nodes > 1 or args.num_gpus > 1:
        if args.deepspeed:
            ddp_strategy = DeepSpeedStrategy(stage=1)
        elif args.deepspeed2:
            ddp_strategy = 'deepspeed_stage_2'
        else:
            ddp_strategy = DDPStrategy(find_unused_parameters=False, bucket_cap_mb=1500)
    else:
        ddp_strategy = 'auto'

    if training_cfg.use_amp and not training_cfg.get('force_fp32', False):
        amp_type = training_cfg.amp_type
        assert amp_type in ['bf16', '16', '32'], f"Invalid amp_type: {amp_type}"
    else:
        amp_type = 32

    trainer = pl.Trainer(
        max_steps=training_cfg.steps,
        precision=amp_type,
        callbacks=callbacks,
        accelerator="gpu",
        devices=args.num_gpus,
        num_nodes=training_cfg.num_nodes,
        strategy=ddp_strategy,
        gradient_clip_val=training_cfg.get('gradient_clip_val'),
        gradient_clip_algorithm=training_cfg.get('gradient_clip_algorithm'),
        accumulate_grad_batches=args.update_every,
        logger=loggers,
        log_every_n_steps=training_cfg.log_every_n_steps,
        val_check_interval=training_cfg.val_check_interval,
        limit_val_batches=training_cfg.limit_val_batches,
        check_val_every_n_epoch=None,
        detect_anomaly=True,
        enable_progress_bar=True,
        enable_model_summary=True
    )

    if training_cfg.ckpt_path == '': 
        training_cfg.ckpt_path = None
    trainer.fit(model, datamodule=data, ckpt_path=training_cfg.ckpt_path)
