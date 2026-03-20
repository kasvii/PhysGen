import os
import sys
import torch
import argparse
import shutil
import contextlib

from physdec.utils.config import ExperimentConfig, load_config

from omegaconf import OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.utilities.rank_zero import rank_zero_only


from craftsman import find
from craftsman.systems.base import BaseSystem
from physdec.pressure_net import PressureEstimator
from physdec.drivarnetplus import DrivarNetPlusDataModule
from craftsman.utils.callbacks import (
    CodeSnapshotCallback,
    ConfigSnapshotCallback,
    CustomProgressBar,
    ProgressCallback,
)
from craftsman.models.autoencoders.utils import FourierEmbedder



from craftsman.utils.misc import get_rank


def build_trainer(cfg, loggers, callbacks):
    resume_path = cfg.get("resume_from_checkpoint", None)

    trainer = pl.Trainer(
        num_nodes=cfg.trainer.num_nodes,
        max_epochs=cfg.trainer.max_epochs,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        num_sanity_val_steps=cfg.trainer.num_sanity_val_steps,
        check_val_every_n_epoch=cfg.trainer.check_val_every_n_epoch,
        enable_progress_bar=cfg.trainer.enable_progress_bar,
        precision=cfg.trainer.precision,
        callbacks=callbacks,
        logger=loggers,
    )

    return trainer

def build_model(cfg):
    encoder_cls = find(cfg.system.shape_model_type)
    shape_model_cfg = OmegaConf.structured(encoder_cls.Config(**cfg.system.shape_model))
    autoencoder = encoder_cls(shape_model_cfg)
    
    ckpt_path = cfg.system.shape_model.pretrained_model_name_or_path
    pretrained_ckpt = torch.load(ckpt_path, map_location="cpu")
    _pretrained_ckpt = {}
    for k, v in pretrained_ckpt['state_dict'].items():
        if k.startswith('shape_model.'):
            _pretrained_ckpt[k.replace('shape_model.', '')] = v
    pretrained_ckpt = _pretrained_ckpt
    autoencoder.load_state_dict(pretrained_ckpt, strict=True)
    
    encoder = autoencoder.encoder
    encoder.requires_grad_(False)
    pre_kl = autoencoder.pre_kl
    pre_kl.requires_grad_(False)
    embedder = autoencoder.embedder

    model = PressureEstimator(
        encoder=encoder,
        embedder=embedder,
        pre_kl=pre_kl,
        cfg=cfg.system.phys_model,
        out_dim=cfg.system.phys_model.out_dim,
        num_latents=cfg.system.phys_model.num_latents,
        width=cfg.system.phys_model.width,
        heads=cfg.system.phys_model.heads,
        init_scale=cfg.system.phys_model.init_scale,
        qkv_bias=cfg.system.phys_model.qkv_bias,
        use_flash=cfg.system.phys_model.use_flash,
        use_checkpoint=cfg.system.phys_model.use_checkpoint,
        vis_mesh=cfg.data.vis_mesh,
    )
    
    model.set_save_dir(os.path.join(cfg.trial_dir, "save"))

    return model


def main(args, extras):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env_gpus_str = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    env_gpus = list(env_gpus_str.split(",")) if env_gpus_str else []
    selected_gpus = [0]
    torch.set_float32_matmul_precision("high")

    devices = -1
    if len(env_gpus) > 0:
        n_gpus = len(env_gpus)
    else:
        selected_gpus = list(args.gpu.split(","))
        n_gpus = len(selected_gpus)
        print(f"Using {n_gpus} GPUs: {selected_gpus}")
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        
    cfg: ExperimentConfig
    cfg = load_config(args.config, cli_args=extras, n_gpus=n_gpus)

    pl.seed_everything(cfg.seed + get_rank()+1, workers=True)
    
    datamodule = DrivarNetPlusDataModule(cfg.data)
    
    callbacks = []
    if args.train:
        callbacks += [
            ModelCheckpoint(
                dirpath=os.path.join(cfg.trial_dir, "ckpts"), **cfg.checkpoint
            ),
            LearningRateMonitor(logging_interval="step"),
            CodeSnapshotCallback(
                os.path.join(cfg.trial_dir, "code"), use_version=False
            ),
            ConfigSnapshotCallback(
                args.config,
                cfg,
                os.path.join(cfg.trial_dir, "configs"),
                use_version=False,
            ),
        ]
        callbacks += [CustomProgressBar(refresh_rate=1)]

    def write_to_text(file, lines):
        with open(file, "w") as f:
            for line in lines:
                f.write(line + "\n")

    loggers = []
    if args.train:
        rank_zero_only(
            lambda: os.makedirs(os.path.join(cfg.trial_dir, "tb_logs"), exist_ok=True)
        )()
        loggers += [
            TensorBoardLogger(cfg.trial_dir, name="tb_logs"),
            CSVLogger(cfg.trial_dir, name="csv_logs"),
        ]
        rank_zero_only(
            lambda: write_to_text(
                os.path.join(cfg.trial_dir, "cmd.txt"),
                ["python " + " ".join(sys.argv), str(args)],
            )
        )()
    
    if not os.path.exists(os.path.join(cfg.trial_dir, "code")):
        shutil.copytree('./physdec', os.path.join(cfg.trial_dir, "code"))
        shutil.copy2('./launch_physdec.py', os.path.join(cfg.trial_dir, "code", "launch_physdec.py"))
    
    model = build_model(cfg)
    
    trainer = build_trainer(cfg, loggers, callbacks)

    if args.train:
        trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.resume)
        trainer.test(model, datamodule=datamodule)
    elif args.validate:
        trainer.validate(model, datamodule=datamodule, ckpt_path=cfg.resume)
    elif args.test:
        trainer.test(model, datamodule=datamodule, ckpt_path=cfg.resume)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config file")

    parser.add_argument(
        "--gpu",
        default="0",
        help="GPU(s) to use, e.g., 0 or 0,1,2",
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--train", action="store_true")
    group.add_argument("--validate", action="store_true")
    group.add_argument("--test", action="store_true")

    args, extras = parser.parse_known_args()

    with contextlib.redirect_stdout(sys.stderr):
        main(args, extras)
