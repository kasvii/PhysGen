#!/usr/bin/env python3
"""
Comprehensive launch script for multi-task fine-tuning
Supports direct joint training of all components using pretrained models
- Uses pretrained shape model checkpoint
- Uses pretrained physics model checkpoint  
- Uses pretrained drag model checkpoint
- Performs joint fine-tuning to optimize all components together
"""
import argparse
import os
import sys
import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, 'finetune_all'))

import craftsman
from craftsman.utils.config import load_config
from craftsman.utils.misc import get_rank
from craftsman.utils.callbacks import (
    ConfigSnapshotCallback,
    ProgressCallback,
)

from finetune_all.comprehensive_system import ComprehensiveMultiTaskSystem


def update_config_for_task(cfg, task):
    """Update configuration for specific task"""
    cfg.system.current_task = task
    
    if task == "shape":
        cfg.checkpoint.monitor = "val/shape_iou"
        cfg.checkpoint.mode = "max"
    elif task == "physics":
        cfg.checkpoint.monitor = "val/physics_mse"
        cfg.checkpoint.mode = "min"
    elif task == "drag":
        cfg.checkpoint.monitor = "val/drag_mse"
        cfg.checkpoint.mode = "min"
    elif task == "joint":
        cfg.checkpoint.monitor = "val/loss"
        cfg.checkpoint.mode = "min"
    
    cfg.tag = f"{cfg.tag}_{task}"
    
    return cfg


def setup_test_visualization(cfg, args):
    """Pick the test samples to visualize (stratified over car categories unless uids are given)
    and tell the system / datamodule where to find the GT surfaces and where to save outputs."""
    import json
    from omegaconf import open_dict
    from finetune_all.visualization import select_vis_uids, uid_category, uid_from_path

    num = args.vis_num_samples if args.vis_num_samples is not None else cfg.system.get("vis_num_samples", 0)
    explicit = args.vis_uids.split(",") if args.vis_uids else list(cfg.system.get("vis_uids", []))
    explicit = [uid_from_path(u.strip()) for u in explicit if u.strip()]
    if num <= 0 and not explicit:
        return

    with open(os.path.join(cfg.data.shape_root_dir, "test.json")) as f:
        test_uids = [uid_from_path(p) for p in json.load(f)]
    selected = explicit or select_vis_uids(test_uids, num, seed=cfg.system.get("vis_seed", 0))
    missing = sorted(set(selected) - set(test_uids))
    if missing:
        print(f"[vis] warning: not in the test split, will not be visualized: {missing}")

    vis_dir = cfg.system.get("vis_dir") or os.path.join(cfg.trial_dir, "vis")
    os.makedirs(vis_dir, exist_ok=True)
    with open_dict(cfg):
        cfg.system.vis_uids = selected
        cfg.system.vis_dir = vis_dir
        cfg.system.vis_pressure_dir = cfg.system.get("vis_pressure_dir") or cfg.data.pressure_path
        if args.vis_only:
            cfg.data.test_uids = selected
    with open(os.path.join(vis_dir, "selected_uids.json"), "w") as f:
        json.dump([{"uid": u, "category": uid_category(u)} for u in selected], f, indent=2)
    print(f"[vis] {len(selected)} test samples -> {vis_dir}")
    for u in selected:
        print(f"[vis]   {u}")


def train_single_task(cfg, task, config_path, resume_from=None, test_mode=False):
    """Train a single task"""
    print(f"\n{'='*50}")
    if test_mode:
        print(f"Testing {task.upper()} task")
    else:
        print(f"Training {task.upper()} task")
    print(f"{'='*50}")
    
    cfg = update_config_for_task(cfg, task)
    
    if task == "shape":
        dm = craftsman.find("objaverse-datamodule")(cfg.data)
    elif task == "physics":
        sys.path.append(os.path.join(os.path.dirname(current_dir), 'physdec'))
        from drivarnetplus import DrivAerNetPlusDataModule
        dm = DrivAerNetPlusDataModule(cfg.data)
    elif task == "drag":
        sys.path.append(os.path.join(os.path.dirname(current_dir), 'dragdec'))
        from drag_dataset import DragDataModule
        dm = DragDataModule(cfg.data)
    else:
        if cfg.data_type == "multitask-datamodule":
            sys.path.append(os.path.join(current_dir, 'finetune_all'))
            from multitask_dataset import MultiTaskDataModule
            dm = MultiTaskDataModule(cfg.data)
        else:
            dm = craftsman.find("objaverse-datamodule")(cfg.data)
    
    system = ComprehensiveMultiTaskSystem(cfg.system)
    
    import shutil
    
    if not test_mode:
        if not os.path.exists(os.path.join(cfg.trial_dir, "finetune_all")):
            finetune_all_src = os.path.join(current_dir, "finetune_all")
            if os.path.exists(finetune_all_src):
                shutil.copytree(finetune_all_src, os.path.join(cfg.trial_dir, "finetune_all"))
        
        launch_script_dst = os.path.join(cfg.trial_dir, "finetune_all")
        os.makedirs(launch_script_dst, exist_ok=True)
        shutil.copy2(__file__, os.path.join(launch_script_dst, "launch_finetuneall.py"))
        shutil.copy2("commands/train_totalvae_single_node.sh", os.path.join(launch_script_dst, "train_totalvae_single_node.sh"))
        shutil.copy2("configs/finetune_all/MultiTaskJoint.yaml", os.path.join(launch_script_dst, "MultiTaskJoint.yaml"))
    
    callbacks = []
    
    if not test_mode:
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(cfg.trial_dir, "ckpts"),
            filename=f"{task}_epoch={{epoch:04d}}-step={{step:07d}}",
            monitor=cfg.checkpoint.monitor,
            mode=cfg.checkpoint.mode,
            save_top_k=cfg.checkpoint.save_top_k,
            save_last=True,
            every_n_epochs=cfg.checkpoint.every_n_epochs,
            save_weights_only=False,
            auto_insert_metric_name=False,
        )
        callbacks.append(checkpoint_callback)
    
    if get_rank() == 0:
        lr_monitor = LearningRateMonitor(logging_interval="step")
        callbacks.append(lr_monitor)
    
    loggers = []
    if get_rank() == 0 and not test_mode:
        csv_logger = CSVLogger(cfg.trial_dir, name=f"{task}_logs", version="")
        loggers.append(csv_logger)
        
        tb_logger = TensorBoardLogger(cfg.trial_dir, name=f"{task}_tb", version="")
        loggers.append(tb_logger)
    
    trainer_config = cfg.get("trainer", {})
    
    if test_mode:
        trainer = Trainer(
            devices="auto",
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            strategy="ddp_find_unused_parameters_true" if torch.cuda.device_count() > 1 else "auto",
            precision=trainer_config.get("precision", "16-mixed"),
            enable_model_summary=False,
            callbacks=callbacks,
            logger=loggers,
        )
    else:
        trainer = Trainer(
            devices="auto",
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            strategy="ddp_find_unused_parameters_true",
            precision=trainer_config.get("precision", "16-mixed"),
            max_epochs=trainer_config.get("max_epochs", 500),
            check_val_every_n_epoch=trainer_config.get("check_val_every_n_epoch", 10),
            log_every_n_steps=trainer_config.get("log_every_n_steps", 50),
            gradient_clip_val=trainer_config.get("gradient_clip_val", 1.0),
            accumulate_grad_batches=trainer_config.get("accumulate_grad_batches", 4),
            enable_model_summary=False,
            callbacks=callbacks,
            logger=loggers,
        )
    
    try:
        if test_mode:
            trainer.test(
                system,
                datamodule=dm,
                ckpt_path=resume_from
            )
            return None
        else:
            trainer.fit(
                system,
                datamodule=dm,
                ckpt_path=resume_from
            )
            
            best_ckpt_path = checkpoint_callback.best_model_path
            return best_ckpt_path
        
    except Exception as e:
        print(f"{task} task {'testing' if test_mode else 'training'} failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise


def main():
    parser = argparse.ArgumentParser(description="Comprehensive multi-task training launcher")
    parser.add_argument("--config", type=str, 
                       default="configs/finetune_all/ComprehensiveMultiTask.yaml",
                       help="Path to config file")
    parser.add_argument("--gpu", type=str, default="0", help="GPU devices to use")
    parser.add_argument("--task", type=str, default="sequential", 
                       choices=["shape", "physics", "drag", "joint", "sequential"],
                       help="Task to run: 'joint' for direct multi-task training, 'sequential' for joint training (skips individual tasks), or single task name")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--test", action="store_true", help="Run in test/validation mode")
    parser.add_argument("--vis_num_samples", type=int, default=None,
                        help="[test] number of test samples to visualize, stratified over car categories "
                             "(default: system.vis_num_samples in the config; 0 disables)")
    parser.add_argument("--vis_uids", type=str, default=None,
                        help="[test] comma-separated uids to visualize instead of sampling")
    parser.add_argument("--vis_only", action="store_true",
                        help="[test] only run the test on the visualized samples")

    args = parser.parse_args()
    
    if args.gpu != "auto":
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    base_cfg = load_config(args.config, n_gpus=len(args.gpu.split(",")))
    
    if "seed" in base_cfg:
        pl.seed_everything(base_cfg.seed, workers=True)
    
    if args.test:
        cfg = base_cfg.copy()
        if args.task == "sequential":
            args.task = "joint"
        if args.task == "joint":
            setup_test_visualization(cfg, args)
        train_single_task(cfg, args.task, args.config, args.resume, test_mode=True)
        
    elif args.task == "sequential":
        cfg_joint = base_cfg.copy()
        cfg_joint.system.freeze_encoder = False
        cfg_joint.system.freeze_shape_decoder = False
        train_single_task(cfg_joint, "joint", args.config, args.resume)
        train_single_task(cfg, "joint", args.config, args.resume, test_mode=True)
    else:
        cfg = base_cfg.copy()
        train_single_task(cfg, args.task, args.config, args.resume)


if __name__ == "__main__":
    main()
