import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from craftsman import register
from craftsman.utils.config import parse_structured

@dataclass
class DragDataModuleConfig:
    data_root: str = "data/drivaernet_plus/surfaces"
    csv_path: str = "data/drivaernet_plus/drag/DrivAerNetPlusPlus_Cd_8k_Updated.csv"
    subset_dir: str = "models/physgen/DrivAerNet/train_val_test_splits"
    root_dir: str = None
    data_type: str = "sdf"
    add_noise: bool = False
    rotate_points: bool = False
    batch_size: int = 32
    num_workers: int = 4
    small: bool = False

class DragDataset(Dataset):
    """Dataset for drag coefficient prediction."""

    def __init__(self, cfg: Any, split: str) -> None:
        super().__init__()
        self.cfg: DragDataModuleConfig = cfg
        self.split = split

        self.drag_df = pd.read_csv(self.cfg.csv_path)
        self.id_to_drag = dict(zip(self.drag_df["ID"], self.drag_df["Drag_Value"]))

        split_file = os.path.join(self.cfg.root_dir, f"{split}.json")

        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Split file not found: {split_file}")

        with open(split_file, "r") as f:
            self.uids = json.load(f)
            
        if self.split in ('val'):
            self.uids = self.uids[:3]

        self.uids = [
            uid
            for uid in self.uids
            if self._get_shape_id_from_uid(uid) in self.id_to_drag
        ]

    def __len__(self):
        return len(self.uids)

    def _get_shape_id_from_uid(self, uid):
        """Extract the shape ID from a UID path."""
        if isinstance(uid, str):
            return uid.split("/")[-1].replace(".npz", "")
        return str(uid)

    def _load_shape(self, index) -> Dict[str, Any]:
        """Load a shape sample."""
        uid = self.uids[index]
        shape_id = self._get_shape_id_from_uid(uid)

        if self.cfg.data_type == "sdf":
            data = np.load(self.uids[index])
            coarse_fps_surface = data["fps_coarse_surface"]
            sharp_fps_surface = data["fps_sharp_surface"]
        else:
            raise NotImplementedError(f"Data type {self.cfg.data_type} not implemented")

        coarse_surface = np.nan_to_num(coarse_fps_surface[:, 0, :], nan=1.0)
        sharp_surface = np.nan_to_num(sharp_fps_surface[:, 0, :], nan=1.0)
        drag_coefficient = self.id_to_drag.get(shape_id, 0.0)

        return {
            "uid": uid,
            "shape_id": shape_id,
            "coarse_surface": coarse_surface.astype(np.float32),
            "sharp_surface": sharp_surface.astype(np.float32),
            "drag_coefficient": np.float32(drag_coefficient),
        }

    def get_data(self, index):
        return self._load_shape(index)

    def __getitem__(self, index):
        if self.split == "train":
            index %= len(self.uids)
        try:
            return self.get_data(index)
        except Exception as e:
            logging.error(f"Error loading data for index {index}: {e}")
            return self.__getitem__(np.random.randint(len(self)))

    def collate(self, batch):
        return torch.utils.data.default_collate(batch)

@register("drag-datamodule")
class DragDataModule(pl.LightningDataModule):
    """PyTorch Lightning data module for drag coefficient prediction."""

    cfg: DragDataModuleConfig

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        super().__init__()
        self.cfg = parse_structured(DragDataModuleConfig, cfg)

    def setup(self, stage: str = None):
        if stage in [None, "fit"]:
            self.train_dataset = DragDataset(self.cfg, "train")
        if stage in [None, "fit", "validate"]:
            self.val_dataset = DragDataset(self.cfg, "val")
        if stage in [None, "test", "predict"]:
            self.test_dataset = DragDataset(self.cfg, "test")
    
    def prepare_data(self):
        pass

    def general_loader(self, dataset, batch_size, collate_fn=None, num_workers=0) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=True,
        )

    def train_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            collate_fn=self.train_dataset.collate,
            num_workers=self.cfg.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.val_dataset,
            batch_size=1,
            num_workers=self.cfg.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.test_dataset,
            batch_size=1,
            num_workers=self.cfg.num_workers,
        )

    def predict_dataloader(self) -> DataLoader:
        return self.general_loader(self.test_dataset, batch_size=1)