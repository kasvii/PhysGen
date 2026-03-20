#  Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# 
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
# 
#  http://www.apache.org/licenses/LICENSE-2.0
# 
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import math
import os, sys
import json
import trimesh
from dataclasses import dataclass, field

import random
import numpy as np 
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from craftsman import register
from craftsman.utils.base import Updateable
from craftsman.utils.config import parse_structured
from craftsman.utils.typing import *
import pyvista as pv
import logging
sys.path.append('/scratch/cvlab/home/yiyou/models/physgen/Dora/pytorch_lightning')

PRESSURE_MEAN = -94.5
PRESSURE_STD = 117.25

@dataclass
class DrivarNetPlusDataModuleConfig:
    pressure_path: str = "/scratch/cvlab/home/yiyou/data/drivaernet_plus/pressure/PressureVTK"
    subset_dir: str = "/scratch/cvlab/home/yiyou/models/physgen/DrivAerNet/train_val_test_splits"
    cache_dir: str = "/scratch/cvlab/home/yiyou/data/drivaernet_plus/pressure/physdec_data"
    num_points: int = 400000
    preprocess: bool = True
    
    root_dir: str = None
    data_type: str = "sdf"
    
    load_supervision: bool = True
    supervision_type: str = "tsdf"
    n_supervision: list[int] = field(default_factory=lambda: [21384, 10000, 10000])

    rotate_points: bool = False
    add_noise: bool = True
    batch_size: int = 32
    num_workers: int = 0
    
    vis_mesh: bool = True
    small: bool = False

class DrivarNetPlusDataset(Dataset):
    def __init__(self, cfg: Any, split: str) -> None:
        super().__init__()
        self.cfg: DrivarNetPlusDataModuleConfig = cfg
        self.split = split
        self.uids = json.load(open(f'{cfg.root_dir}/{split}.json'))

        print(f"Loaded {len(self.uids)} {split} uids")
        
    def __len__(self):
        if self.split =='train':
            return len(self.uids)
        else:
            return len(self.uids)

    def _load_shape(self, index) -> Dict[str, Any]:
        if self.cfg.data_type == "sdf":
            data = np.load(f'{self.uids[index]}')
            coarse_fps_surface = data["fps_coarse_surface"]
            sharp_fps_surface = data['fps_sharp_surface']
        else:
            raise NotImplementedError(f"Data type {self.cfg.data_type} not implemented")

        coarse_surface = coarse_fps_surface[:,0,:]

        nan_mask = np.isnan(coarse_surface)
        if np.any(nan_mask):
            print("nan exist in coarse surface")
        coarse_surface = np.where(nan_mask, 1, coarse_surface)

        sharp_surface = sharp_fps_surface[:,0,:]

        nan_mask = np.isnan(sharp_surface)
        if np.any(nan_mask):
            print("nan exist in sharp surface! ")
        sharp_surface = np.where(nan_mask, 1, sharp_surface)

        ret = {
            "uid": self.uids[index],
            "coarse_surface": coarse_surface.astype(np.float32),
            "sharp_surface": sharp_surface.astype(np.float32),
            "data":data
        }

        return ret

    def _load_shape_supervision(self, index, data) -> Dict[str, Any]:
        ret = {}
        coarse_rand_points = data['rand_points'][:,:3]
        coarse_sdfs = data['rand_points'][:,3]
        sharp_near_points = data['sharp_near_surface'][:,:3]
        sharp_sdfs = data['sharp_near_surface'][:,3]

        rng = np.random.default_rng()
        ind2 = rng.choice(sharp_near_points.shape[0], self.cfg.n_supervision[0], replace=False)
        ind3 = rng.choice(coarse_rand_points[:400000].shape[0], self.cfg.n_supervision[1], replace=False)
        ind4 = rng.choice(coarse_rand_points[400000:].shape[0], self.cfg.n_supervision[2], replace=False)
        rand_points2 = sharp_near_points[ind2] 
        rand_points3 = coarse_rand_points[:400000][ind3]
        rand_points4 = coarse_rand_points[400000:][ind4]
        rand_points = np.concatenate([rand_points2, rand_points3,rand_points4], axis=0) 
        ret["rand_points"] = rand_points.astype(np.float32)
        ret["number_sharp"] = self.cfg.n_supervision[0]

        return ret

    def get_data(self, index):
        ret = self._load_shape(index)
        if self.cfg.load_supervision:
            ret.update(self._load_shape_supervision(index,ret['data']))
        del ret['data']
        return ret
        
    def __getitem__(self, index):
        if self.split == 'train':
            index %= len(self.uids)
        try:
            return self.get_data(index)
        except Exception as e:
            print(f"Error in {self.uids[index]}: {e}")
            return self.__getitem__(np.random.randint(len(self)))


    def collate(self, batch):
        batch = torch.utils.data.default_collate(batch)
        return batch

class DrivarNetPlusDataModule(pl.LightningDataModule):
    cfg: DrivarNetPlusDataModuleConfig

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        super().__init__()
        self.cfg = parse_structured(DrivarNetPlusDataModuleConfig, cfg)

    def setup(self, stage=None) -> None:
        if stage in [None, "fit"]:
            self.train_dataset = DrivarNetPlusDataset(self.cfg, "train")
        if stage in [None, "fit", "validate"]:
            self.val_dataset = DrivarNetPlusDataset(self.cfg, "val")
        if stage in [None, "test", "predict"]:
            self.test_dataset = DrivarNetPlusDataset(self.cfg, "test")

    def prepare_data(self):
        pass

    def general_loader(self, dataset, batch_size, collate_fn=None, num_workers=0) -> DataLoader:
        return DataLoader(
            dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=num_workers
        )

    def train_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            collate_fn=self.train_dataset.collate,
            num_workers=self.cfg.num_workers
        )

    def val_dataloader(self) -> DataLoader:
        return self.general_loader(self.val_dataset, batch_size=1, num_workers=self.cfg.num_workers)

    def test_dataloader(self) -> DataLoader:
        return self.general_loader(self.test_dataset, batch_size=1, num_workers=self.cfg.num_workers)

    def predict_dataloader(self) -> DataLoader:
        return self.general_loader(self.test_dataset, batch_size=1)