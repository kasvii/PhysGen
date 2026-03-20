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
import os
import json
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



def random_rotation_matrix():
    """
    Create a random rotation matrix.
    """
    angle = np.random.uniform(0, 2 * np.pi)
    axis = np.random.normal(size=3)
    axis /= np.linalg.norm(axis)
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    R = np.array([[cos_angle + axis[0]**2 * (1 - cos_angle),
                   axis[0]*axis[1] * (1 - cos_angle) - axis[2]*sin_angle,
                   axis[0]*axis[2] * (1 - cos_angle) + axis[1]*sin_angle],
                  [axis[1]*axis[0] * (1 - cos_angle) + axis[2]*sin_angle,
                   cos_angle + axis[1]**2 * (1 - cos_angle),
                   axis[1]*axis[2] * (1 - cos_angle) - axis[0]*sin_angle],
                  [axis[2]*axis[0] * (1 - cos_angle) - axis[1]*sin_angle,
                   axis[2]*axis[1] * (1 - cos_angle) + axis[0]*sin_angle,
                   cos_angle + axis[2]**2 * (1 - cos_angle)]])
    return R

def random_mirror_matrix():
    """
    Create a random mirror matrix.
    """
    if np.random.rand() < 0.75:
        axis = np.random.choice([0, 1, 2], size=1)[0]
        M = np.eye(3)
        M[axis, axis] = -1
    else:
        M = np.eye(3)
    return M


def apply_transformation(points, normals, transform):
    """
    Apply a transformation matrix to points and normals.
    """
    transformed_points = np.dot(points, transform.T)
    if normals is not  None:
        norms = np.linalg.norm(normals, axis=1, keepdims=True)

        epsilon = 1e-6
        norms[norms == 0] = epsilon
        norms[np.isinf(norms)] = 1
        norms[np.isnan(norms)] = 1

        normals /= norms
        
        transformed_normals = np.dot(normals, transform.T)
        norms = np.linalg.norm(transformed_normals, axis=1, keepdims=True)

        epsilon = 1e-6
        norms[norms == 0] = epsilon
        norms[np.isinf(norms)] = 1
        norms[np.isnan(norms)] = 1

        transformed_normals /= norms
    else:
        transformed_normals=None
    return transformed_points, transformed_normals

@dataclass
class ObjaverseDataModuleConfig:
    root_dir: str = None
    data_type: str = "sdf"         # sdf
    
    load_supervision: bool = True        # whether to load supervision
    supervision_type: str = "tsdf"       # tsdf
    n_supervision: list[int] = field(default_factory=lambda: [21384, 10000, 10000])           # number of points in supervision

    rotate_points: bool = False          # whether to rotate the input point cloud and the supervision
    batch_size: int = 32
    num_workers: int = 0

class ObjaverseDataset(Dataset):
    def __init__(self, cfg: Any, split: str) -> None:
        super().__init__()
        self.cfg: ObjaverseDataModuleConfig = cfg
        self.split = split
        print(f'{cfg.root_dir}/{split}.json')
        self.uids = json.load(open(f'{cfg.root_dir}/{split}.json'))
        print(f"Loaded {len(self.uids)} {split} uids")

    def __len__(self):
        if self.split =='train':
            return len(self.uids) # * 100
        else:
            return len(self.uids)

    def _load_shape(self, index, mirror_matrix,rotation_matrix) -> Dict[str, Any]:
        if self.cfg.data_type == "sdf":
            data = np.load(f'{self.uids[index]}')
            coarse_fps_surface = data["fps_coarse_surface"]
            sharp_fps_surface = data['fps_sharp_surface']
        else:
            raise NotImplementedError(f"Data type {self.cfg.data_type} not implemented")

        # rng = np.random.default_rng() 
        # ind = rng.choice(5, 1, replace=False)
        # coarse_surface = coarse_fps_surface[:,ind,:].squeeze() #（m,1,p) >(m.p)
        coarse_surface = coarse_fps_surface[:,0,:] #（m,1,p) >(m.p)

        nan_mask = np.isnan(coarse_surface)
        if np.any(nan_mask):
            print("nan exist in coarse surface")
        coarse_surface = np.where(nan_mask, 1, coarse_surface)

        # ind = rng.choice(5, 1, replace=False)
        # sharp_surface = sharp_fps_surface[:,ind,:].squeeze() #（m,1,p) >(m.p)
        sharp_surface = sharp_fps_surface[:,0,:]

        nan_mask = np.isnan(sharp_surface)
        if np.any(nan_mask):
            print("nan exist in sharp surface! ")
        sharp_surface = np.where(nan_mask, 1, sharp_surface)
        

        if self.cfg.rotate_points and self.split=="train":
            mirrored_points, mirrored_normals = apply_transformation(coarse_surface[:,:3], coarse_surface[:,3:], mirror_matrix)
            surface, normal = apply_transformation(mirrored_points, mirrored_normals, rotation_matrix)
            coarse_surface = np.concatenate([surface, normal], axis=1)
            mirrored_points, mirrored_normals = apply_transformation(sharp_surface[:,:3], sharp_surface[:,3:], mirror_matrix)
            surface, normal = apply_transformation(mirrored_points, mirrored_normals, rotation_matrix)
            sharp_surface = np.concatenate([surface, normal], axis=1)

        ret = {
            "uid": self.uids[index],
            "coarse_surface": coarse_surface.astype(np.float32),
            "sharp_surface": sharp_surface.astype(np.float32),
            "data":data
        }

        return ret

    def _load_shape_supervision(self, index, mirror_matrix,rotation_matrix,data) -> Dict[str, Any]:
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
        if self.cfg.rotate_points and self.split=="train":
            mirrored_points, _ = apply_transformation(rand_points, None, mirror_matrix)
            rand_points, _= apply_transformation(mirrored_points, None, rotation_matrix)
        ret["rand_points"] = rand_points.astype(np.float32)
        ret["number_sharp"] = self.cfg.n_supervision[0]
        if self.cfg.data_type == "sdf":
            if self.cfg.supervision_type == "occupancy":
                sdf2 = sharp_sdfs[ind2]
                sdf3 = coarse_sdfs[:400000][ind3]
                sdf4 = coarse_sdfs[400000:][ind4]
                sdfs = np.concatenate([sdf2,sdf3,sdf4], axis=0) 
                nan_mask = np.isnan(sdfs)
                if np.any(nan_mask):
                    print("nan exist in sdfs")
                sdfs = np.where(nan_mask, 0, sdfs)
                ret["occupancies"] = np.where(sdfs.flatten() < 0, 0, 1).astype(np.float32)
            elif self.cfg.supervision_type == "tsdf":
                sdf2 = sharp_sdfs[ind2]
                sdf3 = coarse_sdfs[:400000][ind3]
                sdf4 = coarse_sdfs[400000:][ind4]
                sdfs = np.concatenate([sdf2,sdf3,sdf4], axis=0)
                nan_mask = np.isnan(sdfs)
                if np.any(nan_mask):
                    print("nan exist in sdfs")
                sdfs = np.where(nan_mask, 0, sdfs)
                ret["sdf"] = sdfs.flatten().astype(np.float32).clip(-0.015,0.015) / 0.015
            else:
                raise NotImplementedError(f"Supervision type {self.cfg.supervision_type} not implemented")

        return ret

    def get_data(self, index):
        mirror_matrix = random_mirror_matrix()
        rotation_matrix = random_rotation_matrix()
        ret = self._load_shape(index,mirror_matrix,rotation_matrix )
        if self.cfg.load_supervision:
            ret.update(self._load_shape_supervision(index,mirror_matrix,rotation_matrix,ret['data']))
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

@register("objaverse-datamodule")
class ObjaverseDataModule(pl.LightningDataModule):
    cfg: ObjaverseDataModuleConfig

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        super().__init__()
        self.cfg = parse_structured(ObjaverseDataModuleConfig, cfg)

    def setup(self, stage=None) -> None:
        if stage in [None, "fit"]:
            self.train_dataset = ObjaverseDataset(self.cfg, "train")
        if stage in [None, "fit", "validate"]:
            self.val_dataset = ObjaverseDataset(self.cfg, "val")
        if stage in [None, "test", "predict"]:
            self.test_dataset = ObjaverseDataset(self.cfg, "test")

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