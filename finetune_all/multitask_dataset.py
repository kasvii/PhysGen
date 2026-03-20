"""
Multi-Task Dataset for Joint Training.
Combines shape, pressure, and drag coefficient data.
"""
import os
import json
import torch
import numpy as np
import pandas as pd
import trimesh
import pyvista as pv
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Union, Any, Optional
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from omegaconf import DictConfig

import pickle
import random
import math

from craftsman import register
from craftsman.utils.base import Updateable
from craftsman.utils.config import parse_structured
from craftsman.utils.typing import *

def random_rotation_matrix():
    """Create a random rotation matrix."""
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
    """Create a random mirror matrix."""
    if np.random.rand() < 0.75:
        axis = np.random.choice([0, 1, 2], size=1)[0]
        M = np.eye(3)
        M[axis, axis] = -1
    else:
        M = np.eye(3)
    return M

def apply_transformation(points, normals, transform):
    """Apply a transformation matrix to points and normals."""
    transformed_points = np.dot(points, transform.T)
    if normals is not None:
        epsilon = 1e-6
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms[norms == 0] = epsilon
        norms[np.isinf(norms)] = 1
        norms[np.isnan(norms)] = 1
        normals /= norms

        transformed_normals = np.dot(normals, transform.T)
        norms = np.linalg.norm(transformed_normals, axis=1, keepdims=True)
        norms[norms == 0] = epsilon
        norms[np.isinf(norms)] = 1
        norms[np.isnan(norms)] = 1
        transformed_normals /= norms
    else:
        transformed_normals = None
    return transformed_points, transformed_normals

PRESSURE_MEAN = -94.5
PRESSURE_STD = 117.25

@dataclass
class MultiTaskDataModuleConfig:
    shape_root_dir: str = "./data/dataset1"

    pressure_path: str = ""
    physics_cache_dir: str = ""
    num_pressure_points: int = 400000

    drag_data_root: str = ""
    csv_path: str = ""

    subset_dir: str = ""
    data_type: str = "sdf"

    load_supervision: bool = True
    supervision_type: str = "tsdf"
    n_supervision: list[int] = field(default_factory=lambda: [21384, 10000, 10000])

    rotate_points: bool = True
    add_noise: bool = False

    batch_size: int = 32
    num_workers: int = 16

    task_weights: Dict[str, float] = field(default_factory=lambda: {
        "shape": 0.4,
        "physics": 0.3,
        "drag": 0.3
    })

    small: bool = False
    preprocess: bool = True

class MultiTaskDataset(Dataset):
    """Multi-task dataset combining shape, physics, and drag data."""

    def __init__(self, cfg: MultiTaskDataModuleConfig, split: str) -> None:
        super().__init__()
        self.cfg = cfg
        self.split = split

        self._load_uids()
        self._load_drag_data()

        self.task_names = list(self.cfg.task_weights.keys())
        self.task_probs = np.array(list(self.cfg.task_weights.values()))
        self.task_probs = self.task_probs / self.task_probs.sum()

    def _load_uids(self):
        """Load UIDs from subset directory."""
        split_map = {
            'train': 'train.json',
            'val': 'val.json',
            'test': 'test.json'
        }

        json_file = os.path.join(self.cfg.shape_root_dir, split_map[self.split])
        if os.path.exists(json_file):
            with open(json_file, 'r') as f:
                self.uids = json.load(f)

        if self.split == 'val':
            self.uids = self.uids[:3]

    def _load_drag_data(self):
        """Load drag coefficient CSV data."""
        if os.path.exists(self.cfg.csv_path):
            self.drag_df = pd.read_csv(self.cfg.csv_path)

            if 'Drag_Value' in self.drag_df.columns:
                self.id_to_drag = dict(zip(self.drag_df['ID'], self.drag_df['Drag_Value']))
            elif 'Cd' in self.drag_df.columns:
                self.id_to_drag = dict(zip(self.drag_df['ID'], self.drag_df['Cd']))
            else:
                self.id_to_drag = dict(zip(self.drag_df.iloc[:, 0], self.drag_df.iloc[:, 1]))
        else:
            self.drag_df = None
            self.id_to_drag = {}

    def __len__(self) -> int:
        return len(self.uids)

    def _load_shape_data(self, index: int, mirror_matrix: np.ndarray, rotation_matrix: np.ndarray) -> Dict[str, Any]:
        """Load shape data."""
        uid_path = self.uids[index]

        if not os.path.exists(uid_path):
            raise FileNotFoundError(f"Shape data not found: {uid_path}")

        data = np.load(uid_path)

        coarse_surface = data["fps_coarse_surface"][:, 0, :]
        sharp_surface = data["fps_sharp_surface"][:, 0, :]

        nan_mask_coarse = np.isnan(coarse_surface)
        if np.any(nan_mask_coarse):
            coarse_surface = np.where(nan_mask_coarse, 1, coarse_surface)

        nan_mask_sharp = np.isnan(sharp_surface)
        if np.any(nan_mask_sharp):
            sharp_surface = np.where(nan_mask_sharp, 1, sharp_surface)

        uid = os.path.basename(uid_path).replace('.npz', '')

        return {
            "uid": uid,
            "coarse_surface": coarse_surface.astype(np.float32),
            "sharp_surface": sharp_surface.astype(np.float32),
            "data": data
        }

    def _load_shape_supervision(self, index: int, mirror_matrix: np.ndarray,
                               rotation_matrix: np.ndarray, data: Dict[str, Any]) -> Dict[str, Any]:
        """Load shape supervision data."""
        if not self.cfg.load_supervision:
            return {}

        coarse_rand_points = data['rand_points'][:, :3]
        coarse_sdfs = data['rand_points'][:, 3]
        sharp_near_points = data['sharp_near_surface'][:, :3]
        sharp_sdfs = data['sharp_near_surface'][:, 3]

        rng = np.random.default_rng()
        ind2 = rng.choice(sharp_near_points.shape[0], self.cfg.n_supervision[0], replace=False)
        ind3 = rng.choice(coarse_rand_points[:400000].shape[0], self.cfg.n_supervision[1], replace=False)
        ind4 = rng.choice(coarse_rand_points[400000:].shape[0], self.cfg.n_supervision[2], replace=False)

        rand_points2 = sharp_near_points[ind2]
        rand_points3 = coarse_rand_points[:400000][ind3]
        rand_points4 = coarse_rand_points[400000:][ind4]
        rand_points = np.concatenate([rand_points2, rand_points3, rand_points4], axis=0)

        ret = {
            "rand_points": rand_points.astype(np.float32),
            "number_sharp": self.cfg.n_supervision[0]
        }

        if self.cfg.data_type == "sdf":
            sdf2 = sharp_sdfs[ind2]
            sdf3 = coarse_sdfs[:400000][ind3]
            sdf4 = coarse_sdfs[400000:][ind4]
            sdfs = np.concatenate([sdf2, sdf3, sdf4], axis=0)

            nan_mask = np.isnan(sdfs)
            sdfs = np.where(nan_mask, 0, sdfs)

            if self.cfg.supervision_type == "occupancy":
                ret["occupancies"] = np.where(sdfs.flatten() < 0, 0, 1).astype(np.float32)
            elif self.cfg.supervision_type == "tsdf":
                ret["sdf"] = sdfs.flatten().astype(np.float32).clip(-0.015, 0.015) / 0.015

        return ret

    def _get_cache_path(self, vtk_file_path):
        """Get the corresponding .npz cache path for a given .vtk file."""
        base_name = os.path.basename(vtk_file_path).replace('.vtk', '.npz')
        return os.path.join(self.cfg.physics_cache_dir, base_name)

    def _save_to_cache(self, cache_path, point_cloud, pressures):
        """Save preprocessed point cloud and pressure data."""
        np.savez_compressed(cache_path, points=point_cloud.points, pressures=pressures)

    def _load_from_cache(self, cache_path):
        """Load preprocessed point cloud and pressure data from cache."""
        data = np.load(cache_path)
        points = data['points']
        pressures = data['pressures']
        num_points = self.cfg.num_pressure_points

        if num_points > 0:
            if points.shape[0] >= num_points:
                indices = np.random.choice(points.shape[0], num_points, replace=False)
            else:
                indices = np.random.choice(points.shape[0], num_points, replace=True)
        else:
            indices = np.arange(points.shape[0])

        sampled_points = points[indices]
        sampled_pressures = pressures[indices]

        point_cloud = pv.PolyData(sampled_points)
        return point_cloud, sampled_pressures

    def sample_point_cloud_with_pressure(self, mesh, n_points=400000):
        """Sample points from the surface mesh and get corresponding pressure values."""
        points = mesh.points
        bbmin = points.min(0)
        bbmax = points.max(0)
        center = (bbmin + bbmax) / 2
        scale = 2.0 / (bbmax - bbmin).max()
        points = (points - center) * scale

        if n_points > 0:
            if mesh.n_points >= n_points:
                indices = np.random.choice(mesh.n_points, n_points, replace=False)
            else:
                pad_size = n_points - mesh.n_points
                pad_indices = np.random.choice(mesh.n_points, pad_size, replace=True)
                indices = np.concatenate([np.arange(mesh.n_points), pad_indices], axis=0)
        else:
            indices = np.arange(mesh.n_points)

        sampled_points = points[indices]
        sampled_pressures = mesh.point_data['p'][indices]
        sampled_pressures = sampled_pressures.flatten()
        sampled_pressures = (sampled_pressures - PRESSURE_MEAN) / PRESSURE_STD

        if self.cfg.add_noise and self.split == 'train':
            noised_sampled_points = [
                sampled_points[:self.cfg.num_points//5],
                sampled_points[self.cfg.num_points//5:self.cfg.num_points//5*2] + np.random.normal(scale=0.001, size=(self.cfg.num_points//5, 3)),
                sampled_points[self.cfg.num_points//5*2:self.cfg.num_points//5*3] + np.random.normal(scale=0.005, size=(self.cfg.num_points//5, 3)),
                sampled_points[self.cfg.num_points//5*3:self.cfg.num_points//5*4] + np.random.normal(scale=0.007, size=(self.cfg.num_points//5, 3)),
                sampled_points[self.cfg.num_points//5*4:] + np.random.normal(scale=0.01, size=(len(sampled_points[self.cfg.num_points//5*4:]), 3)),
            ]
            sampled_points = np.concatenate(noised_sampled_points, axis=0)

        return pv.PolyData(sampled_points), sampled_pressures

    def _load_physics_data(self, index: int) -> Dict[str, Any]:
        """Load physics (pressure) data."""
        uid_path = self.uids[index]
        uid = os.path.basename(uid_path).replace('.npz', '')

        vtk_file_path = os.path.join(self.cfg.pressure_path, f"{uid}.vtk")
        cache_path = self._get_cache_path(vtk_file_path)

        if os.path.exists(cache_path):
            point_cloud, pressures = self._load_from_cache(cache_path)
        else:
            if self.cfg.preprocess:
                try:
                    mesh = pv.read(vtk_file_path)
                except Exception as e:
                    logging.error(f"Failed to load VTK file: {vtk_file_path}. Error: {e}")
                    return None, None

                point_cloud, pressures = self.sample_point_cloud_with_pressure(mesh, self.cfg.num_points)
                self._save_to_cache(cache_path, point_cloud, pressures)
            else:
                logging.error(f"Cache file not found for {vtk_file_path} and preprocessing is disabled.")
                return None, None

        return {
            "query_points": point_cloud.points.astype(np.float32),
            "pressure": pressures.astype(np.float32)
        }

    def _load_drag_data_item(self, index: int) -> Dict[str, Any]:
        """Load drag coefficient data for a single item."""
        uid_path = self.uids[index]
        shape_id = os.path.basename(uid_path).replace('.npz', '')
        drag_coefficient = self.id_to_drag.get(shape_id, 0.0)

        return {"drag_coefficient": np.float32(drag_coefficient)}

    def get_data(self, index: int) -> Dict[str, Any]:
        """Get data sample combining all tasks."""
        mirror_matrix = np.eye(3)
        rotation_matrix = np.eye(3)

        ret = self._load_shape_data(index, mirror_matrix, rotation_matrix)

        if self.cfg.load_supervision:
            ret.update(self._load_shape_supervision(index, mirror_matrix, rotation_matrix, ret['data']))

        physics_data = self._load_physics_data(index)
        ret.update(physics_data)

        drag_data = self._load_drag_data_item(index)
        ret.update(drag_data)

        if 'data' in ret:
            del ret['data']

        return ret

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if self.split == 'train':
            index %= len(self.uids)
        try:
            return self.get_data(index)
        except Exception as e:
            logging.warning(f"Error loading {self.uids[index]}: {e}")
            return self.__getitem__(np.random.randint(len(self)))

    def collate(self, batch):
        """Collate function for batching."""
        return torch.utils.data.default_collate(batch)

@register("multitask-datamodule")
class MultiTaskDataModule(pl.LightningDataModule):
    """PyTorch Lightning Data Module for multi-task training."""

    cfg: MultiTaskDataModuleConfig

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        super().__init__()
        self.cfg = parse_structured(MultiTaskDataModuleConfig, cfg)

    def setup(self, stage: str = None) -> None:
        if stage in [None, "fit"]:
            self.train_dataset = MultiTaskDataset(self.cfg, "train")
        if stage in [None, "fit", "validate"]:
            self.val_dataset = MultiTaskDataset(self.cfg, "val")
        if stage in [None, "test", "predict"]:
            self.test_dataset = MultiTaskDataset(self.cfg, "test")

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
        return self.general_loader(
            self.val_dataset,
            batch_size=1,
            num_workers=self.cfg.num_workers
        )

    def test_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.test_dataset,
            batch_size=1,
            num_workers=self.cfg.num_workers
        )
