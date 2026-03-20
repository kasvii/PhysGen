import os
import json
import trimesh
from dataclasses import dataclass, field

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from craftsman import register
from craftsman.utils.config import parse_structured
from craftsman.utils.typing import *
import pyvista as pv
import logging

# Constants for normalization
PRESSURE_MEAN = -94.5
PRESSURE_STD = 117.25

@dataclass
class DrivarNetPlusDataModuleConfig:
    pressure_path: str = ""
    subset_dir: str = ""
    cache_dir: str = ""
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

    test_obj_path: str = None
    vis_mesh: bool = True
    small: bool = False

class DrivarNetPlusDataset(Dataset):
    def __init__(self, cfg: Any, split: str) -> None:
        super().__init__()
        self.cfg: DrivarNetPlusDataModuleConfig = cfg
        self.split = split
        print(f'{cfg.root_dir}/{split}.json')
        self.uids = json.load(open(f'{cfg.root_dir}/{split}.json'))
        self.vtk_files = [
            os.path.join(cfg.pressure_path, f"{uid.split('/')[-1].replace('npz', 'vtk')}")
            for uid in self.uids
        ]

        # Filter to only existing files
        valid_uids = []
        valid_vtk_files = []
        for uid_path, vtk_file in zip(self.uids, self.vtk_files):
            if os.path.exists(uid_path) and os.path.exists(vtk_file):
                valid_uids.append(uid_path)
                valid_vtk_files.append(vtk_file)

        self.uids = valid_uids
        self.vtk_files = valid_vtk_files
        
        if self.split in ('val'):
            self.uids = self.uids[:3]

        print(f"Loaded {len(self.uids)} {split} samples")

        if self.split in ('test', 'val'):
            self.test_files = [
                os.path.join(cfg.test_obj_path, f"{uid.split('/')[-1].replace('npz', 'obj')}")
                for uid in self.uids
            ]

    def __len__(self):
        return len(self.uids)

    def _load_shape(self, index) -> Dict[str, Any]:
        if self.cfg.data_type == "sdf":
            data = np.load(f'{self.uids[index]}')
            coarse_fps_surface = data["fps_coarse_surface"]
            sharp_fps_surface = data['fps_sharp_surface']
        else:
            raise NotImplementedError(f"Data type {self.cfg.data_type} not implemented")

        coarse_surface = coarse_fps_surface[:, 0, :]
        nan_mask = np.isnan(coarse_surface)
        if np.any(nan_mask):
            logging.warning("NaN found in coarse surface, replacing with 1")
        coarse_surface = np.where(nan_mask, 1, coarse_surface)

        sharp_surface = sharp_fps_surface[:, 0, :]
        nan_mask = np.isnan(sharp_surface)
        if np.any(nan_mask):
            logging.warning("NaN found in sharp surface, replacing with 1")
        sharp_surface = np.where(nan_mask, 1, sharp_surface)

        return {
            "uid": self.uids[index],
            "coarse_surface": coarse_surface.astype(np.float32),
            "sharp_surface": sharp_surface.astype(np.float32),
            "data": data
        }

    def _load_shape_supervision(self, index, data) -> Dict[str, Any]:
        ret = {}
        coarse_rand_points = data['rand_points'][:, :3]
        sharp_near_points = data['sharp_near_surface'][:, :3]

        rng = np.random.default_rng()
        ind2 = rng.choice(sharp_near_points.shape[0], self.cfg.n_supervision[0], replace=False)
        ind3 = rng.choice(coarse_rand_points[:400000].shape[0], self.cfg.n_supervision[1], replace=False)
        ind4 = rng.choice(coarse_rand_points[400000:].shape[0], self.cfg.n_supervision[2], replace=False)
        rand_points = np.concatenate([
            sharp_near_points[ind2],
            coarse_rand_points[:400000][ind3],
            coarse_rand_points[400000:][ind4]
        ], axis=0)
        ret["rand_points"] = rand_points.astype(np.float32)
        ret["number_sharp"] = self.cfg.n_supervision[0]
        return ret

    def _get_cache_path(self, vtk_file_path):
        """Get the cache .npz path for a given .vtk file."""
        base_name = os.path.basename(vtk_file_path).replace('.vtk', '.npz')
        return os.path.join(self.cfg.cache_dir, base_name)

    def _save_to_cache(self, cache_path, point_cloud, pressures):
        """Save preprocessed point cloud and pressure data to cache."""
        np.savez_compressed(cache_path, points=point_cloud.points, pressures=pressures)

    def _load_from_cache(self, cache_path):
        """Load preprocessed point cloud and pressure data from cache."""
        data = np.load(cache_path)
        points = data['points']
        pressures = data['pressures']
        num_points = self.cfg.num_points

        if num_points > 0:
            if points.shape[0] >= num_points:
                indices = np.random.choice(points.shape[0], num_points, replace=False)
            else:
                indices = np.random.choice(points.shape[0], num_points, replace=True)
        else:
            indices = np.arange(points.shape[0])

        sampled_points = points[indices]
        sampled_pressures = pressures[indices]
        return pv.PolyData(sampled_points), sampled_pressures

    def sample_point_cloud_with_pressure(self, mesh, n_points=5000):
        """
        Sample points from the surface mesh and get corresponding pressure values.

        Args:
            mesh: PyVista mesh with pressure in point_data['p'].
            n_points: Number of points to sample (-1 for all).

        Returns:
            Tuple of (sampled point cloud, normalized pressures).
        """
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
        sampled_pressures = mesh.point_data['p'][indices].flatten()
        sampled_pressures = (sampled_pressures - PRESSURE_MEAN) / PRESSURE_STD

        if self.cfg.add_noise and self.split == 'train':
            chunk = self.cfg.num_points // 5
            noised_sampled_points = [
                sampled_points[:chunk],
                sampled_points[chunk:chunk*2] + np.random.normal(scale=0.001, size=(chunk, 3)),
                sampled_points[chunk*2:chunk*3] + np.random.normal(scale=0.005, size=(chunk, 3)),
                sampled_points[chunk*3:chunk*4] + np.random.normal(scale=0.007, size=(chunk, 3)),
                sampled_points[chunk*4:] + np.random.normal(scale=0.01, size=(len(sampled_points[chunk*4:]), 3)),
            ]
            sampled_points = np.concatenate(noised_sampled_points, axis=0)

        return pv.PolyData(sampled_points), sampled_pressures

    def _load_pressure(self, index, ret) -> Dict[str, Any]:
        vtk_file_path = self.vtk_files[index]
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
                logging.error(f"Cache not found for {vtk_file_path} and preprocessing is disabled.")
                return None, None

        ret['phys_points'] = point_cloud.points.astype(np.float32)
        ret['phys_pressures'] = pressures.astype(np.float32)

        if self.cfg.vis_mesh and self.split in ('test', 'val'):
            gen_mesh = trimesh.load(self.test_files[index])
            ret['gen_points'] = np.array(gen_mesh.vertices, dtype=np.float32)
            ret['gen_faces'] = np.array(gen_mesh.faces, dtype=np.int32)

        return ret

    def get_data(self, index):
        ret = self._load_shape(index)
        ret = self._load_pressure(index, ret)
        if self.cfg.load_supervision:
            ret.update(self._load_shape_supervision(index, ret['data']))
        del ret['data']
        return ret

    def __getitem__(self, index):
        if self.split == 'train':
            index %= len(self.uids)
        try:
            return self.get_data(index)
        except Exception as e:
            logging.warning(f"Error loading {self.uids[index]}: {e}")
            return self.__getitem__(np.random.randint(len(self)))

    def collate(self, batch):
        return torch.utils.data.default_collate(batch)


@register("drivarnetplus-datamodule")
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
            self.train_dataset, batch_size=self.cfg.batch_size,
            collate_fn=self.train_dataset.collate, num_workers=self.cfg.num_workers
        )

    def val_dataloader(self) -> DataLoader:
        return self.general_loader(self.val_dataset, batch_size=1, num_workers=self.cfg.num_workers)

    def test_dataloader(self) -> DataLoader:
        return self.general_loader(self.test_dataset, batch_size=1, num_workers=self.cfg.num_workers)

    def predict_dataloader(self) -> DataLoader:
        return self.general_loader(self.test_dataset, batch_size=1)