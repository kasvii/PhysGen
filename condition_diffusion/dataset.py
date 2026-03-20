#  Copyright (c) 2024 Bytedance Ltd. and/or its affiliates

#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at

#  http://www.apache.org/licenses/LICENSE-2.0


#  distributed under the License is distributed on an "AS IS" BASIS,

#  See the License for the specific language governing permissions and
#  limitations under the License.
import math
import os, sys
import json
import trimesh
from dataclasses import dataclass, field

import random
import numpy as np 
import pandas as pd
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
from PIL import Image
import torchvision.transforms as transforms

PRESSURE_MEAN = -94.5
PRESSURE_STD = 117.25

@dataclass
class DrivarNetPlusDataModuleConfig:
    pressure_path: str = "data/drivaernet_plus/pressure/PressureVTK"
    condition_path: str = "data/drivaernet_plus/condition"
    condition_type: str = "matched_sketches-CannyEdge"
    subset_dir: str = "train_val_test_splits"
    cache_dir: str = "data/drivaernet_plus/pressure/physdec_data"
    num_points: int = 400000
    preprocess: bool = True
    
    drag_data_root: str = "data/drivaernet_plus/surfaces"
    csv_path: str = "data/drivaernet_plus/drag/DrivAerNetPlusPlus_Cd_8k_Updated.csv"
    
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
    
    test_set: str = "test"

@dataclass
class DrivarNetPlusDataDemoModuleConfig:
    condition_path: str = "demo"
    condition_type: str = "matched_sketches-CLIPasso"
    
    root_dir: str = "./data/dataset1"
    data_type: str = "sdf"
    
    num_points: int = 40000
    preprocess: bool = True
    
    batch_size: int = 4
    num_workers: int = 4
    
    rotate_points: bool = False
    add_noise: bool = False
    
    load_supervision: bool = False
    
    vis_mesh: bool = True
    small: bool = False
    seed: int = 0

class DrivarNetPlusDataset(Dataset):
    def __init__(self, cfg: Any, split: str) -> None:
        super().__init__()
        self.cfg: DrivarNetPlusDataModuleConfig = cfg
        self.split = split
        
        try:
            import torchvision.transforms as transforms
            self.image_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        except ImportError:
            print("Warning: torchvision not available, using basic transform")
            self.image_transform = None
        
        self.uids = json.load(open(f'{cfg.root_dir}/{split}.json'))

        self._verify_condition_alignment()

        self._load_drag_data()
        
        self.check_alignment_stats()

    def _verify_condition_alignment(self):
        valid_uids = []
        missing_count = 0
        
        for uid in self.uids:
            shape_id = uid.split('/')[-1].replace('.npz', '')
            
            condition_image_path = os.path.join(
                self.cfg.condition_path, 
                self.cfg.condition_type, 
                f"{shape_id}.png"
            )
            
            if not os.path.exists(condition_image_path):
                condition_image_path = condition_image_path.replace('.png', '.jpg')
            
            if os.path.exists(uid) and os.path.exists(condition_image_path):
                valid_uids.append(uid)
            else:
                missing_count += 1
                if missing_count <= 5:
                    pass
        
        if missing_count > 0:
            print(f"Warning: {missing_count} samples removed due to missing files")
            
        self.uids = valid_uids

    def __len__(self):
        if self.split =='train':
            return len(self.uids)
        else:
            return len(self.uids)

    def _load_condition_image(self, uid) -> torch.Tensor:
        shape_id = uid.split('/')[-1].replace('.npz', '')
        
        condition_image_path = os.path.join(
            self.cfg.condition_path, 
            self.cfg.condition_type, 
            f"{shape_id}.png"
        )
        
        if not os.path.exists(condition_image_path):
            condition_image_path = condition_image_path.replace('.png', '.jpg')
        
        if not os.path.exists(condition_image_path):
            print(f"Error: Condition image not found: {condition_image_path}")
            return torch.zeros(3, 224, 224)
        
        try:
            image = Image.open(condition_image_path).convert('RGB')
            
            if self.image_transform is not None:
                image = self.image_transform(image)
            else:
                image = image.resize((224, 224))
                image = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                image = (image - mean) / std
            
            return image
            
        except Exception as e:
            print(f"Error: Failed to load condition image {condition_image_path}: {e}")

            return torch.zeros(3, 224, 224)

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
            pass
        coarse_surface = np.where(nan_mask, 1, coarse_surface)

        sharp_surface = sharp_fps_surface[:,0,:]

        nan_mask = np.isnan(sharp_surface)
        if np.any(nan_mask):
            pass
        sharp_surface = np.where(nan_mask, 1, sharp_surface)

        condition_image = self._load_condition_image(self.uids[index])

        ret = {
            "uid": self.uids[index],
            "coarse_surface": coarse_surface.astype(np.float32),
            "sharp_surface": sharp_surface.astype(np.float32),
            "condition_image": condition_image,
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
        
        if self.cfg.data_type == "sdf":
            if self.cfg.supervision_type == "occupancy":
                sdf2 = sharp_sdfs[ind2]
                sdf3 = coarse_sdfs[:400000][ind3]
                sdf4 = coarse_sdfs[400000:][ind4]
                sdfs = np.concatenate([sdf2, sdf3, sdf4], axis=0)
                
                nan_mask = np.isnan(sdfs)
                if np.any(nan_mask):
                    pass
                sdfs = np.where(nan_mask, 0, sdfs)
                
                ret["occupancies"] = np.where(sdfs.flatten() < 0, 0, 1).astype(np.float32)
            elif self.cfg.supervision_type == "tsdf":
                sdf2 = sharp_sdfs[ind2]
                sdf3 = coarse_sdfs[:400000][ind3]
                sdf4 = coarse_sdfs[400000:][ind4]
                sdfs = np.concatenate([sdf2, sdf3, sdf4], axis=0)
                
                nan_mask = np.isnan(sdfs)
                if np.any(nan_mask):
                    pass
                sdfs = np.where(nan_mask, 0, sdfs)
                
                ret["sdf"] = sdfs.flatten().astype(np.float32).clip(-0.015, 0.015) / 0.015

        return ret
    
    def _load_drag_data(self):
        """Load drag coefficient CSV data"""
        if os.path.exists(self.cfg.csv_path):
            self.drag_df = pd.read_csv(self.cfg.csv_path)

            if 'Drag_Value' in self.drag_df.columns:
                self.id_to_drag = dict(zip(self.drag_df['ID'], self.drag_df['Drag_Value']))
            elif 'Cd' in self.drag_df.columns:
                self.id_to_drag = dict(zip(self.drag_df['ID'], self.drag_df['Cd']))
            else:

                self.id_to_drag = dict(zip(self.drag_df.iloc[:, 0], self.drag_df.iloc[:, 1]))
        else:
            print(f"Warning: Drag CSV not found at {self.cfg.csv_path}")
            self.drag_df = None
            self.id_to_drag = {}
    
    def _load_drag_data_item(self, index: int) -> Dict[str, Any]:
        """Load drag coefficient data - aligned with dragdec/drag_dataset.py"""

        uid_path = self.uids[index]
        uid = os.path.basename(uid_path).replace('.npz', '')
        
        shape_id = uid_path.split('/')[-1].replace('.npz', '') if isinstance(uid_path, str) else str(uid_path)
        
        drag_coefficient = self.id_to_drag.get(shape_id, 0.0)
        
        return {"drag_coefficient": np.float32(drag_coefficient)}

    def get_data(self, index):
        ret = self._load_shape(index)
        if self.cfg.load_supervision:
            ret.update(self._load_shape_supervision(index,ret['data']))
        drag_data = self._load_drag_data_item(index)
        ret.update(drag_data)
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

    def check_alignment_stats(self):
        total_uids = len(self.uids)
        condition_dir = os.path.join(self.cfg.condition_path, self.cfg.condition_type)
        
        condition_files = set()
        if os.path.exists(condition_dir):
            for ext in ['.png', '.jpg']:
                condition_files.update([
                    f.replace(ext, '') for f in os.listdir(condition_dir) 
                    if f.endswith(ext)
                ])
        
        shape_ids = set([uid.split('/')[-1].replace('.npz', '') for uid in self.uids])
        
        matched = condition_files.intersection(shape_ids)
        missing_conditions = shape_ids - condition_files
        extra_conditions = condition_files - shape_ids
        
        if missing_conditions and len(missing_conditions) <= 10:
            pass
        
        return {
            'total_uids': total_uids,
            'total_conditions': len(condition_files),
            'matched': len(matched),
            'missing_conditions': len(missing_conditions),
            'extra_conditions': len(extra_conditions)
        }

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

class DrivarNetPlusDatasetDemo(Dataset):
    def __init__(self, cfg: Any, split: str = "test") -> None:
        super().__init__()
        self.cfg: DrivarNetPlusDataDemoModuleConfig = cfg
        self.split = split
        
        try:
            import torchvision.transforms as transforms
            self.image_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        except ImportError:
            print("Warning: torchvision not available, using basic transform")
            self.image_transform = None
        
        self.condition_images = self._load_demo_condition_images()

    def _load_demo_condition_images(self):
        """Load demo condition images from the demo directory"""
        condition_images = []
        condition_dir = os.path.join(self.cfg.condition_path, self.cfg.condition_type)
        
        if not os.path.exists(condition_dir):
            print(f"Warning: Demo condition directory not found: {condition_dir}")
            return []
        

        for filename in os.listdir(condition_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(condition_dir, filename)
                condition_images.append({
                    'path': image_path,
                    'filename': filename,
                    'shape_id': os.path.splitext(filename)[0]
                })
        
        condition_images.sort(key=lambda x: x['filename'])
        return condition_images

    def __len__(self):
        return len(self.condition_images)

    def _load_condition_image(self, image_path) -> torch.Tensor:
        """Load condition image from path"""
        if not os.path.exists(image_path):
            print(f"Error: Condition image not found: {image_path}")
            return torch.zeros(3, 224, 224)
        
        try:
            image = Image.open(image_path).convert('RGB')
            
            if self.image_transform is not None:
                image = self.image_transform(image)
            else:
                image = image.resize((224, 224))
                image = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                image = (image - mean) / std
            
            return image
            
        except Exception as e:
            print(f"Error: Failed to load condition image {image_path}: {e}")

            return torch.zeros(3, 224, 224)

    def _create_dummy_shape_data(self, shape_id):
        """Create dummy shape data for demo (since we only have condition images)"""

        seed = self.cfg.seed
        

        rng = np.random.default_rng(seed)
        
        dummy_points = rng.standard_normal((self.cfg.num_points, 3)).astype(np.float32) * 0.5
        
        return {
            "coarse_surface": dummy_points,
            "sharp_surface": dummy_points,
        }
        
    def _load_drag_data_item(self, shape_id) -> Dict[str, Any]:
        """Load drag coefficient data - aligned with dragdec/drag_dataset.py"""

        uid = int(shape_id.split('_')[1])
        

        if uid in [1, 4, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17]:
            drag_coefficient = 0.280
        elif uid in [3, 5, 18]:
            drag_coefficient = 0.269
        elif uid in [2, 14]:
            drag_coefficient = 0.234
        else:
            drag_coefficient = 0.190
        
        return {"drag_coefficient": np.float32(drag_coefficient)}

    def __getitem__(self, index):
        try:
            image_info = self.condition_images[index]
            condition_image = self._load_condition_image(image_info['path'])
            

            dummy_shape_data = self._create_dummy_shape_data(image_info['shape_id'])
            drag_data = self._load_drag_data_item(image_info['shape_id'])
            
            ret = {
                "uid": f"demo_{image_info['shape_id']}",
                "condition_image": condition_image,
                "shape_id": image_info['shape_id'],
                "image_path": image_info['path'],
                "drag_coefficient": drag_data["drag_coefficient"],
                **dummy_shape_data
            }
            
            return ret
            
        except Exception as e:
            print(f"Error in demo dataset index {index}: {e}")

            return {
                "uid": f"demo_fallback_{index}",
                "condition_image": torch.zeros(3, 224, 224),
                "shape_id": f"fallback_{index}",
                "image_path": "",
                "coarse_surface": np.zeros((self.cfg.num_points, 3), dtype=np.float32),
                "sharp_surface": np.zeros((self.cfg.num_points, 3), dtype=np.float32),
            }

    def collate(self, batch):
        batch = torch.utils.data.default_collate(batch)
        return batch

class DrivarNetPlusDataDemoModule(pl.LightningDataModule):
    cfg: DrivarNetPlusDataDemoModuleConfig

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        super().__init__()
        self.cfg = parse_structured(DrivarNetPlusDataDemoModuleConfig, cfg)

    def setup(self, stage=None) -> None:
        if stage in [None, "test", "predict"]:
            self.test_dataset = DrivarNetPlusDatasetDemo(self.cfg, "test")
        

        if stage in [None, "fit"]:

            self.train_dataset = None
        if stage in [None, "fit", "validate"]:
            self.val_dataset = None

    def prepare_data(self):
        condition_dir = os.path.join(self.cfg.condition_path, self.cfg.condition_type)
        if not os.path.exists(condition_dir):
            print(f"Warning: Demo condition directory not found: {condition_dir}")

    def general_loader(self, dataset, batch_size, collate_fn=None, num_workers=0) -> DataLoader:
        if dataset is None:
            return None
        return DataLoader(
            dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=num_workers
        )

    def train_dataloader(self) -> DataLoader:

        if hasattr(self, 'train_dataset') and self.train_dataset is not None:
            return self.general_loader(
                self.train_dataset,
                batch_size=self.cfg.batch_size,
                collate_fn=self.train_dataset.collate,
                num_workers=self.cfg.num_workers
            )
        return None

    def val_dataloader(self) -> DataLoader:
        if hasattr(self, 'val_dataset') and self.val_dataset is not None:
            return self.general_loader(self.val_dataset, batch_size=1, num_workers=self.cfg.num_workers)
        return None

    def test_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.test_dataset, 
            batch_size=self.cfg.batch_size, 
            collate_fn=self.test_dataset.collate,
            num_workers=self.cfg.num_workers
        )

    def predict_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.test_dataset, 
            batch_size=self.cfg.batch_size,
            collate_fn=self.test_dataset.collate,
            num_workers=self.cfg.num_workers
        )
