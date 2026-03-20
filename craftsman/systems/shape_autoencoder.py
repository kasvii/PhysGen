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
from dataclasses import dataclass, field
import numpy as np
import torch
from skimage import measure
from einops import repeat, rearrange

import craftsman
from craftsman.systems.base import BaseSystem
from craftsman.utils.ops import generate_dense_grid_points
from craftsman.utils.typing import *
from craftsman.utils.misc import get_rank
import torch.distributed as dist
import os
from craftsman.models.geometry.utils import Mesh
import trimesh
from torchvision.utils import save_image,make_grid
from craftsman.systems.utils import *
import time


@craftsman.register("shape-autoencoder-system")
class ShapeAutoEncoderSystem(BaseSystem):
    @dataclass
    class Config(BaseSystem.Config):
        shape_model_type: str = None
        shape_model: dict = field(default_factory=dict)
        sample_posterior: bool = True
        save_mesh: bool = False

    cfg: Config

    def configure(self):
        super().configure()

        self.shape_model = craftsman.find(self.cfg.shape_model_type)(self.cfg.shape_model)

    def _maybe_save_mesh(self, batch: Dict[str, Any], latents: torch.Tensor) -> None:
        if not self.cfg.save_mesh:
            return
        if self.trainer is not None and not self.trainer.is_global_zero:
            return

        uid = os.path.basename(batch["uid"][0]).replace(".npz", "")
        mesh_rel_path = f"it{self.true_global_step}/{uid}.obj"
        mesh_abs_path = self.get_save_path(mesh_rel_path)
        if os.path.exists(mesh_abs_path):
            return

        try:
            save_slice_dir = self.get_save_path(f"it{self.true_global_step}/{uid}")
            mesh_v_f, _ = self.shape_model.extract_geometry_by_diffdmc(
                latents,
                octree_depth=9,
                save_slice_dir=save_slice_dir,
            )
            self.save_mesh(mesh_rel_path, mesh_v_f[0][0], mesh_v_f[0][1])
        except Exception as e:
            print(f"Failed to save mesh for {uid}: {e}")

    def forward(self, batch: Dict[str, Any],split: str) -> Dict[str, Any]:
        num = batch["number_sharp"][0].item()
        rand_points = batch["rand_points"] 
        if "sdf" in batch:
                target = batch["sdf"]
                coarse_target = target[:,num:]
                sharp_target = target[:,:num]
                criteria = torch.nn.MSELoss()
        elif "occupancies" in batch:
            target = batch["occupancies"] 
            coarse_target = target[:,num:]
            sharp_target = target[:,:num]
            criteria = torch.nn.BCEWithLogitsLoss()
        else:
            raise NotImplementedError
        _, latents, posterior, logits, mean_value, variance_value = self.shape_model(
            batch["coarse_surface"],
            batch["sharp_surface"],
            rand_points, 
            sample_posterior=self.cfg.sample_posterior,
            split= split
        )

        coarse_logits = logits[:,num:]
        sharp_logits = logits[:,:num]

        outputs = {
            "loss_coarse_logits": criteria(coarse_logits, coarse_target).mean(),
            "loss_sharp_logits": criteria(sharp_logits, sharp_target).mean(),
            "overall_logits": logits,
            "coarse_logits": coarse_logits,
            "sharp_logits": sharp_logits,
            "overall_target": target,
            "coarse_target": coarse_target,
            "sharp_target": sharp_target,
            "latents": latents
        }

        if self.cfg.sample_posterior:
            loss_kl = posterior.kl()
            loss_kl = torch.sum(loss_kl) / loss_kl.shape[0]
            outputs["loss_kl"] = loss_kl

        return outputs

    def training_step(self, batch, batch_idx):
        out = self(batch,'train')

        loss = 0.
        for name, value in out.items():
            if name.startswith("loss_"):
                self.log(f"train/{name}", value)
                loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        return {"loss": loss}

    def _compute_metrics(self, out: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Compute accuracy and IoU metrics for overall, coarse, and sharp predictions."""
        threshold = 0
        metrics = {}

        for prefix in ("overall", "coarse", "sharp"):
            outputs = out[f"{prefix}_logits"]
            labels = out[f"{prefix}_target"]
            labels = (labels >= threshold).float()
            pred = torch.zeros_like(outputs)
            pred[outputs >= threshold] = 1

            accuracy = (pred == labels).float().sum(dim=1) / labels.shape[1]
            metrics[f"{prefix}_accuracy"] = accuracy.mean()

            intersection = (pred * labels).sum(dim=1)
            union = (pred + labels).gt(0).sum(dim=1)
            iou = intersection * 1.0 / union + 1e-5
            metrics[f"{prefix}_iou"] = iou.mean()

        return metrics

    def _log_metrics(self, metrics: Dict[str, torch.Tensor], prefix: str = "val"):
        """Log all metrics under the given prefix."""
        for name, value in metrics.items():
            self.log(f"{prefix}/{name}", value, sync_dist=True, on_epoch=True)

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        self.eval()
        out = self(batch, 'val')
        self._maybe_save_mesh(batch, out["latents"])

        metrics = self._compute_metrics(out)
        self._log_metrics(metrics, prefix="val")

        torch.cuda.empty_cache()
        return {"val/loss": out["loss_sharp_logits"]}
    
    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        self.eval()
        out = self(batch, 'val')
        self._maybe_save_mesh(batch, out["latents"])

        metrics = self._compute_metrics(out)
        self._log_metrics(metrics, prefix="val")

        torch.cuda.empty_cache()
        return {"val/loss": out["loss_sharp_logits"]}

    def on_validation_epoch_end(self):
        pass
