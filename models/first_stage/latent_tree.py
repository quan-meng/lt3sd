import torch
import pytorch_lightning as pl
import os
from typing import Dict
from tqdm import tqdm
import numpy as np
import shutil

from tools import common_utils, mesh_utils
from tools.logger import get_logger
from models.first_stage.get_stats import get_stats

logger = get_logger(file_name=__file__)


class LatentTree(pl.LightningModule):
    def __init__(
        self,
        backbone: Dict,
        level: str,
        learning_rate: float = 1e-4,
        rec_weight: float = 1.0e2,
        log_dir: str = "./",
        stats_dir: str = "./",
        out_dir: str = "./",
    ):
        super().__init__()
        self.save_hyperparameters()

        # Initialize the original model as backbone
        backbone |= {"levels": [level], "stats_dir": stats_dir}
        self.backbone = common_utils.instantiate_from_config(backbone)
        self.level = level
        self.learning_rate = learning_rate
        self.log_dir = log_dir
        self.stats_dir = stats_dir
        self.out_dir = out_dir
        self.rec_weight = rec_weight
        os.makedirs(self.stats_dir, exist_ok=True)
        os.makedirs(self.out_dir, exist_ok=True)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """Training step"""
        all_loss, loss_dict = self.backbone.compute_loss(
            target=batch["latent"], level=self.level, rec_weight=self.rec_weight
        )

        return {"loss": all_loss, "loss_dict": loss_dict}

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # Log training metrics
        self.log_dict(
            {f"train/{k}": v.item() for k, v in outputs["loss_dict"].items()},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch["latent"].shape[0],
        )

    @torch.no_grad()
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """Validation step"""
        target = batch["latent"]
        all_loss, loss_dict = self.backbone.compute_loss(
            target=target, level=self.level, rec_weight=self.rec_weight
        )

        # Log validation metrics
        self.log_dict(
            {f"val/{k}": v.item() for k, v in loss_dict.items()},
            on_step=True,
            on_epoch=True,
            batch_size=batch["latent"].shape[0],
        )

        # Save reconstruction visualization periodically
        if batch_idx == 0:
            """Save reconstruction visualizations"""
            mode, vxl_size_i, _ = common_utils.parse_level(self.level)
            threshold = common_utils.ada_threshold(vxl_size_i, mode)
            threshold = min(self.backbone.truncation * 0.8, threshold)

            # Save ground truth
            mesh = mesh_utils.volume_to_mesh(target[0, 0], threshold=threshold)
            mesh.export(os.path.join(self.out_dir, f"gt_{self.level}.ply"))

            # Save reconstruction
            x_rec, _ = self.backbone(target, self.level)
            mesh = mesh_utils.volume_to_mesh(x_rec[0, 0], threshold=threshold)
            mesh.export(os.path.join(self.out_dir, f"rec_{self.level}.ply"))

        return all_loss

    def configure_optimizers(self):
        """Configure optimizers"""
        return torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, betas=(0.9, 0.99)
        )

    @torch.no_grad()
    def export_stats(self, train_loader):
        """Export statistics after training and clean up temporary files"""
        tmp_dir = os.path.join(self.log_dir, "tmp", self.level)
        os.makedirs(tmp_dir, exist_ok=True)

        num_latent = 100
        file_paths = []
        idx = 0

        self.eval()
        for i in tqdm(range(num_latent), desc=f"Exporting {num_latent} latents..."):
            data = next(train_loader)
            target = data["latent"].to(self.device)
            latent = self.backbone.encode_H(target, self.level)

            for j in range(latent.shape[0]):
                file_path = os.path.join(tmp_dir, f"{idx}.npy")
                file_paths.append(file_path)
                np.save(file_path, latent[j].cpu().numpy())
                idx += 1

        logger.info("Calculating statistics...")
        stats_dir = os.path.join(self.log_dir, "stats", self.level)
        os.makedirs(stats_dir, exist_ok=True)

        get_stats(file_paths, stats_dir=stats_dir)

        # Clean up temporary directory after computing stats
        logger.info(f"Cleaning up temporary directory: {tmp_dir}")
        shutil.rmtree(tmp_dir)

    @torch.no_grad()
    def compute_rec_loss(self, test_loader, n_sample=1000):
        """Compute reconstruction loss on test set"""
        self.eval()
        all_loss = []

        for i in tqdm(
            test_loader,
            total=len(test_loader),
            desc=f"Computing Rec Loss: Level-{self.level}",
        ):
            data = next(test_loader)
            target = data["latent"].to(self.device)

            _, loss_dict = self.backbone.compute_loss(target=target, level=self.level)
            loss_rec = loss_dict["loss_rec"].item()
            all_loss.append(loss_rec)

        all_loss = np.mean(all_loss)

        # Save reconstruction loss
        out_dir = os.path.join(self.log_dir, "rec_loss")
        os.makedirs(out_dir, exist_ok=True)

        out_path = os.path.join(out_dir, f"rec_loss_{self.level}.txt")
        with open(out_path, "w") as f:
            f.write(f"Rec Loss: {all_loss}")
