import tyro
import os
import yaml
import torch
import dataclasses
import datetime
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
import pytorch_lightning as pl

from tools import common_utils, submit_utils
from tools.logger import get_logger
from configs.opt import FirstStage
from tools.lightning import rank_zero_only_context, CustomProgressBar

logger = get_logger(file_name=__file__)


class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, data, level: str):
        super().__init__()
        self.level = level

        # Initialize voxel_sizes based on level
        _, vxl_size_i, vxl_size_o = common_utils.parse_level(level)
        data |= {"voxel_size": vxl_size_i}

        data = data.copy()
        chunk_shape = data["chunk_shape"]

        # Training dataset
        # Adjust chunk shape based on voxel size
        factor = min(2, int(round(vxl_size_o / vxl_size_i)))
        data["chunk_shape"] = [factor * x for x in chunk_shape]
        data["split"] = "train"
        self.train_dataset = common_utils.instantiate_from_config(data)

        # Validation dataset
        factor = min(4, int(round(vxl_size_o / vxl_size_i)))
        data["chunk_shape"] = [factor * x for x in chunk_shape]
        data["split"] = "val"
        self.val_dataset = common_utils.instantiate_from_config(data)

        # Test dataset
        data["split"] = "test"
        self.test_dataset = common_utils.instantiate_from_config(data)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=4,
            num_workers=4,
            drop_last=False,
            shuffle=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset, batch_size=4, num_workers=4, drop_last=False, shuffle=True
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=4,
            num_workers=4,
            drop_last=False,
            shuffle=True,
        )


def train(
    level: str,
    model,
    data,
    log_dir: str,
    now_str: str,
    learning_rate: float = 1.0e-4,
    max_steps: int = 5e4,
    devices: int = -1,  # Number of GPUs to use (-1 for all available)
    seed: int = 16,
    use_wandb: bool = True,
    strategy: str = "ddp",  # Distributed training strategy
    check_val_every_n_epoch: int = 1,
    accelerator: str = "gpu",
):
    seed_everything(seed)
    """Train function for a single level"""
    # Setup directories
    ckpt_dir = os.path.join(log_dir, "checkpoint", level)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Initialize data module
    data_module = DataModuleFromConfig(data, level)

    # Setup logger
    wandb_logger = None
    if use_wandb:
        wandb_logger = WandbLogger(
            project="lt3sd_first_stage",
            name=f"{now_str}_level_{level}",
            save_dir=log_dir,
            id=f"{now_str}_level_{level}",
        )

    # Initialize Lightning trainer
    trainer = pl.Trainer(
        max_steps=max_steps,
        accelerator=accelerator,
        devices=devices,
        strategy=(
            DDPStrategy(find_unused_parameters=True) if strategy == "ddp" else None
        ),
        logger=wandb_logger,
        callbacks=[
            ModelCheckpoint(
                dirpath=ckpt_dir, filename=level, save_last=True, save_top_k=0
            ),
            CustomProgressBar(name=now_str),
        ],
        check_val_every_n_epoch=check_val_every_n_epoch,
        default_root_dir=None,  # This prevents lightning_logs creation
    )

    # Initialize model
    model |= {"level": level, "learning_rate": learning_rate, "log_dir": log_dir}
    model["backbone"] |= {"truncation": data["truncation"]}
    trainer.logger.log_hyperparams(model)
    model = common_utils.instantiate_from_config(model)

    with rank_zero_only_context():
        os.makedirs(os.path.join(log_dir, "model"), exist_ok=True)
        # Save model architecture
        with open(
            os.path.join(log_dir, "model", f"first_stage_model_{level}.txt"), "w"
        ) as f:
            f.writelines(repr(model.backbone))

    # Train model
    trainer.fit(model, data_module)

    # Export stats after training
    model.export_stats(iter(data_module.train_dataloader()))

    # Compute reconstruction loss
    model.compute_rec_loss(
        iter(data_module.test_dataloader()), batch_size=4, n_sample=1000
    )


if __name__ == "__main__":
    """Main function to handle training setup and execution"""
    opt = tyro.cli(FirstStage)

    fn_kwargs_share = dataclasses.asdict(opt)
    resume = fn_kwargs_share.pop("resume")
    if resume is None:
        now_str = str(datetime.datetime.now().strftime("%y%m%d-%H%M%S"))
        log_dir = opt.resume = opt.log_dir = os.path.join(opt.log_dir, now_str)
        os.makedirs(opt.log_dir, exist_ok=True)
        logger.info(f"Start a new training run in logdir: {log_dir}")
    else:
        log_dir = opt.log_dir = resume
        now_str = os.path.basename(log_dir)
        logger.info(f"Resume training run in logdir: {log_dir}")

    # Update paths
    fn_kwargs_share["model"] |= {
        "out_dir": os.path.join(log_dir, "reconstruction"),
    }
    fn_kwargs_share["model"]["backbone"] |= {
        "ckpt_dir": os.path.join(log_dir, "checkpoint"),
        "stats_dir": os.path.join(log_dir, "stats"),
    }
    logger.info(tyro.to_yaml(opt))

    # Prepare job submissions
    fn_kwargs_share |= {"now_str": now_str, "log_dir": log_dir}
    fn_kwargs_list = [{"level": level} for level in fn_kwargs_share.pop("levels")]

    # Save configuration
    with rank_zero_only_context():
        with open(os.path.join(log_dir, "first_stage_config.yaml"), "w") as f:
            yaml.dump(fn_kwargs_share, f)

    logger.info(f"All jobs: {len(fn_kwargs_list)}")

    slurm_kwargs = fn_kwargs_share.pop("slurm")
    # Submit jobs
    submit_utils.submit_jobs(
        fn=train,
        slurm_kwargs=slurm_kwargs,
        fn_kwargs_share=fn_kwargs_share,
        fn_kwargs_list=fn_kwargs_list,
    )
