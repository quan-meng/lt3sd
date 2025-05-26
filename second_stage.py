import os
import datetime
import tyro
import dataclasses
import yaml
from typing import Optional, Union, List

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.trainer import Trainer
from pytorch_lightning import seed_everything
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from tools.logger import get_logger
from tools.submit_utils import submit_jobs
from configs.opt import SecondStage
from tools.common_utils import parse_level, instantiate_from_config
from tools.lightning import (
    CustomProgressBar,
    rank_zero_only_context,
    DataModuleFromConfig,
)


def train(
    level: str,
    model,  # Model configuration
    data,  # Data configuration,
    task,  # Task configuration
    learning_rate: float,  # Learning rate for optimization
    levels: Optional[List[str]] = None,
    batch_size: int = 8,  # Number of samples per batch
    num_workers: int = 8,  # Number of subprocesses for data loading
    log_dir: str = "./",  # Directory for saving logs
    check_val_every_n_epoch: int = 10,
    val_check_interval: Optional[Union[int, float]] = None,
    max_epochs: int = 16000,  # Maximum number of epochs to train
    max_steps: int = -1,  # Maximum number of steps to train (-1 for no limit)
    accelerator: str = "gpu",  # Accelerator type
    devices: int = -1,  # Number of GPUs to use (-1 for all available)
    strategy: str = "ddp",  # Distributed training strategy
    now_str: str = "",  # Current timestamp
    use_wandb: bool = True,  # Whether to use Weights & Biases for logging
    benchmark: bool = True,  # Enable cudnn benchmark for faster runtime
    profiler: str = None,  # Profiler to use
    precision: str = "32-true",  # Precision for training
    seed: int = 23,
):
    logger = get_logger(file_name=__file__)

    seed_everything(seed)

    # trainer --------------------------------------------------------------
    wandb_logger = None
    if use_wandb and task["name"] == "train":
        wandb_logger = WandbLogger(
            name=f"{now_str}_{level}",
            project="lt3sd_second_stage",
            save_dir=log_dir,
            id=f"{now_str}_{level}",
        )

    ckpt_dir = os.path.join(
        log_dir,
        "checkpoint",
        level if level != model["start_level"] else f"{level}_start",
    )
    os.makedirs(ckpt_dir, exist_ok=True)

    trainer = Trainer(
        benchmark=benchmark,
        max_epochs=max_epochs,
        precision=precision,
        max_steps=max_steps,
        accelerator=accelerator,
        devices=devices,
        strategy=(
            DDPStrategy(find_unused_parameters=True) if strategy == "ddp" else None
        ),
        val_check_interval=val_check_interval,
        check_val_every_n_epoch=check_val_every_n_epoch,
        callbacks=[
            LearningRateMonitor(logging_interval="step"),
            ModelCheckpoint(
                dirpath=ckpt_dir,
                save_last=True,
                save_top_k=0,
                every_n_train_steps=10000,
            ),
            CustomProgressBar(name=now_str),
        ],
        logger=wandb_logger,
        profiler=profiler,
        default_root_dir=log_dir,
    )

    _, level_in, level_out = parse_level(level)
    factor = int(round(level_out / level_in))

    data |= {
        "voxel_size": level_in,
        "chunk_shape": [x * factor for x in model["chunk_shape"]],
    }
    datamodule = DataModuleFromConfig(
        data,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle_val_dataloader=True,
        use_worker_init_fn=False,
    )

    model |= {
        "levels": [level] if task["name"] == "train" else levels,
        "learning_rate": learning_rate,
        "log_dir": log_dir,
        "task_hyparam": task,
    }
    model["first_stage_config"]["truncation"] = data["truncation"]
    trainer.logger.log_hyperparams(model)
    model = instantiate_from_config(model)

    if task["name"] == "train":
        ckpt_path = os.path.join(
            log_dir,
            "checkpoint",
            level if level != model.start_level else f"{level}_start",
            "last.ckpt",
        )
        if not os.path.exists(ckpt_path):
            ckpt_path = None
        trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)
    else:
        logger.info("Running generation step")
        trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    logger = get_logger(file_name=__file__)

    opt = tyro.cli(SecondStage)
    fn_kwargs_share = dataclasses.asdict(opt)

    # resume first stage from first_stage_config.yaml file
    first_stage_dir = fn_kwargs_share.pop("first_stage_dir")
    with open(
        os.path.join(first_stage_dir, "first_stage_config.yaml"), "r"
    ) as yaml_file:
        first_stage_cfg = yaml.unsafe_load(yaml_file)
        fn_kwargs_share["model"]["first_stage_config"] = first_stage_cfg["model"][
            "backbone"
        ]
        fn_kwargs_share["data"] = first_stage_cfg["data"]

    resume = fn_kwargs_share.pop("resume")
    if resume:
        if not os.path.exists(resume):
            raise ValueError("Cannot find {}".format(resume))
        assert os.path.isdir(resume), f"{resume} is not a directory!"
        logdir = resume.rstrip("/")
        now_str = logdir.split("/")[-1]
        logger.info(f"Resume training run in logdir: {logdir}")
        ckpt_dir = os.path.join(logdir, "checkpoint")
    else:
        now_str = str(datetime.datetime.now().strftime("%y%m%d-%H%M%S"))
        logdir = os.path.join(first_stage_dir, opt.model.name, now_str)
        os.makedirs(logdir, exist_ok=True)
        logger.info(f"Start a new training run in logdir: {logdir}")
        ckpt_dir = opt.model.ckpt_dir

    fn_kwargs_share |= {"log_dir": logdir, "now_str": now_str}
    fn_kwargs_share["model"]["ckpt_dir"] = ckpt_dir

    slurm_kwargs = fn_kwargs_share.pop("slurm")
    if fn_kwargs_share["task"]["name"] == "train":
        fn_kwargs_list = [{"level": level} for level in fn_kwargs_share.pop("levels")]

        # Create logdirs and save configs
        with rank_zero_only_context():
            with open(os.path.join(logdir, "second_stage_config.yaml"), "w") as f:
                yaml.dump(fn_kwargs_share, f)
    else:
        fn_kwargs_share["level"] = fn_kwargs_share["levels"][-1]
        fn_kwargs_list = []
        task = fn_kwargs_share.pop("task")
        n_sample_per_node = task.pop("n_sample") // slurm_kwargs["nodes"]
        for idx in range(slurm_kwargs["nodes"]):
            task_i = task.copy()
            task_i["start_idx"] = task_i["start_idx"] + idx * n_sample_per_node
            task_i["n_sample"] = n_sample_per_node
            fn_kwargs_list.append({"task": task_i})
        logger.info(
            f"Running generation step with {len(fn_kwargs_list)} nodes with {n_sample_per_node} samples each"
        )

    submit_jobs(
        fn=train,
        slurm_kwargs=slurm_kwargs,
        fn_kwargs_share=fn_kwargs_share,
        fn_kwargs_list=fn_kwargs_list,
    )
