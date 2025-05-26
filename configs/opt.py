import dataclasses
from typing import Union, Optional, Tuple
from dataclasses import field

from configs.dataset import *
from configs.first_stage import *
from configs.second_stage import *
from configs.task import *
from tools.submit_utils import Slurm


@dataclasses.dataclass(kw_only=True)
class FirstStage:
    # Slurm: Slurm configuration for job submission
    slurm: Slurm
    # Model configuration for first stage training
    model: LatentTree
    # Dataset configuration for training
    data: Front3D

    # str: Root directory for log files
    log_dir: str = "./lt3sd"
    # Optional[str]: Path to resume from previous run
    resume: Optional[str] = None
    # float: Learning rate for optimization
    learning_rate: float = 1.0e-4
    # int: Maximum number of training steps
    max_steps: int = 100000  # -1
    # bool: Whether to use Weights & Biases logging
    use_wandb: bool = True
    # Optional[Tuple[str, ...]]: Training levels for hierarchical structure
    levels: Optional[Tuple[str, ...]] = None
    # int: Number of GPUs to use (-1 for all available)
    devices: int = -1
    # str: Training strategy (ddp for distributed training)
    strategy: str = "ddp"
    # int: Frequency of validation checks in epochs
    check_val_every_n_epoch: int = 1
    # int: Random seed for reproducibility
    seed: int = 16


@dataclasses.dataclass(kw_only=True)
class SecondStage:
    # Slurm: Slurm configuration for job submission
    slurm: Slurm
    # Dataset configuration for training
    data: Front3D
    # DDPM: Model configuration for second stage
    model: DDPM
    task: Union[Train, Generation] = field(default_factory=Train)

    # Directory of first stage log
    first_stage_dir: str = "./"
    # Optional[str]: Path to resume training from checkpoint
    resume: Optional[str] = None
    # Optional[str]: Directory for saving logs
    log_dir: Optional[str] = None
    # int: Number of samples per batch
    batch_size: int = 8
    # int: Number of data loading workers
    num_workers: int = 8

    # float: Learning rate for optimization
    learning_rate: float = 1.0e-4
    # bool: Enable CUDNN benchmarking
    benchmark: bool = True
    # int: Maximum number of training epochs
    max_epochs: int = -1
    # int: Maximum number of steps (-1 for no limit)
    max_steps: int = 800000
    # int: Number of GPUs to use (-1 for all available)
    devices: int = -1
    # str: Training strategy (ddp for distributed training)
    strategy: str = "ddp"
    # str: Numerical precision for training
    precision: str = "32-true"
    # str: Type of profiler to use
    profiler: str = "simple"
    # int: Frequency of validation checks in epochs
    check_val_every_n_epoch: int = 2
    # Optional[Union[int, float]]: Validation check interval
    val_check_interval: Optional[Union[int, float]] = None
    # bool: Whether to use Weights & Biases logging
    use_wandb: bool = True
    # Optional[Tuple[str, ...]]: Training levels for hierarchical structure
    levels: Optional[Tuple[str, ...]] = None
    # int: Random seed for reproducibility
    seed: int = 16
