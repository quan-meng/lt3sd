from typing import Any
from functools import partial
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from contextlib import contextmanager
import torch.distributed as dist
from pytorch_lightning.callbacks import TQDMProgressBar

from tools.common_utils import instantiate_from_config


@contextmanager
def rank_zero_only_context():
    if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0:
        yield
    else:
        yield None


class CustomProgressBar(TQDMProgressBar):
    def __init__(self, name, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.name = name

    def on_train_epoch_start(self, trainer, *args, **kwargs):
        super().on_train_epoch_start(trainer, *args, **kwargs)
        self.train_progress_bar.set_description(
            f"Name-{self.name} Epoch {trainer.current_epoch}"
        )


def worker_init_fn(worker_id):
    return np.random.seed(np.random.get_state()[1][0] + worker_id)


class KwargsDataloader:
    def __iter__(self):
        yield None


class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(
        self,
        data_config,
        batch_size,
        num_workers,
        use_worker_init_fn=False,
        shuffle_val_dataloader=False,
        splits=["train", "val", "test"],
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_worker_init_fn = use_worker_init_fn
        self.data_config = data_config
        self.dataset_configs = {}
        for split in splits:
            config_dict = data_config.copy()
            config_dict.update({"split": split})
            self.dataset_configs.update({split: config_dict})

        if "train" in splits:
            self.train_dataloader = self._train_dataloader
        if "val" in splits:
            self.val_dataloader = partial(
                self._val_dataloader, shuffle=shuffle_val_dataloader
            )
        if "test" in splits:
            self.test_dataloader = partial(self._test_dataloader, shuffle=False)

        self.datasets = dict(
            (k, instantiate_from_config(self.dataset_configs[k]))
            for k in self.dataset_configs
        )

    def _train_dataloader(self):
        init_fn = worker_init_fn if self.use_worker_init_fn else None
        return DataLoader(
            self.datasets["train"],
            batch_size=self.batch_size,
            worker_init_fn=init_fn,
            drop_last=True,
            num_workers=self.num_workers,
            pin_memory=True,
            prefetch_factor=16,
            persistent_workers=True,
        )

    def _val_dataloader(self, shuffle=False):
        init_fn = worker_init_fn if self.use_worker_init_fn else None
        return DataLoader(
            self.datasets["val"],
            batch_size=min(self.batch_size, len(self.datasets["val"])),
            num_workers=2,
            worker_init_fn=init_fn,
            shuffle=shuffle,
            pin_memory=True,
            prefetch_factor=16,
            persistent_workers=True,
            drop_last=True,
        )

    def _test_dataloader(self, shuffle=False):
        init_fn = worker_init_fn if self.use_worker_init_fn else None
        return DataLoader(
            self.datasets["test"],
            batch_size=min(self.batch_size, len(self.datasets["test"])),
            num_workers=2,
            worker_init_fn=init_fn,
            shuffle=shuffle,
            pin_memory=True,
            prefetch_factor=16,
            persistent_workers=True,
            drop_last=True,
        )
