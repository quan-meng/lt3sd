import torch
import h5py
import random
import os
import glob
import numpy as np
from typing import Tuple, Optional

from tools.logger import get_logger
from tools import scene_utils, common_utils

logger = get_logger(file_name=__file__, debug="dataset")


def split_file_paths(file_paths, ratios=[0.8, 0.05, 0.15], split="all", max_sample=-1):
    if max_sample > 0:  # Use subset
        file_paths = file_paths[:max_sample]
    else:
        val_nums = [int(len(file_paths) * x) for x in ratios]
        val_nums[-1] = len(file_paths) - sum(val_nums[:-1])
        if split == "train":
            file_paths = file_paths[: val_nums[0]]
        elif split == "val":
            file_paths = file_paths[val_nums[0] : val_nums[0] + val_nums[1]]
        elif split == "test":
            file_paths = file_paths[val_nums[0] + val_nums[1] :]
        elif split == "all":
            pass
        else:
            raise ValueError

    return file_paths


class Augmentations(object):
    # perform random rotation in degrees
    def rotation(self, inputs, times=[1, 2, 3]):
        assert isinstance(inputs, list)

        k = random.choice(times)
        for i in range(len(inputs)):
            inputs[i] = torch.rot90(inputs[i], k=k, dims=[-1, -3])

        return inputs

    def flip(self, inputs):
        k = random.choice([-1, -3])  # flip along x or z axis

        for i in range(len(inputs)):
            inputs[i] = torch.flip(inputs[i], dims=[k])

        return inputs

    def __call__(self, inputs):
        """
        Args:
            volume: [C, G, G, G]
        """
        if random.random() < 0.5:
            inputs = self.rotation(inputs)
        if random.random() < 0.5:
            inputs = self.flip(inputs)

        return inputs


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root_dir,
        split,
        category,
        importance_sampling=True,
        max_sample=-1,
        voxel_size: float = 0.08,
        truncation: Optional[float] = None,
        chunk_shape: Tuple[int, int, int] = (64, 64, 64),
        augmentation=False,
        padding: int = 1000,  # -1 not padding
    ):
        self.root_dir = root_dir
        self.split = split
        self.category = category
        self.max_sample = max_sample
        self.voxel_size = voxel_size
        self.chunk_shape = chunk_shape
        self.importance_sampling = importance_sampling
        self.padding = padding
        self.truncation = common_utils.default(truncation, self.voxel_size * 3)
        self.file_paths = self.list_file_paths()
        self.n_data = len(self.file_paths)

        logger.info(
            f"Dataset, split: {self.split}, len: {self.n_data}, voxel_size: {self.voxel_size}, chunk_shape: {self.chunk_shape}"
        )

        if augmentation:
            self.aug_func = Augmentations()
        else:
            self.aug_func = None

        super().__init__()

    def __len__(self):
        if self.padding > 0 and self.split in ["train", "all"]:
            repeat = max(self.padding // max(1, len(self.file_paths)), 1)
            return self.n_data * repeat
        else:
            return self.n_data

    def __getitem__(self, idx):
        if self.padding > 0 and self.split in ["train", "all"]:
            idx = idx % len(self.file_paths)
        return self.get_sample(idx)

    def list_file_paths(self):
        file_paths = []

        for k in self.category:
            txt_path = os.path.join(self.root_dir, k, f"base_names.txt")
            if not os.path.exists(txt_path):
                paths_k = sorted(
                    glob.glob(
                        os.path.join(
                            self.root_dir, k, f"udf_voxel_{self.voxel_size}", "*.npy"
                        )
                    )
                )
                basenames = [os.path.basename(x).replace(".npy", "") for x in paths_k]
                with open(txt_path, "w") as f:
                    for x in basenames:
                        f.write(f"{x}\n")

            with open(txt_path, "r") as f:
                basenames = [x.strip() for x in f.readlines()]

            file_paths_k = [
                os.path.join(
                    self.root_dir, k, f"udf_voxel_{self.voxel_size}", f"{x}.npy"
                )
                for x in basenames
            ]
            file_paths += file_paths_k

        assert len(file_paths) > 0, f"Level-{self.voxel_size} has 0 samples"

        file_paths = split_file_paths(
            file_paths, split=self.split, max_sample=self.max_sample
        )

        return file_paths

    def sample_bbox(self, scene_shape, bbox_path: Optional[str] = None):
        if random.random() < 0.8 and self.importance_sampling:
            max_height = scene_shape[1] / self.voxel_size + 1

            with h5py.File(bbox_path, "r") as h5file:
                all_bboxes = np.array(h5file["bbox"][:])  # [N, 2, 3]
                all_bboxes = np.clip(all_bboxes, a_min=0, a_max=1.0)  # [N, 2, 3]
                all_bboxes = np.flip(all_bboxes, axis=-1)

                all_bboxes = all_bboxes * scene_shape[None, None]  # [N, 2, 3]

                all_bboxes[:, 0] = np.clip(
                    np.floor(all_bboxes[:, 0]), a_min=0, a_max=None
                )
                all_bboxes[:, 1] = np.clip(
                    np.ceil(all_bboxes[:, 1]), a_min=None, a_max=scene_shape[None]
                )

            scene_shape[1] = np.clip(scene_shape[1], a_min=0, a_max=max_height)
            all_bboxes[..., 1] = np.clip(
                all_bboxes[..., 1], a_min=0, a_max=max_height
            )  # [N, 2, 3]
            all_bboxes = all_bboxes.astype(np.int32)

            pos = scene_utils.sample_chunk_in_bbox(
                scene_shape, self.chunk_shape, all_bboxes.copy()
            )
        else:
            pos = scene_utils.sample_random_chunk(scene_shape, self.chunk_shape)

        bbox_min, bbox_max = scene_utils.pose2bbox(pos, self.chunk_shape)  # [6]
        bbox_min = torch.from_numpy(np.array(bbox_min)).long()
        bbox_max = torch.from_numpy(np.array(bbox_max)).long()

        logger.debug(
            f"chunk_shape: {self.chunk_shape}, scene_shape: {scene_shape}, pos: {pos}"
        )

        return bbox_min, bbox_max

    def get_sample(self, idx):
        data_path = self.file_paths[idx]
        scene = np.load(data_path, allow_pickle=True, mmap_mode="c")
        scene_shape = np.array(scene.shape[-3:])  # [3]

        bbox_path = data_path.replace(f"udf_voxel_{self.voxel_size}", "bbox").replace(
            ".npy", ".h5"
        )
        bbox_min, bbox_max = self.sample_bbox(scene_shape, bbox_path)  # [6]

        chunk, _ = scene_utils.crop_with_bbox(
            scene, bbox_min=bbox_min, bbox_max=bbox_max, pad_value=self.truncation
        )  # [G1, G2, G3]
        chunk = torch.clamp(chunk, min=0.0, max=self.truncation)  # [g, g, g]

        if self.aug_func is not None:
            chunk = self.aug_func([chunk])[0]

        data = {
            "data_path": data_path,
            "indices": torch.from_numpy(np.array(idx)),  # tensor
            "latent": chunk[None].float(),  # [1, g, g, g]
            "voxel_size": self.voxel_size,
        }

        return data
