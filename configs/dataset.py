import dataclasses
from typing import Literal, Tuple


@dataclasses.dataclass(kw_only=True)
class Front3D:
    name: str = "3D-FRONT"
    root_dir: str = "./mydata"
    split: Literal["train", "val", "test", "all"] = "all"
    max_sample: int = -1  # -1 for all
    category: Tuple[str, ...] = ("House",)
    target: str = "data.dataset.Dataset"
    augmentation: bool = True
    importance_sampling: bool = True
    chunk_shape: Tuple[int, int, int] = (32, 16, 32)
    voxel_size: float = 0.02
    padding: int = 20000  # -1 not padding
    truncation: float = 0.1  # 0.022 * 3 < 0.1 < 0.088 * 3
