import dataclasses
from typing import Union, Optional, Tuple


@dataclasses.dataclass(kw_only=True)
class Train:
    name: str = "train"
    export_mesh: bool = True


@dataclasses.dataclass(kw_only=True)
class Generation:
    name: str = "generation"
    # int: Starting index for generation
    start_idx: int = 0
    # int: Number of samples to generate
    n_sample: int = 10

    overlap: Union[int, float] = 0.5
    # bool: Whether to use chunked generation
    with_chunk: bool = False
    # bool: Whether to use diffusion in generation
    with_diffusion: bool = False
    # Optional[Tuple[int, int, int]]: Shape of the generated scene
    scene_shape: Optional[Tuple[int, int, int]] = (256, 128, 256)
