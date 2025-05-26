import dataclasses
from typing import Literal, Optional, Tuple


@dataclasses.dataclass(kw_only=True)
class Encoder:
    target: str = "models.first_stage.common.Encoder"
    in_channels: int = 1
    ch: int = 128
    double_z: bool = True
    num_res_blocks: int = 1
    attn_resolutions: Tuple[int, ...] = (32, 16, 8)
    dropout: float = 0.1
    ch_mult: Tuple[int, ...] = (1, 1, 1, 1, 1)


@dataclasses.dataclass(kw_only=True)
class Decoder:
    target: str = "models.first_stage.common.Decoder"
    ch: int = 128
    out_ch: int = 1
    num_res_blocks: int = 1
    attn_resolutions: Tuple[int, ...] = (32, 16, 8)
    dropout: float = 0.1
    ch_mult: Tuple[int, ...] = (1, 1, 1, 1, 1)


@dataclasses.dataclass(kw_only=True)
class AE:
    encoder: Encoder
    decoder: Decoder

    name: str = "ae"
    target: str = "models.first_stage.conv.AE"
    channels: int = 4
    levels: Optional[Tuple[str, ...]] = None
    ckpt_dir: Optional[str] = None
    stats_dir: Optional[str] = None
    scale_mode: Literal["scale_by_std", "scale_by_range"] = "scale_by_std"


@dataclasses.dataclass(kw_only=True)
class VAE:
    encoder: Encoder
    decoder: Decoder

    name: str = "vae"
    target: str = "models.first_stage.conv.VAE"
    channels: int = 4
    levels: Optional[Tuple[str, ...]] = None
    stats_dir: Optional[str] = None
    scale_mode: Literal["scale_by_std", "scale_by_range"] = "scale_by_std"


@dataclasses.dataclass(kw_only=True)
class LatentTree:
    backbone: AE

    name: str = "latent_tree"
    target: str = "models.first_stage.latent_tree.LatentTree"
    rec_weight: float = 1.0e3
