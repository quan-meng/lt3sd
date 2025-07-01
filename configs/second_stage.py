import dataclasses
from typing import Literal, Union, Optional, Tuple

from configs.first_stage import *


@dataclasses.dataclass
class UnetConfig:
    name: str = "unet"
    target: str = "models.modules.unet.UNetModel"
    dims: int = 3
    in_channels: int = 1
    out_channels: int = 1
    model_channels: int = 128  # 128, 192
    concat_dim: int = 0
    dropout: float = 0.1  # 0.1
    use_new_attention_order: bool = False
    num_heads_upsample: int = -1
    attention_resolutions: Tuple[int, ...] = (
        16,
        8,
        4,
    )  # (8, 4, 2) for 16, (16, 8, 4) for 32
    num_res_blocks: int = 2
    channel_mult: Tuple[int, ...] = (1, 2, 2, 4)  # 8 -> 4 -> 2
    use_scale_shift_norm: bool = True
    num_head_channels: int = 32  # num_heads: int = 8
    resblock_updown: bool = True
    use_spatial_transformer: bool = False
    context_dim: Optional[int] = None


@dataclasses.dataclass(kw_only=True)
class DDPM:
    unet_config: UnetConfig
    first_stage_config: AE

    name: str = "second_stage"
    target: str = "models.second_stage.ddpm.Net"
    ckpt_dir: Optional[str] = None  # Use only ckpt_dict
    linear_start: float = 0.0015  # 0.0015
    linear_end: float = 0.0195  # 0.0195
    loss_type: Literal["l1", "l2", "l2_cos"] = "l2"
    log_every_t: int = 20
    timesteps: int = 1000
    first_stage_key: str = "latent"
    parameterization: Literal["eps", "x0"] = "eps"
    monitor: str = "val/loss_simple_ema"
    cond_key: Optional[str] = None
    use_scheduler: bool = True
    chunk_shape: Tuple[int, int, int] = (32, 16, 32)
    levels: Optional[Tuple[str, ...]] = None

    start_level: Optional[str] = "32_16"
    ema_update_batch: int = 100  # 100
