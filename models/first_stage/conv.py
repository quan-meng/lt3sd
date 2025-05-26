import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from tools.common_utils import parse_level, default
from tools import common_utils
from tools.logger import get_logger

logger = get_logger(file_name=__file__, debug="conv")


class Layer(nn.Module):
    def __init__(self, encoder, decoder):
        super(Layer, self).__init__()
        self.encoder = common_utils.instantiate_from_config(encoder)
        self.decoder = common_utils.instantiate_from_config(decoder)


class Base(nn.Module):
    def __init__(
        self,
        name="latent_tree",
        ckpt_dir=None,
        stats_dir="./",
        levels=None,
        scale_mode="scale_by_std",
        truncation: Optional[float] = None,
    ):
        super().__init__()
        self.name = name
        self.ckpt_dir = ckpt_dir
        self.scale_mode = scale_mode
        self.levels = levels
        self.truncation = truncation

        self.get_stats(stats_dir)

    def get_stats(self, stats_dir):
        self.factors, self.mean_vals, self.scale_vals = [], {}, {}
        for level in self.levels:
            mode, vxl_size_i, vxl_size_o = parse_level(level)
            factor = int(round(vxl_size_o / vxl_size_i))
            truncation = default(self.truncation, vxl_size_i * 3.0)
            self.factors.append(factor)

            if os.path.exists(os.path.join(stats_dir, level, f"statistics.npz")):
                stats = np.load(
                    os.path.join(stats_dir, level, f"statistics.npz"), allow_pickle=True
                )
                latent_scale_val = 1.0 / torch.from_numpy(stats["std"])
                mean_val = torch.from_numpy(stats["mean_val"])
                if mode == "tsdf":
                    geo_scale_val = torch.ones((1,)) / truncation
                    geo_mean_val = torch.zeros((1,))
                elif mode == "tudf":
                    geo_scale_val = torch.ones((1,)) * 2.0 / truncation
                    geo_mean_val = torch.ones((1,)) * truncation / 2.0
                else:
                    raise NotImplementedError(f"Mode {mode} not implemented")

                mean_val = torch.cat((geo_mean_val, mean_val)).view(1, -1, 1, 1, 1)
                scale_val = torch.cat((geo_scale_val, latent_scale_val)).view(
                    1, -1, 1, 1, 1
                )
                self.mean_vals[level] = mean_val
                self.scale_vals[level] = scale_val
            else:
                continue

    def normalize(self, latent, level=None):
        device = latent.device
        mean_val = self.mean_vals[level].to(device)
        scale_val = self.scale_vals[level].to(device)
        return (latent - mean_val) * scale_val

    def denormalize(self, latent, level=None):
        device = latent.device
        mean_val = self.mean_vals[level].to(device)
        scale_val = self.scale_vals[level].to(device)
        return latent / scale_val + mean_val

    def resume(self, ckpt_dir):
        for level in self.levels:
            try:
                ckpt_path = os.path.join(ckpt_dir, level, f"last.ckpt")
                sd = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
                sd = {
                    k[len(f"backbone.models.{level}.") :]: v
                    for k, v in sd["state_dict"].items()
                }
            except:
                ckpt_path = os.path.join(ckpt_dir, f"{level}.pth")
                sd = torch.load(ckpt_path, map_location=lambda storage, loc: storage)

            missing, unexpected = self.models[level].load_state_dict(sd, strict=True)
            logger.info(
                f"Resume First Stage Model of level {level} with {len(missing)} missing and {len(unexpected)} unexpected keys"
            )

    @staticmethod
    def encode_L(x, level):
        _, vxl_size_i, vxl_size_o = parse_level(level)
        kernel_size = int(round(vxl_size_o / vxl_size_i))
        return F.avg_pool3d(x, kernel_size=kernel_size)  # [B, 1, g, g, g]

    def encode_H(self, x, level):
        return self.models[level].encoder(x)[0]

    def encode(self, x, level):
        L = self.encode_L(x, level)
        H = self.encode_H(x, level)
        return torch.cat((L, H), dim=1)

    def decode(self, x, level):
        """
        Returns:
            x: [B, 1, G, G, G]
        """
        return self.models[level].decoder(x)

    def get_channels(self, level):
        return self.channels[level]


class AE(Base):
    def __init__(self, encoder, decoder, channels=None, **kwargs):
        super(AE, self).__init__(**kwargs)

        self.models = {}
        for factor, level in zip(self.factors, self.levels):
            model = Layer(
                encoder={"z_channels": channels, "factor": factor, **encoder},
                decoder={"in_channels": channels + 1, "factor": factor, **decoder},
            )
            self.models.update({level: model})
        self.models = nn.ModuleDict(self.models)

        self.channels = {
            level: self.models[level].encoder.z_channels for level in self.levels
        }

        logger.info(f"First Stage Model Channels: {self.channels}")
        for level in self.levels:
            logger.info(
                f"Level-{level}, First Stage, Encoder size: {common_utils.get_model_size(self.models[level].encoder)}, Decoder size: {common_utils.get_model_size(self.models[level].decoder)}"
            )

    def forward(self, x, level):
        z = self.encode(x, level)
        dec = self.decode(z, level)

        return dec, None

    def compute_loss(self, target, level, rec_weight=1.0e1, **kwargs):
        x_rec, _ = self(target, level)
        rec_loss = F.mse_loss(x_rec, target, reduction="mean") * rec_weight
        loss = rec_loss

        log = {
            "loss_rec": rec_loss.detach().mean(),
        }

        return loss, log


class VAE(AE):
    def __init__(self, **kwargs):
        super(VAE, self).__init__(**kwargs)

    def encode_H(self, x, level, sample=False):
        mean, logvar = self.models[level].encoder(x)

        if sample:
            logvar = torch.clamp(logvar, -30.0, 20.0)
            out = mean + torch.randn_like(mean) * torch.exp(0.5 * logvar)
            return out, (mean, logvar)
        else:
            return mean

    def forward(self, x, level, sample=False):
        geo = self.encode_L(x, level)

        if sample:
            z, (mean, logvar) = self.encode_H(x, level, sample=True)
        else:
            mean = logvar = None
            z = self.encode_H(x, level)

        dec = self.decode(torch.cat((geo, z), dim=1), level)

        return dec, (mean, logvar)

    def compute_loss(
        self, target, level, rec_weight=1.0e1, kl_weight=1.0e-2, **unused_kwargs
    ):
        x_rec, (mu, logvar) = self(target, level, sample=True)
        rec_loss = F.mse_loss(x_rec, target, reduction="mean") * rec_weight
        kl_loss = (
            0.5
            * torch.mean(torch.pow(mu, 2) + torch.exp(logvar) - 1.0 - logvar)
            * kl_weight
        )

        loss = rec_loss + kl_loss

        log = {
            "loss_total": loss.detach().mean(),
        }

        return loss, log
