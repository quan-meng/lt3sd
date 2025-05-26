import torch
import torch.nn as nn
import numpy as np
import os
import copy
from tqdm import tqdm
import glob
import pytorch_lightning as pl
from typing import Tuple, Union, Optional, Dict
from contextlib import contextmanager
from functools import partial
from torch.optim.lr_scheduler import StepLR
from pytorch_lightning import seed_everything

from models.modules.ema import LitEma
from models.second_stage.ddim import DDIMSampler
from models.second_stage.fusion import fusion, Scene
from tools import mesh_utils, common_utils
from models.modules.util import make_beta_schedule, extract_into_tensor, noise_like
from tools.lightning import rank_zero_only_context


from tools.logger import get_logger

# import torchvision
# import trimesh
# from tools.metrics_utils import render_mesh, compute_fid_2d, cam_center_horiz_rot

logger = get_logger(file_name=__file__, debug="ddpm")


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class DiffusionWrapper(nn.Module):
    def __init__(
        self, unet_config, channels=None, concat_channels=None, cond_keys=None
    ):
        super().__init__()
        self.cond_keys = cond_keys

        self.diffusion_model = {}
        for level, channel in channels.items():
            unet_config_i = copy.deepcopy(unet_config)
            unet_config_i["in_channels"] = unet_config_i["out_channels"] = channel
            unet_config_i["concat_dim"] = concat_channels[level]
            self.diffusion_model[level] = common_utils.instantiate_from_config(
                unet_config_i
            )

        self.diffusion_model = nn.ModuleDict(self.diffusion_model)

    def forward(self, x, t, c_concat: list = None, c_crossattn: list = None, **kwargs):
        level = kwargs.pop("level")

        cond_key = self.cond_keys[level]
        model = self.diffusion_model[level]

        if cond_key is None:
            out = model(x, t, **kwargs)
        elif cond_key == "concat":
            x = torch.cat([x] + c_concat, dim=1)
            out = model(x, t, **kwargs)
        elif cond_key == "crossattn":
            cc = torch.cat(c_crossattn, 1)
            out = model(x, t, context=cc, **kwargs)
        elif cond_key == "hybrid":
            x = torch.cat([x] + c_concat, dim=1)
            cc = torch.cat(c_crossattn, 1)
            out = model(x, t, context=cc, **kwargs)
        else:
            raise NotImplementedError()

        return out


class Net(pl.LightningModule, DDIMSampler):
    # classic DDPM with Gaussian diffusion, in image space
    def __init__(
        self,
        first_stage_config: Dict,  # model config for the first stage
        unet_config: Dict,  # config for the diffusion model
        timesteps=1000,  # number of steps
        beta_schedule="linear",  # linear, cosine, fixed
        loss_type="l2",  # l2 or l1
        ckpt_dir=None,  # path to pretrained model
        ignore_keys: Tuple[str] = [],  # keys to ignore in state dict
        monitor="val/loss",
        use_ema=True,
        first_stage_key="latent",
        log_every_t=100,
        clip_denoised=False,
        linear_start=1e-4,
        linear_end=2e-2,
        cosine_s=8e-3,
        given_betas=None,
        original_elbo_weight=0.0,
        v_posterior=0.0,  # weight for choosing posterior variance as sigma = (1-v) * beta_tilde + v * beta
        l_simple_weight=1.0,
        cond_key=None,
        parameterization="eps",  # all assuming fixed variance schedules
        use_scheduler=False,
        learn_logvar=False,
        logvar_init=0.0,
        learning_rate=1e-4,
        levels=(0,),
        ema_update_batch: int = 100,
        chunk_shape: Tuple[int, int, int] = (32, 16, 32),
        start_level=None,
        log_dir: str = None,
        task_hyparam: Dict = None,
    ):
        super().__init__()
        assert parameterization in [
            "eps",
            "x0",
        ], 'currently only supporting "eps" and "x0"'
        self.parameterization = parameterization
        with rank_zero_only_context():
            logger.info(
                f"{self.__class__.__name__}: Running in {self.parameterization}-prediction mode"
            )

        assert cond_key in [None, "concat", "crossattn", "hybrid"]

        self.unet_config = unet_config
        self.clip_denoised = clip_denoised
        self.log_every_t = log_every_t
        self.first_stage_key = first_stage_key
        self.use_scheduler = use_scheduler
        self.v_posterior = v_posterior
        self.original_elbo_weight = original_elbo_weight
        self.l_simple_weight = l_simple_weight
        self.monitor = monitor
        self.clip_denoised = False
        self.loss_type = loss_type
        self.learn_logvar = learn_logvar
        self.learning_rate = learning_rate
        self.levels = levels
        self.ema_update_batch = ema_update_batch
        self.start_level = start_level
        self.log_dir = log_dir
        self.task_hyparam = task_hyparam

        self.register_schedule(
            given_betas=given_betas,
            beta_schedule=beta_schedule,
            timesteps=timesteps,
            linear_start=linear_start,
            linear_end=linear_end,
            cosine_s=cosine_s,
        )

        self.logvar = torch.full(fill_value=logvar_init, size=(self.num_timesteps,))
        if self.learn_logvar:
            self.logvar = nn.Parameter(self.logvar, requires_grad=True)

        self.chunk_shape = {}
        for i, level in enumerate(levels):
            self.chunk_shape.update({str(level): chunk_shape})

        # init first stage model
        first_stage_config.update({"levels": levels})
        self.first_stage_model = common_utils.instantiate_from_config(
            first_stage_config
        )

        # modify second stage config to match the first stage
        self.channels, self.concat_channels, self.cond_keys = (
            {k: 0 for k in self.levels},
            {k: 0 for k in self.levels},
            {k: None for k in self.levels},
        )
        for level in self.levels:
            if level == self.start_level:
                self.channels[level] = self.first_stage_model.get_channels(level) + 1
                self.concat_channels[level] = 0
                self.cond_keys[level] = cond_key
            else:
                self.channels[level] = self.first_stage_model.get_channels(level)
                self.concat_channels[level] = 1
                self.cond_keys[level] = "concat"

        logger.info(f"Second Stage Model Channels: {self.channels}")
        logger.info(f"Second Stage Model Concat Channels: {self.concat_channels}")
        logger.info(f"Second Stage Model Cond Keys: {self.cond_keys}")

        self.model = DiffusionWrapper(
            unet_config,
            channels=self.channels,
            concat_channels=self.concat_channels,
            cond_keys=self.cond_keys,
        )
        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self.model)
            with rank_zero_only_context():
                logger.info(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        if ckpt_dir is not None:
            self.resume_second_stage(ckpt_dir, ignore_keys=ignore_keys, strict=False)
        self.resume_first_stage(first_stage_config)

        for level in self.levels:
            logger.info(
                f"Level-{level}, DDPM Model Size: {common_utils.get_model_size(self.model.diffusion_model[str(level)])}"
            )

    def register_schedule(
        self,
        given_betas=None,
        beta_schedule="linear",
        timesteps=1000,
        linear_start=1e-4,
        linear_end=2e-2,
        cosine_s=8e-3,
    ):
        if given_betas is not None:
            betas = given_betas
        else:
            betas = make_beta_schedule(
                beta_schedule,
                timesteps,
                linear_start=linear_start,
                linear_end=linear_end,
                cosine_s=cosine_s,
            )
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert (
            alphas_cumprod.shape[0] == self.num_timesteps
        ), "alphas have to be defined for each timestep"

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer("alphas_cumprod_prev", to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1.0 - alphas_cumprod))
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod", to_torch(np.log(1.0 - alphas_cumprod))
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod))
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod - 1))
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (
            1.0 - alphas_cumprod_prev
        ) / (1.0 - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer("posterior_variance", to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer(
            "posterior_log_variance_clipped",
            to_torch(np.log(np.maximum(posterior_variance, 1e-20))),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            to_torch(
                (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)
            ),
        )

        if self.parameterization == "eps":
            lvlb_weights = self.betas**2 / (
                2
                * self.posterior_variance
                * to_torch(alphas)
                * (1 - self.alphas_cumprod)
            )
        elif self.parameterization == "x0":
            lvlb_weights = (
                0.5
                * np.sqrt(torch.Tensor(alphas_cumprod))
                / (2.0 * 1 - torch.Tensor(alphas_cumprod))
            )
        else:
            raise NotImplementedError("mu not supported")
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer("lvlb_weights", lvlb_weights, persistent=False)
        assert not torch.isnan(self.lvlb_weights).all()

    def resume_first_stage(self, config):
        if "ckpt_dir" in config and config["ckpt_dir"] is not None:
            self.first_stage_model.resume(config["ckpt_dir"])
        else:
            with rank_zero_only_context():
                logger.info(f"No checkpoint for first stage model")

        self.first_stage_model.eval()
        self.first_stage_model.train = disabled_train
        common_utils.requires_grad(self.first_stage_model, False)

    def resume_second_stage(
        self, ckpt_dir, ignore_keys=["first_stage_model"], strict=False
    ):
        state_dict = {}
        for level in self.levels:
            level = f"{level}_start" if level == self.start_level else level
            ckpt_path = os.path.join(ckpt_dir, level, f"last.ckpt")
            if os.path.exists(ckpt_path):
                sd = torch.load(ckpt_path, map_location=lambda storage, loc: storage)[
                    "state_dict"
                ]
                sd = {
                    k: v
                    for k, v in sd.items()
                    if all([k.find(ig) == -1 for ig in list(ignore_keys)])
                }
                state_dict.update(sd)
        
                if "lr_schedulers" in sd:
                    self.learning_rate = sd["lr_schedulers"][0]["_last_lr"][0]
        
        if len(state_dict) == 0:
            with rank_zero_only_context():
                logger.info(f"No checkpoint for second stage model")
            return

        missing, unexpected = self.load_state_dict(state_dict, strict=strict)
        with rank_zero_only_context():
            logger.info(
                f"Resume Second Stage Model of levels: {self.levels} with {len(missing)} missing and {len(unexpected)} unexpected keys"
            )
            if len(missing) > 0:
                logger.info(f"Missing Keys: {missing}")
            if len(unexpected) > 0:
                logger.info(f"Unexpected Keys: {unexpected}")

    @contextmanager
    def ema_scope(self):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).
        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
            * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    @torch.no_grad()
    def p_sample(
        self,
        inputs,
        t,
        clip_denoised=False,
        repeat_noise=False,
        return_x0=False,
        temperature=1.0,
        noise_dropout=0.0,
        score_corrector=None,
        corrector_kwargs=None,
        **kwargs,
    ):
        b, *_, device = *inputs["x"].shape, inputs["x"].device
        outputs = self.p_mean_variance(
            inputs=inputs,
            t=t,
            clip_denoised=clip_denoised,
            return_x0=return_x0,
            score_corrector=score_corrector,
            corrector_kwargs=corrector_kwargs,
            **kwargs,
        )

        if return_x0:
            model_mean, _, model_log_variance, x0 = outputs
        else:
            model_mean, _, model_log_variance = outputs

        noise = noise_like(inputs["x"].shape, device, repeat_noise) * temperature
        if noise_dropout > 0.0:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(
            b, *((1,) * (len(inputs["x"].shape) - 1))
        )

        if return_x0:
            return (
                model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise,
                x0,
            )
        else:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(
        self,
        shape,
        inputs={},
        return_intermediates=False,
        verbose=True,
        callback=None,
        timesteps=None,
        img_callback=None,
        log_every_t=None,
        **kwargs,
    ):
        if not log_every_t:
            log_every_t = self.log_every_t
        device = self.betas.device
        b = shape[0]

        intermediates = [inputs["x"]]
        if timesteps is None:
            timesteps = self.num_timesteps

        if "start_T" in inputs:
            timesteps = min(timesteps, inputs["start_T"])

        iterator = (
            tqdm(reversed(range(0, timesteps)), desc="Sampling t", total=timesteps)
            if verbose
            else reversed(range(0, timesteps))
        )

        if "mask" in inputs:
            assert "x0" in inputs
            assert (
                inputs["x0"].shape[2:3] == inputs["mask"].shape[2:3]
            )  # spatial size has to match

        for i in iterator:
            ts = torch.full((b,), i, device=device, dtype=torch.long)

            if self.shorten_cond_schedule:
                assert self.model.cond_key != "hybrid"
                tc = self.cond_ids[ts].to(inputs["c"].device)
                inputs["c"] = self.q_sample(
                    x_start=inputs["c"], t=tc, noise=torch.randn_like(inputs["c"])
                )

            inputs["x"] = self.p_sample(
                inputs, ts, clip_denoised=self.clip_denoised, **kwargs
            )

            if "mask" in inputs:
                img_orig = self.q_sample(inputs["x0"], ts)
                inputs["x"] = (
                    img_orig * inputs["mask"] + (1.0 - inputs["mask"]) * inputs["x"]
                )

            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(inputs["x"])
            if callback:
                callback(i)
            if img_callback:
                img_callback(inputs["x"], i)

        if return_intermediates:
            return inputs["x"], intermediates
        return inputs["x"]

    @torch.no_grad()
    def ddpm_sample(
        self,
        inputs,
        batch_size=16,
        return_intermediates=False,
        verbose=True,
        timesteps=None,
        shape=None,
        **kwargs,
    ):
        if inputs["c"] is not None:
            cond = inputs["c"]
            if isinstance(cond, dict):
                inputs["c"] = {
                    key: (
                        cond[key][:batch_size]
                        if not isinstance(cond[key], list)
                        else list(map(lambda x: x[:batch_size], cond[key]))
                    )
                    for key in cond
                }
            else:
                inputs["c"] = (
                    [c[:batch_size] for c in cond]
                    if isinstance(cond, list)
                    else cond[:batch_size]
                )
        return self.p_sample_loop(
            shape,
            inputs,
            return_intermediates=return_intermediates,
            verbose=verbose,
            timesteps=timesteps,
            **kwargs,
        )

    def q_sample(self, x_start, t, noise=None):
        noise = common_utils.default(noise, lambda: torch.randn_like(x_start))
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )

    def get_loss(self, pred, target, mean=True):
        if self.loss_type == "l1":
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == "l2":
            if mean:
                loss = torch.nn.functional.mse_loss(pred, target)
            else:
                loss = torch.nn.functional.mse_loss(pred, target, reduction="none")
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss

    def p_losses(self, inputs, t, noise=None, **kwargs):
        x_start = inputs["x"]
        noise = common_utils.default(noise, lambda: torch.randn_like(x_start))
        inputs["x"] = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_output = self.apply_model(inputs, t, **kwargs)

        loss_dict = {}
        prefix = "train" if self.training else "val"

        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        else:
            raise NotImplementedError()

        mean_dims = [1, 2, 3, 4]

        loss_simple = self.get_loss(model_output, target, mean=False).mean(mean_dims)
        loss_dict.update({f"{prefix}/loss_simple": loss_simple.mean()})

        logvar_t = self.logvar.to(self.device)[t]
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        # loss = loss_simple / torch.exp(self.logvar) + self.logvar
        if self.learn_logvar:
            loss_dict.update({f"{prefix}/loss_gamma": loss.mean()})
            loss_dict.update({"logvar": self.logvar.data.mean()})

        loss = self.l_simple_weight * loss.mean()

        loss_vlb = self.get_loss(model_output, target, mean=False).mean(dim=mean_dims)
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss_dict.update({f"{prefix}/loss_vlb": loss_vlb})
        loss += self.original_elbo_weight * loss_vlb
        loss_dict.update({f"{prefix}/loss": loss})

        return loss, loss_dict

    def apply_model(self, inputs, t, **kwargs):
        if isinstance(inputs["c"], dict):
            cond = {}
            for key, value in inputs["c"].items():
                cond[key] = [value] if not isinstance(value, list) else value
        elif inputs["c"] is None:
            cond = {}
        else:
            raise NotImplementedError

        x_recon = self.model(t=t, **cond, **inputs)

        if isinstance(x_recon, tuple):
            return x_recon[0]
        else:
            return x_recon

    def p_mean_variance(
        self,
        inputs,
        t,
        clip_denoised: bool,
        return_x0=False,
        score_corrector=None,
        corrector_kwargs=None,
        **kwargs,
    ):
        t_in = t
        model_out = self.apply_model(inputs, t_in, **kwargs)

        if score_corrector is not None:
            assert self.parameterization == "eps"
            model_out = score_corrector.modify_score(
                self, model_out, inputs["x"], t, inputs["c"], **corrector_kwargs
            )

        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(inputs["x"], t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        else:
            raise NotImplementedError()

        if clip_denoised:
            x_recon.clamp_(-1.0, 1.0)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=inputs["x"], t=t
        )

        if return_x0:
            return model_mean, posterior_variance, posterior_log_variance, x_recon
        else:
            return model_mean, posterior_variance, posterior_log_variance

    def losses(self, inputs, *args, **kwargs):
        if "t" not in inputs:
            t = torch.randint(
                0, self.num_timesteps, (inputs["x"].shape[0],), device=self.device
            ).long()
        else:
            t = inputs["t"]

        return self.p_losses(inputs, t, *args, **kwargs)

    @torch.no_grad()
    def progressive_denoising(
        self,
        shape,
        inputs=None,
        verbose=False,
        callback=None,
        img_callback=None,
        x0=None,
        temperature=1.0,
        noise_dropout=0.0,
        score_corrector=None,
        corrector_kwargs=None,
        batch_size=None,
        log_every_t=None,
    ):
        if not log_every_t:
            log_every_t = self.log_every_t
        timesteps = self.num_timesteps
        if batch_size is not None:
            b = batch_size if batch_size is not None else shape[0]
            shape = [batch_size] + list(shape)
        else:
            b = batch_size = shape[0]

        intermediates = []
        if "c" in inputs:
            cond = inputs["c"]
            if isinstance(cond, dict):
                inputs["c"] = {
                    key: (
                        cond[key][:batch_size]
                        if not isinstance(cond[key], list)
                        else list(map(lambda x: x[:batch_size], cond[key]))
                    )
                    for key in cond
                }
            else:
                inputs["c"] = (
                    [c[:batch_size] for c in cond]
                    if isinstance(cond, list)
                    else cond[:batch_size]
                )

        if "start_T" in inputs:
            timesteps = min(timesteps, inputs["start_T"])
        iterator = (
            tqdm(
                reversed(range(0, timesteps)),
                desc="Progressive Generation",
                total=timesteps,
            )
            if verbose
            else reversed(range(0, timesteps))
        )

        if type(temperature) == float:
            temperature = [temperature] * timesteps

        for i in iterator:
            ts = torch.full((b,), i, device=self.device, dtype=torch.long)

            inputs["x"], x0_partial = self.p_sample(
                inputs,
                cond,
                ts,
                clip_denoised=self.clip_denoised,
                return_x0=True,
                temperature=temperature[i],
                noise_dropout=noise_dropout,
                score_corrector=score_corrector,
                corrector_kwargs=corrector_kwargs,
            )
            if "mask" in inputs:
                assert x0 is not None
                img_orig = self.q_sample(x0, ts)
                inputs["x"] = (
                    img_orig * inputs["mask"] + (1.0 - inputs["mask"]) * inputs["x"]
                )

            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(x0_partial)
            if callback:
                callback(i)
            if img_callback:
                img_callback(inputs["x"], i)
        return inputs["x"], intermediates

    def configure_optimizers(self):
        names = ["Unet"]
        params = list(self.model.parameters())

        if self.learn_logvar:
            params.append(self.logvar)
            names.append("Logvar")

        with rank_zero_only_context():
            logger.info(f"Optimizing parameters: {names}")

        opt = torch.optim.AdamW(params, lr=self.learning_rate)

        if self.use_scheduler:
            with rank_zero_only_context():
                logger.info("Setting up scheduler...")
            scheduler = [
                {
                    "scheduler": StepLR(opt, step_size=100, gamma=0.95),
                    "interval": "epoch",
                    "frequency": 1,
                }
            ]
            return [opt], scheduler
        return opt

    @torch.no_grad()
    def sample(self, inputs, batch_size, ddim, shape=None, **kwargs):
        if ddim:
            z, intermediates = self.ddim_sample(
                inputs, batch_size, shape, verbose=False, **kwargs
            )
        else:
            z, intermediates = self.ddpm_sample(
                inputs=inputs,
                batch_size=batch_size,
                shape=shape,
                return_intermediates=True,
                **kwargs,
            )
        return z, intermediates, inputs

    @torch.no_grad()
    def get_inputs(self, batch):
        level = self.levels[0]
        geometry = batch["latent"].to(self.device)  # [B, C, G, G, G]

        x = self.first_stage_model.encode(geometry, level=level)
        x = self.first_stage_model.normalize(x, level=level).detach()
        x = x.to(memory_format=torch.contiguous_format).float()

        if level == self.start_level:
            inputs = {"x": x, "c": None, "level": level}
        else:
            inputs = {"x": x[:, 1:], "c": {"c_concat": [x[:, :1]]}, "level": level}

        stats = {
            "L_var": x[:, :1].var().item(),
            "L_mean": x[:, :1].mean().item(),
            "H_var": x[:, 1:].var().item(),
            "H_mean": x[:, 1:].mean().item(),
        }

        return inputs, stats

    def get_latent(self, inputs):
        if inputs["level"] == self.start_level:
            x = inputs["x"]
        else:
            x = torch.cat((inputs["c"]["c_concat"][0], inputs["x"]), dim=1)

        return x

    def shared_step(self, batch):
        inputs, stats_dict = self.get_inputs(batch)
        loss, loss_dict = self.losses(inputs)
        return loss, loss_dict, stats_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict, stats_dict = self.shared_step(batch)

        return {"loss": loss, "loss_dict": loss_dict, "stats_dict": stats_dict}

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.log_dict(
            outputs["loss_dict"],
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )

        self.log_dict(
            outputs["stats_dict"],
            prog_bar=False,
            logger=True,
            on_step=True,
            sync_dist=True,
        )

        if self.use_scheduler:
            lr = self.optimizers().param_groups[0]["lr"]
            self.log(
                "lr_abs",
                lr,
                prog_bar=True,
                logger=True,
                on_step=True,
                on_epoch=True,
                sync_dist=True,
            )

        if self.use_ema and (self.global_step % self.ema_update_batch == 0):
            self.model_ema(self.model)

    @torch.no_grad()
    def latent_to_mesh(self, latent, level):
        latent = self.first_stage_model.denormalize(latent, level=level)
        volume = self.first_stage_model.decode(latent, level=level)[0, 0]  # [G, G, G]

        mode, vxl_size_i, _ = common_utils.parse_level(level)
        threshold = common_utils.ada_threshold(vxl_size_i, mode)
        threshold = min(self.first_stage_model.truncation * 0.8, threshold)
        mesh = mesh_utils.volume_to_mesh(volume, threshold=threshold)

        return mesh

    @torch.no_grad()
    def validation_step(
        self,
        batch,
        batch_idx,
        ddim_steps=200,
        ddim_eta=1,
        out_dir: Optional[str] = None,
    ):
        _, loss_dict_no_ema, _ = self.shared_step(batch)
        with self.ema_scope():
            _, loss_dict_ema, _ = self.shared_step(batch)
            loss_dict_ema = {key + "_ema": loss_dict_ema[key] for key in loss_dict_ema}
        self.log_dict(
            {**loss_dict_no_ema, **loss_dict_ema},
            prog_bar=False,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        if batch_idx == 0:
            if out_dir is None:
                out_dir = os.path.join(self.log_dir, "val")
                os.makedirs(out_dir, exist_ok=True)

            use_ddim = ddim_steps is not None
            B = 1
            level = self.levels[0]
            batch = {k: v[:B] for k, v in batch.items()}
            shape = (B, self.channels[level], *self.chunk_shape[level])

            log = {"mesh": {}}
            inputs = self.get_inputs(batch)[0]
            latent = self.get_latent(inputs)
            mesh = self.latent_to_mesh(latent, level)
            log["mesh"].update({"gt": {f"{self.global_step}_{self.levels[0]}": mesh}})

            with self.ema_scope():
                inputs.update({"x": torch.randn(shape, device=self.device)})
                _, _, inputs = self.sample(
                    inputs=inputs,
                    batch_size=B,
                    ddim=use_ddim,
                    shape=shape,
                    ddim_steps=ddim_steps,
                    eta=ddim_eta,
                )
                latent = self.get_latent(inputs)
                mesh = self.latent_to_mesh(latent, level)
                log["mesh"].update({"samples": {f"{self.global_step}_{level}": mesh}})

            for k1, v1 in log["mesh"].items():
                os.makedirs(os.path.join(out_dir, k1), exist_ok=True)
                for k2, mesh in v1.items():
                    mesh.export(os.path.join(out_dir, k1, f"{k2}.ply"), file_type="ply")

    @torch.no_grad()
    def generation(
        self,
        start_idx: int = 0,
        n_sample: int = 1,
        scene_shape: Tuple = (512, 256, 512),
        overlap: Union[float, int] = 0.5,
        with_chunk: bool = False,
    ):
        with self.ema_scope():
            channels = {
                level: self.channels[level] + self.concat_channels[level]
                for level in self.levels
            }
            voxel_sizes = [
                str(common_utils.parse_level(level)[1]) for level in self.levels
            ]

            for sample_idx in tqdm(
                range(start_idx, start_idx + n_sample),
                desc=f"Generating {n_sample} samples from {start_idx}",
            ):
                seed_everything(sample_idx)

                out_dir = os.path.join(
                    self.log_dir,
                    f"gen_o_{overlap}_v_{str(voxel_sizes[-1])}_"
                    + "_".join(map(str, scene_shape)),
                    f"sample_{sample_idx}",
                )
                os.makedirs(out_dir, exist_ok=True)
                logger.info(f"Generation output directory: {out_dir}")

                if os.path.exists(out_dir) and len(
                    glob.glob(os.path.join(out_dir, "*.ply"))
                ) == len(self.levels):
                    logger.info(
                        f"Generation output directory already exists: {out_dir}"
                    )
                    continue

                scene = Scene(
                    self.levels,
                    channels=channels,
                    scene_shape=scene_shape,
                    device=self.device,
                )

                fusion(
                    self, scene, out_dir=out_dir, overlap=overlap, with_chunk=with_chunk
                )

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        kwargs = self.task_hyparam

        if kwargs["name"] == "generation" and batch_idx == 0:
            self.generation(
                start_idx=kwargs.get("start_idx", 0),
                n_sample=kwargs.get("n_sample", 1),
                scene_shape=kwargs.get("scene_shape", (512, 256, 512)),
                overlap=kwargs.get("overlap", 0.5),
            )

    # @torch.no_grad()
    # def log_metrics(self, dataloader, out_dir="./", max_sample=50, batch_size=5):
    # level = self.levels[0]

    # if inputs["level"] == level:
    #     num_sample = 0
    #     gt_mesh_dir = os.path.join(out_dir, level, 'gt', 'mesh')
    #     os.makedirs(gt_mesh_dir, exist_ok=True)
    #     for batch in dataloader:
    #         inputs = self.get_inputs(batch)[0]
    #         latent = self.get_latent(inputs)

    #         for i in range(len(latent)):
    #             if num_sample < max_sample:
    #                 gt_mesh = self.latent_to_mesh(latent[[i]], level)
    #                 gt_mesh.export(os.path.join(gt_mesh_dir, f'{num_sample}.ply'), file_type='ply')

    #                 num_sample += 1
    #             else:
    #                 break

    #     gen_mesh_dir = os.path.join(out_dir, level, 'generation', 'mesh')
    #     os.makedirs(gen_mesh_dir, exist_ok=True)
    #     with self.ema_scope():
    #         curr_idx = 0
    #         for i in range(0, num_sample, batch_size):
    #             nbatch = min(i + batch_size, num_sample) - i
    #             shape = (nbatch, self.channels[level], *self.chunk_shape[level])
    #             inputs = {'x': torch.randn(shape, device=self.device), 'level': level, 'c': None}
    #             _, _, inputs = self.sample(
    #                 inputs=inputs, shape=shape, batch_size=nbatch, ddim=True)
    #             latent = self.get_latent(inputs)
    #             for j in range(nbatch):
    #                 mesh = self.latent_to_mesh(latent[[j]], level)
    #                 mesh.export(os.path.join(gen_mesh_dir, f'{curr_idx}.ply'), file_type='ply')
    #                 curr_idx += 1

    #     os.makedirs(os.path.join(out_dir, level, 'gt', 'image'), exist_ok=True)
    #     os.makedirs(os.path.join(out_dir, level, 'generation', 'image'), exist_ok=True)

    # # render mesh
    # mesh_paths = glob.glob(os.path.join(gt_mesh_dir, '*.ply')) + glob.glob(os.path.join(gen_mesh_dir, '*.ply'))
    # for mesh_path in mesh_paths:
    #     mesh = trimesh.load(mesh_path)

    #     bbox_min = np.min(mesh.vertices, axis=0)
    #     bbox_max = np.max(mesh.vertices, axis=0)

    #     poses = cam_center_horiz_rot(num_views=20, bboxes=np.stack([bbox_min, bbox_max]))

    #     for j, pose in enumerate(poses):
    #         image, _ = render_mesh(mesh, pose)

    #         out_path = mesh_path.replace('mesh', 'image').replace('.ply', f'_{j:02d}.png')
    #         torchvision.utils.save_image(
    #             torch.from_numpy(image.copy()).permute(2, 0, 1), out_path)

    # render mesh in gt_mesh_dir and gen_mesh_dir to gt_img_dir and gen_img_dir
    # fid = compute_fid_2d(os.path.join(out_dir, level, 'generation', 'image'), os.path.join(out_dir, level, 'gt', 'image'))
    # self.log('fid', fid, prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True)

    # else: # reconstuction loss
    #     pass
