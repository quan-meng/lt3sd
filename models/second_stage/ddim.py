"""SAMPLING ONLY."""

import torch
import copy
import numpy as np

from models.modules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like


class DDIMSampler(object):
    def __init__(self, schedule="linear", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.schedule = schedule

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True, device='cpu'):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.num_timesteps, verbose=verbose)
        alphas_cumprod = self.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(device)

        alphas_cumprod = to_torch(alphas_cumprod)
        alphas_cumprod_prev = to_torch(self.alphas_cumprod_prev)
    
        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)
        setattr(self, 'ddim_sigmas', ddim_sigmas)
        setattr(self, 'ddim_alphas', ddim_alphas)
        setattr(self, 'ddim_alphas_prev', torch.from_numpy(ddim_alphas_prev).to(device))
        setattr(self, 'ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - alphas_cumprod_prev) / (1 - alphas_cumprod) * (
                        1 - alphas_cumprod / alphas_cumprod_prev))
        setattr(self, 'ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    @torch.no_grad()
    def ddim_sample(self,
               inputs,
               batch_size,
               shape,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               ddim_steps=200,
               eta=0.,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               log_every_t=20,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):
        device = inputs['x'].device
        self.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=eta, verbose=verbose, device=device)
        # sampling
        size = (batch_size, *shape)
        
        samples, intermediates = self.ddim_sampling(inputs, size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    **kwargs)
        return samples, intermediates

    @torch.no_grad()
    def ddim_sampling(self, inputs, shape, ddim_use_original_steps=False,
                      callback=None, timesteps=None, img_callback=None, log_every_t=20,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None, **kwargs):
        b = shape[0]

        if timesteps is None:
            timesteps = self.num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]
        
        if 'start_T' in inputs:
            if ddim_use_original_steps:
                timesteps = min(timesteps, inputs['start_T'])
            else:
                mask = timesteps <= inputs['start_T']
                timesteps = timesteps[mask]
        
        intermediates = {'x_inter': [inputs['x']], 'pred_x0': [inputs['x']]}
        time_range = reversed(range(0, timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        
        for i, step in enumerate(time_range):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=self.device, dtype=torch.long)
            
            if 'mask' in inputs:
                assert 'x0' in inputs
                img_orig = self.q_sample(inputs['x0'], ts)
                inputs['x'] = img_orig * inputs['mask'] + (1. - inputs['mask']) * inputs['x']

            outs = self.p_sample_ddim(inputs, ts, index=index, use_original_steps=ddim_use_original_steps,
                                      temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs, unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning, **kwargs)
            inputs['x'], pred_x0 = outs
            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(inputs['x'])
                intermediates['pred_x0'].append(pred_x0)

        return inputs['x'], intermediates

    @torch.no_grad()
    def p_sample_ddim(self, inputs, t, index, repeat_noise=False, use_original_steps=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None, **kwargs):
        b, *_, = inputs['x'].shape

        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            e_t = self.apply_model(inputs, t, **kwargs)
        else:
            inputs_copy = copy.deepcopy(inputs)
            inputs_copy['x'] = torch.cat([inputs_copy['x']] * 2)
            t_in = torch.cat([t] * 2)
            inputs_copy['c']['c_concat'] = torch.cat([unconditional_conditioning, inputs_copy['c']['c_concat']])
            e_t_uncond, e_t = self.apply_model(inputs_copy, t_in, **kwargs).chunk(2)
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

        if score_corrector is not None:
            assert self.parameterization == "eps"
            e_t = score_corrector.modify_score(
                self, e_t, inputs['x'], t, inputs['c'], **corrector_kwargs)

        alphas = self.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1, 1), alphas[index], device=self.device)
        a_prev = torch.full((b, 1, 1, 1, 1), alphas_prev[index], device=self.device)
        sigma_t = torch.full((b, 1, 1, 1, 1), sigmas[index], device=self.device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1, 1), sqrt_one_minus_alphas[index],device=self.device)

        # current prediction for x_0
        pred_x0 = (inputs['x'] - sqrt_one_minus_at * e_t) / a_t.sqrt()

        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(inputs['x'].shape, self.device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0
