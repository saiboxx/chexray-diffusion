"""Classes and functions for sr processes."""
from typing import Optional, Tuple

import torch
from torch import Tensor, nn
from tqdm import tqdm

from cheff.sr.schedule import BaseSchedule


class Diffusor:
    """Class for modelling the sr process."""

    def __init__(
        self,
        model: nn.Module,
        schedule: BaseSchedule,
        device: Optional[torch.device] = None,
        clip_denoised: bool = True,
    ) -> None:
        """Initialize Diffusor."""
        self.model = model
        self.schedule = schedule
        self.clip_denoised = clip_denoised

        if device is None:
            self.device = schedule.device
        else:
            self.device = device

    @staticmethod
    def extract_vals(a: Tensor, t: Tensor, x_shape: Tuple) -> Tensor:
        """Extract timestep values from tensor and reshape to target dims."""
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def q_sample(
        self, x_start: Tensor, t: Tensor, noise: Optional[Tensor] = None
    ) -> Tensor:
        """
        Sample from forward process q.

        Given an initial `x_start` and a timestep `t, return perturbed images `x_t`.
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = Diffusor.extract_vals(
            self.schedule.sqrt_alphas_cumprod, t, x_start.shape
        )
        sqrt_one_minus_alphas_cumprod_t = Diffusor.extract_vals(
            self.schedule.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def predict_start_from_noise(self, x_t: Tensor, t: Tensor, noise: Tensor) -> Tensor:
        """Subtract noise from x_t over variance schedule."""
        sqrt_recip_alphas_cumprod_t = Diffusor.extract_vals(
            self.schedule.sqrt_recip_alphas_cumprod, t, x_t.shape
        )
        sqrt_recipm1_alphas_cumprod_t = Diffusor.extract_vals(
            self.schedule.sqrt_recipm1_alphas_cumprod, t, x_t.shape
        )

        x_rec = (
            sqrt_recip_alphas_cumprod_t * x_t - sqrt_recipm1_alphas_cumprod_t * noise
        )

        return x_rec

    def q_posterior(
        self, x_start: Tensor, x_t: Tensor, t: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Compute posterior q."""
        posterior_mean_coef1 = Diffusor.extract_vals(
            self.schedule.post_mean_coef1, t, x_t.shape
        )
        posterior_mean_coef2 = Diffusor.extract_vals(
            self.schedule.post_mean_coef2, t, x_t.shape
        )
        post_var = Diffusor.extract_vals(self.schedule.post_var, t, x_t.shape)
        posterior_log_var_clipped = Diffusor.extract_vals(
            self.schedule.post_log_var_clipped, t, x_t.shape
        )

        posterior_mean = posterior_mean_coef1 * x_start + posterior_mean_coef2 * x_t
        return posterior_mean, post_var, posterior_log_var_clipped

    def p_mean_variance(self, x: Tensor, t: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Compute mean and variance for reverse process."""
        noise_pred = self.model(x, t)
        x_recon = self.predict_start_from_noise(x, t=t, noise=noise_pred)

        if self.clip_denoised:
            x_recon.clamp_(-1.0, 1.0)
        return self.q_posterior(x_start=x_recon, x_t=x, t=t)

    @torch.no_grad()
    def p_sample(self, x: Tensor, t: Tensor) -> Tensor:
        """Sample from reverse process."""
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t)
        noise = torch.randn_like(x)

        if t[0] == 0:
            return model_mean
        else:
            return model_mean + (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape: Tuple) -> Tensor:
        """Initiate generation process."""
        batch_size = shape[0]
        img = torch.randn(shape, device=self.device)

        pbar = tqdm(
            iterable=reversed(range(0, self.schedule.timesteps)),
            desc='sampling loop time step',
            total=self.schedule.timesteps,
            leave=False,
        )
        for i in pbar:
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            img = self.p_sample(img, t)
        return img

    @torch.no_grad()
    def p_sample_loop_with_steps(self, shape: Tuple, log_every_t: int) -> Tensor:
        """Initiate generation process and return intermediate steps."""
        batch_size = shape[0]
        img = torch.randn(shape, device=self.device)

        result = [img]

        pbar = tqdm(
            iterable=reversed(range(0, self.schedule.timesteps)),
            desc='sampling loop time step',
            total=self.schedule.timesteps,
            leave=False,
        )
        for i in pbar:
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            img = self.p_sample(img, t)

            if i % log_every_t == 0 or i == self.schedule.timesteps - 1:
                result.append(img)

        return torch.stack(result)


class SR3Diffusor(Diffusor):
    """Class for modelling the sr process in SR3."""

    def p_mean_variance(  # type: ignore
        self, x: Tensor, sr: Tensor, t: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Compute mean and variance for reverse process."""
        x_in = torch.cat([x, sr], dim=1)
        noise_pred = self.model(x_in, t)
        x_recon = self.predict_start_from_noise(x, t=t, noise=noise_pred)

        if self.clip_denoised:
            x_recon.clamp_(-1.0, 1.0)
        return self.q_posterior(x_start=x_recon, x_t=x, t=t)

    @torch.no_grad()
    def p_sample(self, x: Tensor, sr: Tensor, t: Tensor) -> Tensor:
        """Sample from reverse process."""
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, sr=sr, t=t)
        noise = torch.randn_like(x)

        if t[0] == 0:
            return model_mean
        else:
            return model_mean + (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, sr: Tensor) -> Tensor:
        """Initiate generation process."""
        batch_size = sr.shape[0]
        img = torch.randn(sr.shape, device=self.device)

        pbar = tqdm(
            iterable=reversed(range(0, self.schedule.timesteps)),
            desc='sampling loop time step',
            total=self.schedule.timesteps,
            leave=False,
        )
        for i in pbar:
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            img = self.p_sample(img, sr, t)
        return img

    @torch.no_grad()
    def p_sample_loop_with_steps(self, sr: Tensor, log_every_t: int) -> Tensor:
        """Initiate generation process and return intermediate steps."""
        batch_size = sr.shape[0]
        img = torch.randn(sr.shape, device=self.device)

        result = [img]

        pbar = tqdm(
            iterable=reversed(range(0, self.schedule.timesteps)),
            desc='sampling loop time step',
            total=self.schedule.timesteps,
            leave=False,
        )
        for i in pbar:
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            img = self.p_sample(img, sr, t)

            if i % log_every_t == 0 or i == self.schedule.timesteps - 1:
                result.append(img)

        return torch.stack(result)


class DDIMDiffusor(Diffusor):
    """Class for modelling the sr process with DDIM."""

    def __init__(
        self,
        model: nn.Module,
        schedule: BaseSchedule,
        sampling_steps: Optional[int] = 100,
        eta: Optional[float] = 0.0,
        device: Optional[torch.device] = None,
        clip_denoised: bool = True,
    ) -> None:
        """Initialize DDIM sampler."""
        super().__init__(model, schedule, device, clip_denoised)
        self.sampling_steps = sampling_steps
        self.eta = eta

        self.ddim_timesteps = torch.arange(
            start=0,
            end=schedule.timesteps,
            step=schedule.timesteps // self.sampling_steps,
        )

        # Select alphas for DDIM variance schedule
        self.ddim_alpha_cumprod = self.schedule.alphas_cumprod[self.ddim_timesteps]
        self.ddim_alphas_cumprod_prev = torch.cat(
            [
                self.schedule.alphas_cumprod[0].unsqueeze(0),
                self.schedule.alphas_cumprod[self.ddim_timesteps[:-1]],
            ]
        )
        self.sigmas = self.eta * torch.sqrt(
            (1 - self.ddim_alphas_cumprod_prev)
            / (1 - self.ddim_alpha_cumprod)
            * (1 - self.ddim_alpha_cumprod / self.ddim_alphas_cumprod_prev)
        )

        self.ddim_sqrt_one_minus_alphas_cumprod = torch.sqrt(
            1.0 - self.ddim_alpha_cumprod
        )

    @torch.no_grad()
    def p_sample(self, x: Tensor, t: Tensor, index: int) -> Tensor:
        """Sample from reverse process."""
        b = x.shape[0]
        a_t = torch.full(
            (b, 1, 1, 1), self.ddim_alpha_cumprod[index], device=self.device
        )
        a_prev = torch.full(
            (b, 1, 1, 1), self.ddim_alphas_cumprod_prev[index], device=self.device
        )
        sigma_t = torch.full((b, 1, 1, 1), self.sigmas[index], device=self.device)
        sqrt_one_minus_at = torch.full(
            (b, 1, 1, 1),
            self.ddim_sqrt_one_minus_alphas_cumprod[index],
            device=self.device,
        )

        e_t = self.model(x, t)
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        dir_xt = (1.0 - a_prev - sigma_t**2).sqrt() * e_t

        noise = sigma_t * torch.randn_like(x)

        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev

    @torch.no_grad()
    def p_sample_loop(self, shape: Tuple) -> Tensor:
        """Initiate generation process."""
        batch_size = shape[0]
        img = torch.randn(shape, device=self.device)

        pbar = tqdm(
            iterable=torch.flip(self.ddim_timesteps, dims=(0,)),
            desc='DDIM sampling loop time step',
            total=len(self.ddim_timesteps),
            leave=False,
        )

        for i, step in enumerate(pbar):
            index = len(self.ddim_timesteps) - i - 1
            t = torch.full((batch_size,), step, device=self.device, dtype=torch.long)
            img = self.p_sample(img, t, index)
        return img

    @torch.no_grad()
    def p_sample_loop_with_steps(self, shape: Tuple, log_every_t: int) -> Tensor:
        """Initiate generation process and return intermediate steps."""
        batch_size = shape[0]
        img = torch.randn(shape, device=self.device)

        result = [img]

        pbar = tqdm(
            iterable=torch.flip(self.ddim_timesteps, dims=(0,)),
            desc='DDIM sampling loop time step',
            total=len(self.ddim_timesteps),
            leave=False,
        )

        for i, step in enumerate(pbar):
            index = len(self.ddim_timesteps) - i - 1
            t = torch.full((batch_size,), step, device=self.device, dtype=torch.long)
            img = self.p_sample(img, t, index)

            if i % log_every_t == 0 or i == self.schedule.timesteps - 1:
                result.append(img)

        return torch.stack(result)


class SR3DDIMDiffusor(DDIMDiffusor):
    """Class for modelling the sr process with DDIM for super resolution."""

    @torch.no_grad()
    def p_sample(self, x: Tensor, sr: Tensor, t: Tensor, index: int) -> Tensor:
        """Sample from reverse process."""
        b = x.shape[0]
        a_t = torch.full(
            (b, 1, 1, 1), self.ddim_alpha_cumprod[index], device=self.device
        )
        a_prev = torch.full(
            (b, 1, 1, 1), self.ddim_alphas_cumprod_prev[index], device=self.device
        )
        sigma_t = torch.full((b, 1, 1, 1), self.sigmas[index], device=self.device)
        sqrt_one_minus_at = torch.full(
            (b, 1, 1, 1),
            self.ddim_sqrt_one_minus_alphas_cumprod[index],
            device=self.device,
        )

        x_in = torch.cat([x, sr], dim=1)
        e_t = self.model(x_in, t)
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        dir_xt = (1.0 - a_prev - sigma_t**2).sqrt() * e_t

        noise = sigma_t * torch.randn_like(x)

        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev

    @torch.no_grad()
    def p_sample_loop(self, sr: Tensor) -> Tensor:
        """Initiate generation process."""
        batch_size = sr.shape[0]
        img = torch.randn(sr.shape, device=self.device)

        pbar = tqdm(
            iterable=torch.flip(self.ddim_timesteps, dims=(0,)),
            desc='DDIM sampling loop time step',
            total=len(self.ddim_timesteps),
            leave=False,
        )

        for i, step in enumerate(pbar):
            index = len(self.ddim_timesteps) - i - 1
            t = torch.full((batch_size,), step, device=self.device, dtype=torch.long)
            img = self.p_sample(img, sr, t, index)
        return img

    @torch.no_grad()
    def p_sample_loop_with_steps(self, sr: Tensor, log_every_t: int) -> Tensor:
        """Initiate generation process and return intermediate steps."""
        batch_size = sr.shape[0]
        img = torch.randn(sr.shape, device=self.device)

        result = [img]

        pbar = tqdm(
            iterable=torch.flip(self.ddim_timesteps, dims=(0,)),
            desc='DDIM sampling loop time step',
            total=len(self.ddim_timesteps),
            leave=False,
        )

        for i, step in enumerate(pbar):
            index = len(self.ddim_timesteps) - i - 1
            t = torch.full((batch_size,), step, device=self.device, dtype=torch.long)
            img = self.p_sample(img, sr, t, index)

            if i % log_every_t == 0 or i == self.schedule.timesteps - 1:
                result.append(img)

        return torch.stack(result)
