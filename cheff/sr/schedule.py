"""Classes und functions for variance schedules."""

from abc import ABC, abstractmethod
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import Tensor


class BaseSchedule(ABC):
    """Base class for deriving schedules."""

    def __init__(
        self, timesteps: int, device: Optional[torch.device] = None, *args, **kwargs
    ) -> None:
        """Initialize BaseSchedule."""
        self.timesteps = timesteps
        if device is None:
            self.device = torch.device('cpu')
        else:
            self.device = device

        self.betas = self._get_betas(timesteps).to(device)
        self.alphas = 1.0 - self.betas

        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

        self.post_var = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

        self.post_log_var_clipped = torch.log(
            torch.maximum(self.post_var, torch.tensor(1e-20))
        )
        self.post_mean_coef1 = (
            self.betas
            * torch.sqrt(self.alphas_cumprod_prev)
            / (1.0 - self.alphas_cumprod)
        )
        self.post_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * torch.sqrt(self.alphas)
            / (1.0 - self.alphas_cumprod)
        )

    @abstractmethod
    def _get_betas(self, timesteps: int) -> Tensor:
        """Get betas."""
        pass


class LinearSchedule(BaseSchedule):
    """Linear variance schedule."""

    def __init__(
        self,
        timesteps: int,
        device: Optional[torch.device] = None,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        *args,
        **kwargs
    ) -> None:
        """Initialize linear beta schedule."""
        self.beta_start = beta_start
        self.beta_end = beta_end
        super().__init__(timesteps, device, *args, **kwargs)

    def _get_betas(self, timesteps: int) -> Tensor:
        """Get betas."""
        return torch.linspace(self.beta_start, self.beta_end, timesteps)


class CosineSchedule(BaseSchedule):
    """Cosine variance schedule."""

    def __init__(
        self,
        timesteps: int,
        device: Optional[torch.device] = None,
        s: float = 0.008,
        *args,
        **kwargs
    ) -> None:
        """Initialize cosine beta schedule."""
        self.s = s
        super().__init__(timesteps, device, *args, **kwargs)

    def _get_betas(self, timesteps: int) -> Tensor:
        """Get betas."""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = (
            torch.cos(((x / timesteps) + self.s) / (1 + self.s) * torch.pi * 0.5) ** 2
        )
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)


class QuadraticSchedule(BaseSchedule):
    """Quadratic variance schedule."""

    def __init__(
        self,
        timesteps: int,
        device: Optional[torch.device] = None,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        *args,
        **kwargs
    ) -> None:
        """Initialize quadratic beta schedule."""
        self.beta_start = beta_start
        self.beta_end = beta_end
        super().__init__(timesteps, device, *args, **kwargs)

    def _get_betas(self, timesteps: int) -> Tensor:
        """Get betas."""
        return (
            torch.linspace(self.beta_start**0.5, self.beta_end**0.5, timesteps) ** 2
        )


class SigmoidSchedule(BaseSchedule):
    """Sigmoid variance schedule."""

    def __init__(
        self,
        timesteps: int,
        device: Optional[torch.device] = None,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        *args,
        **kwargs
    ) -> None:
        """Initialize sigmoid beta schedule."""
        self.beta_start = beta_start
        self.beta_end = beta_end
        super().__init__(timesteps, device, *args, **kwargs)

    def _get_betas(self, timesteps: int) -> Tensor:
        """Get betas."""
        betas = torch.linspace(-6, 6, timesteps)
        return (
            torch.sigmoid(betas) * (self.beta_end - self.beta_start) + self.beta_start
        )


class ScheduleFactory:
    """Factory wrapper for variance schedules."""

    @staticmethod
    def get_schedule(name: str, timesteps: int, *args, **kwargs) -> BaseSchedule:
        """Initialize a scheduler by name."""
        cls: Any
        if name == 'linear':
            cls = LinearSchedule
        elif name == 'cosine':
            cls = CosineSchedule
        elif name == 'quadratic':
            cls = QuadraticSchedule
        elif name == 'sigmoid':
            cls = SigmoidSchedule
        else:
            raise ValueError(
                'There is no matching schedule for name "{}".'.format(name)
            )

        return cls(timesteps, *args, **kwargs)
