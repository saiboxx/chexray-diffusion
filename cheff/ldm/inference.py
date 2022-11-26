"""Functions and classes for loading and handling models conveniently."""
import contextlib
from typing import Union, Dict, Optional

import torch
from torch import Tensor

from cheff.ldm.models.autoencoder import AutoencoderKL
from cheff.ldm.models.diffusion.ddpm import LatentDiffusion
from cheff.ldm.models.diffusion.ddim import DDIMSampler


class CheffAEModel:
    def __init__(
            self,
            model_path: str,
            device: Union[str, int, torch.device] = 'cuda'
    ) -> None:
        self.device = device

        with contextlib.redirect_stdout(None):
            self.model = AutoencoderKL(
                embed_dim=3,
                ckpt_path=model_path,
                ddconfig={
                    'double_z': True,
                    'z_channels': 3,
                    'resolution': 256,
                    'in_channels': 3,
                    'out_ch': 3,
                    'ch': 128,
                    'ch_mult': (1, 2, 4),
                    'num_res_blocks': 2,
                    'attn_resolutions': [],
                    'dropout': 0.0
                },
                lossconfig={'target': 'torch.nn.Identity'}
            )
        self.model = self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def encode(self, x: Tensor) -> Tensor:
        return self.model.encode(x).mode()

    @torch.no_grad()
    def decode(self, z: Tensor) -> Tensor:
        return self.model.decode(z)


class CheffLDM:
    def __init__(
            self,
            model_path: str,
            ae_path: Optional[str] = None,
            device: Union[str, int, torch.device] = 'cuda'
    ) -> None:
        self.device = device
        with contextlib.redirect_stdout(None):
            self.model = self._init_checkpoint(model_path, ae_path)

        self.model = self.model.to(self.device)
        self.model.model = self.model.model.to(self.device)
        self.model.eval()

    def sample(
            self,
            batch_size: int = 1,
            sampling_steps: int = 100,
            eta: float = 1.0,
            decode: bool = True,
    ) -> Tensor:
        shape = [
            self.model.model.diffusion_model.in_channels,
            self.model.model.diffusion_model.image_size,
            self.model.model.diffusion_model.image_size
        ]

        ddim = DDIMSampler(self.model)
        samples, _ = ddim.sample(
            sampling_steps, batch_size=batch_size, shape=shape, eta=eta, verbose=False
        )

        if decode:
            samples = self.model.decode_first_stage(samples)

        return samples

    def _init_checkpoint(
            self, model_path: str, ae_path: Optional[str] = None
    ) -> LatentDiffusion:
        config_dict = CheffLDM._get_config_dict(ae_path)
        model = LatentDiffusion(**config_dict)

        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict['state_dict'], strict=False)
        return model

    @staticmethod
    def _get_config_dict(ae_path: Optional[str] = None) -> Dict:
        return {
            'linear_start': 0.0015,
            'linear_end': 0.0195,
            'num_timesteps_cond': 1,
            'log_every_t': 200,
            'timesteps': 1000,
            'first_stage_key': 'image',
            'image_size': 64,
            'channels': 3,
            'monitor': 'val/loss_simple_ema',
            'unet_config': CheffLDM._get_unet_config_dict(),
            'first_stage_config': CheffLDM._get_first_stage_config_dict(ae_path),
            'cond_stage_config': '__is_unconditional__'
        }

    @staticmethod
    def _get_unet_config_dict() -> Dict:
        return {
            'target': 'cheff.ldm.modules.diffusionmodules.openaimodel.UNetModel',
            'params': {
                'image_size': 64,
                'in_channels': 3,
                'out_channels': 3,
                'model_channels': 224,
                'attention_resolutions': [8, 4, 2],
                'num_res_blocks': 2,
                'channel_mult': [1, 2, 3, 4],
                'num_head_channels': 32,
            }
        }

    @staticmethod
    def _get_first_stage_config_dict(ae_path: Optional[str] = None) -> Dict:
        return {
            'target': 'cheff.ldm.models.autoencoder.AutoencoderKL',
            'params': {
                'embed_dim': 3,
                'ckpt_path': ae_path,
                'ddconfig': {
                    'double_z': True,
                    'z_channels': 3,
                    'resolution': 256,
                    'in_channels': 3,
                    'out_ch': 3,
                    'ch': 128,
                    'ch_mult': (1, 2, 4),
                    'num_res_blocks': 2,
                    'attn_resolutions': [],
                    'dropout': 0.0
                },
                'lossconfig': {'target': 'torch.nn.Identity'}
            }
        }
