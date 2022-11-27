import os
from typing import Union, Tuple

from PIL import Image
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize, to_tensor, to_grayscale
from torchvision.utils import save_image

from cheff.sr.model import Unet
from cheff.sr.schedule import ScheduleFactory
from cheff.sr.diffusor import SR3Diffusor, SR3DDIMDiffusor


class CheffSRModel:
    def __init__(
            self,
            model_path: str,
            device: Union[str, int, torch.device] = 'cuda'
    ) -> None:
        self.device = device
        self.model = Unet(
            dim=16,
            channels=2,
            out_dim=1,
            dim_mults=(1, 2, 4, 8, 16, 32, 32, 32),
        )

        state_dict = torch.load(model_path, map_location='cpu')
        self.model.load_state_dict(state_dict['model'])
        self.model.to(self.device)
        self.model.eval()

        self.schedule = ScheduleFactory.get_schedule(
            name='cosine', timesteps=2000, device=self.device)

    def sample_directory(
            self,
            source_dir: str,
            target_dir: str,
            batch_size: int = 1,
            method: str = 'ddim',
            sampling_steps: int = 100,
            eta: float = 0.
    ) -> None:
        ds = DirectoryDataset(source_dir)
        loader = DataLoader(ds, batch_size=batch_size, pin_memory=True)

        os.makedirs(target_dir, exist_ok=True)

        for f_names, imgs in loader:
            imgs_sr = self.sample(imgs, method, sampling_steps, eta)

            for f_name, img_sr in zip(f_names, imgs_sr):
                path = os.path.join(target_dir, f_name)
                save_image(img_sr, path)

    def sample_path(
            self,
            path: str,
            method: str = 'ddim',
            sampling_steps: int = 100,
            eta: float = 0.
    ) -> Tensor:
        img = Image.open(path)
        img = to_tensor(to_grayscale(img)).unsqueeze(0)
        return self.sample(img, method, sampling_steps, eta)

    @torch.no_grad()
    def sample(
            self,
            img: Tensor,
            method: str = 'ddim',
            sampling_steps: int = 100,
            eta: float = 0.
    ) -> Tensor:
        img = img.to(self.device)
        img = img * 2 - 1
        img = resize(img, [1024, 1024], InterpolationMode.BICUBIC)

        if method == 'ddim':
            diffusor = SR3DDIMDiffusor(
                model=self.model,
                schedule=self.schedule,
                sampling_steps=sampling_steps,
                eta=eta
            )
        else:
            diffusor = SR3Diffusor(
                model=self.model,
                schedule=self.schedule,
            )

        img_sr = diffusor.p_sample_loop(sr=img)
        img_sr.clamp_(-1, 1)
        img_sr = (img_sr + 1) / 2
        return img_sr


class DirectoryDataset(Dataset):
    def __init__(self, root: str) -> None:
        self.root = root
        self.files = os.listdir(root)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[str, Tensor]:
        fp = os.path.join(self.root, self.files[idx])

        img = Image.open(fp)
        img = to_tensor(to_grayscale(img))
        return self.files[idx], img
