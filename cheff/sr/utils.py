"""Miscellaneous classes and functions."""
from typing import Any, Optional

import numpy as np
from matplotlib import animation
from matplotlib import pyplot as plt
from torch import Tensor
from torchvision.transforms import (
    Compose,
    Lambda,
    ToPILImage,
)
from torchvision.utils import make_grid


def transform_tensor_to_img() -> Compose:
    """Transform a tensor with a single element to a PIL image."""
    return Compose(
        [
            Lambda(lambda t: t.detach().cpu()),
            Lambda(lambda t: (t + 1) / 2),
            Lambda(lambda t: t.permute(1, 2, 0)),
            Lambda(lambda t: t * 255.0),
            Lambda(lambda t: t.numpy().astype(np.uint8)),
            ToPILImage(),
        ]
    )


def plot_image(
    img: Tensor,
    fig_size: Any = None,
    ncols: Optional[int] = None,
    show: bool = True,
    save_path: Optional[str] = None,
) -> None:
    """Plot a tensor containing image data."""
    img = img.detach().cpu()

    # Shape of 4 implies multiple image inputs
    if len(img.shape) == 4:
        img = make_grid(img, nrow=ncols if ncols is not None else len(img))

    plt.figure(figsize=fig_size)
    plt.imshow(img.permute(1, 2, 0))
    plt.axis('off')

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')

    if show:
        plt.show()
    plt.close()


def make_gif(
    img_arr: Tensor,
    save_path: str,
) -> None:
    """Create a GIF with the output of DiffusionController.generate()."""
    assert len(img_arr) == 5, 'Array has wrong shape.'

    img_arr = img_arr.detach().cpu()

    fig = plt.figure(frameon=False)
    ims = []

    for img_t in img_arr:
        grid = make_grid(img_t, nrow=img_t.shape[0] // 2)
        im = plt.imshow(grid.permute(1, 2, 0), animated=True)
        plt.axis('off')
        plt.tight_layout()
        ims.append([im])

    fig.tight_layout()

    animate = animation.ArtistAnimation(
        fig, ims, interval=100, blit=True, repeat_delay=2000
    )
    animate.save(save_path)
