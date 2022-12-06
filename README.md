# Cascaded Latent Diffusion for High-Resolution Chest X-ray Synthesis

<p align="center">
<img src=assets/intro_sample_grid.png />
</p>

This repository contains code for running and training **Cheff** - a cascaded **che**st
X-ray latent di**ff**usion pipeline.
The cheff pipeline consists of three cascading phases:

1. Modeling a diffusion process in latent space
2. Translating the latent variables into image space with a decoder
3. Refinement and upscaling using a super-resolution diffusion process

Phase 1 and 2 together define an LDM.
Phase 2 and 3 are trained on MaCheX, a collection of over 650,000 chest X-rays and thus,
build a foundational basis for our model stack.
The first phase is task-specific. For unconditional snythesis, we train on full MaCheX
and for report-to-chest-X-ray we use the MIMIC subset.

<p align="center">
<img src=assets/cheff_overview.png />
</p>

## How to use Cheff?

Please have a look into our [tutorial notebook](notebooks/01_cheff.ipynb).


## Models

We provide the weights for 5 models:

- Chest X-ray autoencoder: [Click](https://weights.released.on.accept)
- Chest X-ray super-resolution diffusion model base: [Click](https://weights.released.on.accept)
- Chest X-ray super-resolution diffusion model finetuned: [Click](https://weights.released.on.accept)
- Chest X-ray unconditioned semantic diffusion model: [Click](https://weights.released.on.accept)
- Chest X-ray report-conditioned semantic diffusion model: [Click](https://weights.released.on.accept)

## Training

Our codebase builds heavily on the classic LDM repository. Thus, we share the same
interface with a few adaptions.
A conda environment file for installing necessary dependencies is `environment.yml`.
The full config files are located in `configs`. After adjusting the paths, the training
can be started as follows:

```shell
python scripts/01_train_ldm.py -b <path/to/config.yml> -t --no-test
```


## Acknowledgements

This code builds heavily on the implementation of LDMs and DDPMs from CompVis:
[Repository here](https://github.com/CompVis/latent-diffusion).
