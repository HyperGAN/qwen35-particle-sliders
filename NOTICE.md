Current training uses HyperGAN/particle-sliders core 0.13.0 and released
255BITS/ParticleGAN 0.9.5 (MIT), tag commit
1b8355b184b21bb7d81104ee590813c262f6df0a. Exact shared-core pin is in
pyproject.toml; the core verifies the ParticleGAN release source closure.
GAN mathematics is imported from ParticleGAN; the routed architecture and ridge
fitter are imported from particle-sliders. No GAN loss is copied into this repo.

The following attribution/pins describe the preserved v0.1.0 experiment:

This integration is MIT licensed. It executes, rather than reimplements, the
historical upstream ParticleGAN formulation and ridge distillation code.

- Reference integration: mikkel/meta-glimmer-particle-slider,
  commit 80f22afd04d0794c1408c0058ec2b5f674a27b03 (MIT).
- Shared historical formulation: HyperGAN/particle-sliders,
  commit 1f5b380d317b73779a8aeab63b6bec0093c2c65c (MIT).
- ParticleGAN dependency: 255BITS/ParticleGAN,
  commit 82b72661c03377f348ccb76de37cd4ca02d45a8d (MIT).
- Delta League CLI/scorer/artifact tools: ilblackdragon/deltaleague, version
  0.1.1. Registered Qwen weights retain their Apache-2.0 license and are not
  committed to this repository.
- GSM8K training data: openai/grade-school-math (MIT); downloaded separately,
  hash recorded in data and run contracts. No benchmark examples or raw model
  generation logs are committed here.
