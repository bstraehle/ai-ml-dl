---
status: current
last_reviewed: 2026-11-22
review_every: 12mo
---

# Diffusion models

How image generators like Stable Diffusion work: a network learns to remove noise step by step, so starting from pure noise it can produce a new image.

**Roles:** ML engineer, research engineer; curious AI engineers.
**Prerequisites:** [Deep learning](../../01-foundations/deep-learning/), PyTorch basics

## Key concepts

- **Forward process.** Gradually add Gaussian noise to training images until nothing but noise is left.
- **Reverse process.** Train a network (usually a U-Net) to predict the noise in a noisy image at any step. Subtracting the predicted noise, a little at a time, turns noise back into an image ([sampling](L1_Sampling.ipynb), [training](L2_Training.ipynb)).
- **Add a bit of noise back while sampling.** Without it, samples collapse into blurry averages; the labs show this.
- **Context and conditioning.** Feeding an embedding of a label or a text prompt into the network steers what it generates ([context](L3_Context.ipynb)). Text-to-image models condition on text embeddings.
- **Faster sampling.** DDPM needs hundreds of steps. DDIM skips steps deterministically and is much faster for similar quality ([fast sampling](L4_FastSampling.ipynb)).
- **Latent diffusion.** Stable Diffusion runs the process in a compressed latent space instead of pixels, which makes high-resolution generation affordable.

## Resources

| Resource | Type | Level | Cost |
|---|---|---|---|
| [How diffusion models work](https://www.deeplearning.ai/short-courses/how-diffusion-models-work/) | Short course | Intermediate | Free when reviewed |
| [What are diffusion models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) (Lilian Weng) | Article | Advanced | Free |
| [Hugging Face Diffusers](https://huggingface.co/docs/diffusers/index) | Library | Intermediate | Free |
| [Denoising diffusion probabilistic models (DDPM)](https://arxiv.org/abs/2006.11239) | Paper | Advanced | Free |

## Hands-on

| Lab | What it covers |
|---|---|
| [L1_Sampling.ipynb](L1_Sampling.ipynb) | Sampling from a trained model, and why noise is added back |
| [L2_Training.ipynb](L2_Training.ipynb) | Training the noise-prediction U-Net |
| [L3_Context.ipynb](L3_Context.ipynb) | Conditioning generation on context |
| [L4_FastSampling.ipynb](L4_FastSampling.ipynb) | DDIM vs DDPM speed |

## Next steps

- [Vision models](../vision-models/): prompting, inpainting and fine-tuning Stable Diffusion
