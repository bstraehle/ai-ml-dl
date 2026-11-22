---
status: current
last_reviewed: 2026-11-22
review_every: 12mo
---

# Vision models

Prompting vision models with points, boxes and text to segment, detect and edit images, and fine-tuning an image generator on your own subject.

**Roles:** AI engineer, ML engineer.
**Prerequisites:** [Prompt engineering](../../04-building-with-llms/prompt-eng/), [Diffusion models](../diffusion-models/) helps

## Key concepts

- **Prompts aren't only text.** Segmentation models such as SAM take points or boxes as prompts and return a mask of the object there ([L2](L2_Image_Segmentation.ipynb)).
- **Zero-shot detection.** Models like OWL-ViT find objects described in plain text, with no training on those classes; their boxes can then prompt a segmentation model ([L3](L3_Object_Detection.ipynb)).
- **Inpainting** regenerates only the masked part of an image from a text prompt, so you can replace an object while keeping the rest ([L4](L4_Image_Generation.ipynb)).
- **Personalized generation.** DreamBooth fine-tunes Stable Diffusion on a few photos of a subject, so it can draw that subject in new scenes ([L5](L5_Fine_Tuning.ipynb)).
- **Track experiments.** Image work is hard to compare by eye across runs; the labs log images and parameters with Comet.
- **Vision-language models** now also answer questions about images directly, as in the [OpenAI vision lab](../../10-platforms/openai/reasoning-with-o1/L5.ipynb) and [Llama 3.2 labs](../../10-platforms/llama/llama-3-2-prompt-eng/).

## Resources

| Resource | Type | Level | Cost |
|---|---|---|---|
| [Prompt engineering for vision models](https://www.deeplearning.ai/short-courses/prompt-engineering-for-vision-models/) (Comet) | Short course | Intermediate | Free when reviewed |
| [SAM 2](https://github.com/facebookresearch/sam2) (Meta) | Model | Intermediate | Free |
| [OWL-ViT](https://huggingface.co/docs/transformers/model_doc/owlvit) | Docs | Intermediate | Free |
| [Hugging Face Diffusers](https://huggingface.co/docs/diffusers/index) | Library | Intermediate | Free |

## Hands-on

| Lab | What it covers |
|---|---|
| [L2](L2_Image_Segmentation.ipynb) | Segmentation with point and box prompts |
| [L3](L3_Object_Detection.ipynb) | Zero-shot detection with OWL-ViT, then segmentation |
| [L4](L4_Image_Generation.ipynb) | Inpainting with Stable Diffusion |
| [L5](L5_Fine_Tuning.ipynb) | DreamBooth fine-tuning |

## Next steps

- [Diffusion models](../diffusion-models/)
- [Open-source models](../../10-platforms/open-source-models/): more vision pipelines
