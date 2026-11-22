---
status: current
last_reviewed: 2026-11-22
review_every: 12mo
---

# Quantization

Quantization stores a model's weights (and sometimes activations) in fewer bits, for example 8 or 4 instead of 16 or 32. Models get smaller, faster and cheaper to run, at a small cost in quality.

**Roles:** ML engineer; AI engineers who run open models.
**Prerequisites:** [How LLMs work](../../02-llm-fundamentals/gen-ai-llms/), PyTorch basics

## Key concepts

- **Data types.** Integers (int8, int4) and floating points (fp32, fp16, bf16, fp8) trade range and precision for size. bf16 keeps fp32's range with less precision, which is why it's popular for training ([L2](L2_data_types.ipynb)).
- **Memory math.** A model needs roughly parameters × bytes per parameter: a 7B model is about 14 GB in 16-bit and about 4 GB in 4-bit.
- **Linear quantization** maps a range of floats to integers with a scale and a zero point. Finer granularity (per channel or per group of weights) loses less accuracy ([L4](L4_quantization_theory.ipynb)).
- **When to quantize.** Post-training quantization needs no retraining. Quantization-aware training and QLoRA fine-tuning recover more quality.
- **Common formats and tools:** bitsandbytes, GPTQ and AWQ for GPUs, and GGUF for llama.cpp on laptops and CPUs.
- **Measure the cost.** Check quality on your own task after quantizing; small models and low bit widths lose more.

## Resources

| Resource | Type | Level | Cost |
|---|---|---|---|
| [Quantization fundamentals with Hugging Face](https://www.deeplearning.ai/short-courses/quantization-fundamentals-with-hugging-face/) | Short course | Beginner | Free when reviewed |
| [Quantization in depth](https://www.deeplearning.ai/short-courses/quantization-in-depth/) | Short course | Intermediate | Free when reviewed |
| [A visual guide to quantization](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization) (Maarten Grootendorst) | Article | Intermediate | Free |
| [Transformers quantization overview](https://huggingface.co/docs/transformers/main/en/quantization/overview) | Docs | Intermediate | Free |

## Hands-on

| Lab | What it covers |
|---|---|
| [L2_data_types.ipynb](L2_data_types.ipynb) | Integer and floating-point types, sizes and downcasting |
| [L4_quantization_theory.ipynb](L4_quantization_theory.ipynb) | Linear quantization theory, applied to FLAN-T5 |

## Next steps

- [Serving LLMs efficiently](../../09-ops-and-deployment/efficiently-serving-llms/)
- [On-device AI](../../09-ops-and-deployment/on-device-ai/)
