---
status: current
last_reviewed: 2026-11-22
review_every: 12mo
---

# Pretraining LLMs

Pretraining is the first and most expensive stage of building an LLM: training a model from scratch to predict the next token on a very large text corpus. This folder walks through it at a small scale.

**Roles:** ML engineer, research engineer. Others: read *Key concepts* to understand where models come from.
**Prerequisites:** [How LLMs work](../gen-ai-llms/), PyTorch basics

## Key concepts

- **When pretraining makes sense:** a new language or domain that existing models handle poorly, or a fully controlled model. Otherwise, fine-tuning an existing model is far cheaper ([Lesson 1](Lesson_1.ipynb)).
- **Data is the biggest lever.** Pretraining data is huge, diverse, deduplicated and filtered for quality, unlike small, curated fine-tuning sets ([Lesson 2](Lesson_2.ipynb)).
- **Packing.** Tokenize the text and pack it into fixed-length sequences so no compute is wasted on padding ([Lesson 3](Lesson_3.ipynb)).
- **Model configuration and initialization.** Choose layers, width and heads; start from random weights, or speed up by reusing or upscaling a smaller trained model ([Lesson 4](Lesson_4.ipynb)).
- **Training** runs for many steps with a learning-rate schedule, checkpoints and loss tracking ([Lesson 5](Lesson_5.ipynb)).
- **Evaluate** on standard benchmarks and on samples you read yourself ([Lesson 6](Lesson_6.ipynb)).
- **Cost scales with tokens × parameters.** Scaling laws estimate the compute needed, which is why only a few organizations pretrain frontier models.

## Resources

| Resource | Type | Level | Cost |
|---|---|---|---|
| [Pretraining LLMs](https://www.deeplearning.ai/short-courses/pretraining-llms/) (Upstage) | Short course | Intermediate | Free when reviewed |
| [Let's reproduce GPT-2 (124M)](https://www.youtube.com/watch?v=l8pRSuU81PU) (Andrej Karpathy) | Video | Advanced | Free |
| [nanoGPT](https://github.com/karpathy/nanoGPT) | Code | Advanced | Free |
| [FineWeb: decanting the web for the finest text data](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1) (Hugging Face) | Article | Advanced | Free |

## Hands-on

| Lab | What it covers |
|---|---|
| [Lesson 1](Lesson_1.ipynb) | Why pretraining: comparing general and specialized models |
| [Lesson 2](Lesson_2.ipynb) | Sourcing and cleaning pretraining data |
| [Lesson 3](Lesson_3.ipynb) | Tokenizing and packing the data |
| [Lesson 4](Lesson_4.ipynb) | Model configuration and weight initialization |
| [Lesson 5](Lesson_5.ipynb) | Training the model |
| [Lesson 6](Lesson_6.ipynb) | Evaluating the model |

## Next steps

- [Fine-tuning](../../03-adapting-models/finetuning-llms/)
- [Reasoning models](../reasoning-models/)
