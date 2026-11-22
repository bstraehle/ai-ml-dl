---
status: current
last_reviewed: 2026-11-22
review_every: 6mo
---

# Weights & Biases

Weights & Biases (W&B, now part of CoreWeave) tracks ML experiments, versions models and data, and traces and evaluates LLM applications, so you can compare runs and debug what changed.

**Roles:** ML engineer, AI engineer, data scientist.
**Prerequisites:** [Machine learning](../../01-foundations/machine-learning/)

## Key concepts

- **Track every run.** Log hyperparameters, metrics and sample outputs, and compare runs side by side instead of keeping results in notebooks ([intro](01_intro_starter.ipynb)).
- **Generative models need visual logs.** Log generated images during diffusion training and sampling to see quality change over time ([training](02_diffusion_training_starter.ipynb), [sampling](03_diffusion_sampling_starter.ipynb)).
- **Evaluate and trace LLM apps.** Tables compare prompts and outputs; traces record every step of a chain or agent ([LLM eval](04_llm_eval_starter.ipynb)). Weave is W&B's newer toolkit for this.
- **Fine-tuning runs** log losses and sample generations to spot overfitting early ([fine-tuning](05_train_llm_starter.ipynb)).
- **Artifacts and the model registry** version datasets and checkpoints, record which data produced which model, and promote models to production ([model-registry](model-registry/)).

## Resources

| Resource | Type | Level | Cost |
|---|---|---|---|
| [Evaluating and debugging generative AI](https://www.deeplearning.ai/short-courses/evaluating-debugging-generative-ai/) (W&B) | Short course | Beginner | Free when reviewed |
| [W&B course notebooks](https://github.com/wandb/edu/tree/main) | Notebooks | All levels | Free |
| [W&B Weave documentation](https://docs.coreweave.com/products/wandb/weave) | Docs | Intermediate | Free |

<details>
<summary>More</summary>

- [W&B examples](https://github.com/wandb/examples)
- [wandbot](https://github.com/wandb/wandbot): W&B's own RAG support bot, a production case study
- [Fully Connected reports](https://forge.coreweave.com/forge-reports)
- [My W&B projects](https://forge.coreweave.com/wandb/bstraehle/)

</details>

## Hands-on

| Lab | What it covers |
|---|---|
| [01_intro_starter.ipynb](01_intro_starter.ipynb) | Tracking a sprite classification model |
| [02_diffusion_training_starter.ipynb](02_diffusion_training_starter.ipynb) | Training a diffusion model with W&B |
| [03_diffusion_sampling_starter.ipynb](03_diffusion_sampling_starter.ipynb) | Comparing diffusion samplers |
| [04_llm_eval_starter.ipynb](04_llm_eval_starter.ipynb) | Evaluating and tracing LLM calls |
| [05_train_llm_starter.ipynb](05_train_llm_starter.ipynb) | Fine-tuning a language model |
| [model-registry/](model-registry/) | Logging models from PyTorch, Lightning and Hugging Face; logging by reference; consuming registered models |

## Next steps

- [LLM evals](../../08-evaluation-and-safety/llm-evals/)
- [LLMOps](../llm-ops/)
