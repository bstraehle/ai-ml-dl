---
status: current
last_reviewed: 2026-11-22
review_every: 6mo
---

# LLMOps

LLMOps applies MLOps practice to LLM applications: versioned data and prompts, automated tuning and evaluation pipelines, controlled deployment, and monitoring in production.

**Roles:** ML engineer, AI engineer, platform engineer.
**Prerequisites:** [MLOps](../machine-learning-eng/), [Fine-tuning](../../03-adapting-models/finetuning-llms/)

## Key concepts

- **Data first.** Build training and evaluation datasets with SQL or dataframes, version them, and keep them separate so evaluation stays honest.
- **Pipelines, not notebooks.** Express data preparation, tuning and evaluation as pipeline steps (for example Kubeflow Pipelines), so runs are reproducible, scheduled and comparable.
- **Version everything:** data, prompts, model and adapter versions, and evaluation results, so any production answer can be traced back.
- **Deploy behind an endpoint** with load balancing, and watch latency, cost per request and error rates ([L4](L4_predictions_prompts_safety.ipynb)).
- **Safety is part of serving.** Check prompts and responses with safety attributes or classifiers at prediction time, and log the scores ([L4](L4_predictions_prompts_safety.ipynb)).
- **Close the loop.** Feed production failures back into evaluation sets, and re-run evaluations whenever a prompt, model or retrieval setting changes; see [LLM evals](../../08-evaluation-and-safety/llm-evals/).

## Resources

| Resource | Type | Level | Cost |
|---|---|---|---|
| [LLMOps](https://www.deeplearning.ai/short-courses/llmops/) (Google Cloud) | Short course | Intermediate | Free when reviewed |
| [Kubeflow Pipelines](https://www.kubeflow.org/docs/components/pipelines/) | Docs | Intermediate | Free |

## Hands-on

| Lab | What it covers |
|---|---|
| [L4_predictions_prompts_safety.ipynb](L4_predictions_prompts_safety.ipynb) | Deploying a tuned model, load balancing, getting predictions and safety attributes |

## Next steps

- [Serving LLMs efficiently](../efficiently-serving-llms/)
- [LLM evals](../../08-evaluation-and-safety/llm-evals/)
- [Weights & Biases](../wandb/): experiment tracking and tracing
