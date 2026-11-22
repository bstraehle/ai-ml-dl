---
status: current
last_reviewed: 2026-11-22
review_every: 12mo
---

# Reinforcement learning from human feedback (RLHF)

RLHF aligns a model with human preferences: people compare pairs of responses, a reward model learns those preferences, and reinforcement learning tunes the LLM to score well on them.

**Roles:** ML engineer, research engineer. AI engineers and PMs: read *Key concepts*.
**Prerequisites:** [Fine-tuning](../finetuning-llms/)

## Key concepts

- **Two datasets.** A *preference dataset* holds prompts with two candidate responses and which one a human preferred. A *prompt dataset* holds prompts for the RL stage ([L2](L2_explore_data.ipynb)).
- **Reward model.** A model trained on the preferences predicts which response a person would prefer; it becomes the training signal.
- **RL tuning.** The LLM generates responses, the reward model scores them, and an algorithm such as PPO updates the LLM. A KL penalty keeps it close to the original model, so it doesn't learn to game the reward.
- **Run it as a pipeline.** Reward-model training and RL tuning are long multi-step jobs; the labs use a Vertex AI pipeline ([L3](L3_tune_llm.ipynb)).
- **Evaluate the tuned model** with training curves and side-by-side comparisons against the untuned model ([L4](L4_evaluate_model.ipynb)).
- **Simpler alternatives.** Direct preference optimization (DPO) learns from the same preference data without a reward model or RL loop; see [Fine-tuning](../finetuning-llms/). Reasoning models use RL with automatically checkable rewards instead of human preferences; see [Reasoning models](../../02-llm-fundamentals/reasoning-models/).

## Resources

| Resource | Type | Level | Cost |
|---|---|---|---|
| [Reinforcement learning from human feedback](https://www.deeplearning.ai/short-courses/reinforcement-learning-from-human-feedback/) (Google Cloud) | Short course | Intermediate | Free when reviewed |
| [Illustrating RLHF](https://huggingface.co/blog/rlhf) (Hugging Face) | Article | Beginner | Free |
| [RLHF book](https://rlhfbook.com/) (Nathan Lambert) | Book | Advanced | Free |
| [Training language models to follow instructions with human feedback (InstructGPT)](https://arxiv.org/abs/2203.02155) | Paper | Advanced | Free |

## Hands-on

| Lab | What it covers |
|---|---|
| [L2_explore_data.ipynb](L2_explore_data.ipynb) | Preference and prompt datasets |
| [L3_tune_llm.ipynb](L3_tune_llm.ipynb) | Tuning an LLM with RLHF in a Vertex AI pipeline |
| [L4_evaluate_model.ipynb](L4_evaluate_model.ipynb) | Evaluating with TensorBoard and comparing tuned vs untuned |
| [L5_google_cloud_setup.ipynb](L5_google_cloud_setup.ipynb) | Setting up a Google Cloud project to run the labs |

## Next steps

- [Fine-tuning](../finetuning-llms/): SFT and DPO
- [Reasoning models](../../02-llm-fundamentals/reasoning-models/)
