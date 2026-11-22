---
status: current
last_reviewed: 2026-11-22
review_every: 12mo
---

# Federated learning

Federated learning trains a shared model across many devices or organizations without moving their data. Each client trains locally and sends only model updates to a server, which combines them.

**Roles:** ML engineer, research engineer; anyone working with sensitive or distributed data.
**Prerequisites:** [Deep learning](../../01-foundations/deep-learning/), PyTorch basics

## Key concepts

- **Why:** data that can't be centralized for privacy, legal or practical reasons (hospitals, phones, banks), and models that would otherwise only see one client's data ([course 1, L1](1/L1_Why_Federated_Learning.ipynb)).
- **The training loop.** The server sends the model, clients train on local data, the server averages the updates (FedAvg), and the round repeats ([L2](1/L2_Federated_Training_Process.ipynb)).
- **Tuning.** Client selection, local epochs and the aggregation strategy all change accuracy and speed ([L3](1/L3_Tuning.ipynb)).
- **Privacy isn't automatic.** Updates can leak information about the data; differential privacy adds noise to protect individuals ([L4](1/L4_Data_Privacy.ipynb)).
- **Bandwidth is a real limit.** Sending full models every round is expensive, especially for large models ([L5](1/L5_Bandwidth.ipynb)).
- **Federated LLM fine-tuning.** Parameter-efficient methods make it practical to fine-tune LLMs across clients, and checking memorization helps keep training data private ([course 2](2/)).

## Resources

| Resource | Type | Level | Cost |
|---|---|---|---|
| [Intro to federated learning](https://www.deeplearning.ai/short-courses/intro-to-federated-learning/) (Flower Labs, two courses) | Short course | Intermediate | Free when reviewed |
| [Flower](https://flower.ai/) · [documentation](https://flower.ai/docs/framework/) | Framework | Intermediate | Free |

## Hands-on

| Lab | What it covers |
|---|---|
| [1/L1](1/L1_Why_Federated_Learning.ipynb) | Why federated learning: models trained on partial data |
| [1/L2](1/L2_Federated_Training_Process.ipynb) | The federated training process with Flower |
| [1/L3](1/L3_Tuning.ipynb) | Tuning clients and the server strategy |
| [1/L4](1/L4_Data_Privacy.ipynb) | Differential privacy |
| [1/L5](1/L5_Bandwidth.ipynb) | Bandwidth costs |
| [2/L2](2/L2_Centralized_LLM_Fine-tuning.ipynb) | Centralized LLM fine-tuning as a baseline |
| [2/L3](2/L3_Federated_LLM_Fine-tuning.ipynb) | Federated LLM fine-tuning |
| [2/L4](2/L4_Keeping-LLM-private.ipynb) | Measuring memorization with perplexity to keep data private |

## Next steps

- [On-device AI](../on-device-ai/)
- [AI security](../../08-evaluation-and-safety/ai-security/)
