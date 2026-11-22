---
status: current
last_reviewed: 2026-11-22
review_every: 12mo
---

# Deep learning

Neural networks in depth: how they learn, how to make them train well, how to run ML projects, and the architectures behind vision and language (CNNs, RNNs and transformers).

**Roles:** ML engineer, research engineer; useful background for AI engineers.
**Prerequisites:** [Machine learning](../machine-learning/), [Mathematics](../mathematics/)

## Key concepts

- **Forward and backward propagation.** A network computes predictions layer by layer, then backpropagation computes how each weight affected the error so gradient descent can update it.
- **Training well:** initialization, regularization (dropout, weight decay), normalization (batch norm), optimizers (momentum, Adam), learning-rate schedules, and systematic hyperparameter search.
- **Structuring ML projects.** Pick one evaluation metric, split train/dev/test from the right distributions, compare with human-level performance to separate avoidable bias from variance, and use error analysis to choose what to fix next.
- **Transfer and multi-task learning** reuse what a model learned on one task for another, which is why pretrained models dominate.
- **Convolutional networks (CNNs)** exploit the structure of images: residual networks, object detection, face recognition, style transfer.
- **Sequence models.** RNNs, LSTMs and GRUs process sequences step by step. Word embeddings represent meaning. Attention and the transformer replaced recurrence and power today's LLMs.

## Resources

| Resource | Type | Level | Cost |
|---|---|---|---|
| [Deep Learning Specialization](https://www.deeplearning.ai/courses/deep-learning-specialization/) (DeepLearning.AI) | Course | Intermediate | Free to audit when reviewed |
| [Neural Networks: Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) (Andrej Karpathy) | Videos | Intermediate | Free |
| [Practical Deep Learning for Coders](https://course.fast.ai/) (fast.ai) | Course | Beginner | Free |
| [Dive into Deep Learning](https://d2l.ai/) | Book | Intermediate | Free |

## Hands-on

The specialization's labs run on its own platform. In this repo, see [Attention in transformers](../../02-llm-fundamentals/attention-in-transformers/) for PyTorch code and [Diffusion models](../../07-multimodal/diffusion-models/) for a U-Net trained from scratch.

## Next steps

- [How LLMs work](../../02-llm-fundamentals/gen-ai-llms/)
- [Attention in transformers](../../02-llm-fundamentals/attention-in-transformers/)
