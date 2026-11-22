---
status: current
last_reviewed: 2026-11-22
review_every: 12mo
---

# Attention in transformers

Attention is the mechanism that lets each token in a transformer look at every other token and decide which ones matter. This folder codes it from scratch in PyTorch.

**Roles:** ML engineer, research engineer; curious AI engineers.
**Prerequisites:** [Deep learning](../../01-foundations/deep-learning/), PyTorch basics

## Key concepts

- **Queries, keys and values.** Each token is projected into a query (what it's looking for), a key (what it offers) and a value (what it passes on).
- **Scaled dot-product attention.** Compare each query with every key, scale, apply softmax to get weights, and take the weighted sum of values ([Lesson 3](Lesson_3.ipynb)).
- **Masked self-attention.** Decoder models such as GPT hide future tokens, so each token only attends to earlier ones; that's what makes next-token generation work ([Lesson 6](Lesson_6.ipynb)).
- **Encoder-decoder attention.** In translation-style models, the decoder's queries attend to the encoder's keys and values ([Lesson 9](Lesson_9.ipynb)).
- **Multi-head attention** runs several attention operations in parallel so the model can track different relationships at once.
- **Cost grows with the square of the sequence length,** which is why long contexts are expensive and why KV caching and efficient attention variants matter.

## Resources

| Resource | Type | Level | Cost |
|---|---|---|---|
| [Attention in transformers: concepts and code in PyTorch](https://www.deeplearning.ai/short-courses/attention-in-transformers-concepts-and-code-in-pytorch/) (StatQuest) | Short course | Intermediate | Free when reviewed |
| [Attention in transformers, step-by-step](https://www.youtube.com/watch?v=eMlx5fFNoYc) (3Blue1Brown) | Video | Beginner | Free |
| [The illustrated transformer](https://jalammar.github.io/illustrated-transformer/) (Jay Alammar) | Article | Beginner | Free |
| [Attention is all you need](https://arxiv.org/abs/1706.03762) | Paper | Advanced | Free |

## Hands-on

| Lab | What it covers |
|---|---|
| [Lesson 3](Lesson_3.ipynb) | Coding self-attention |
| [Lesson 6](Lesson_6.ipynb) | Coding masked self-attention |
| [Lesson 9](Lesson_9.ipynb) | Coding encoder-decoder attention |

## Next steps

- [Generative AI with LLMs](../gen-ai-llms/)
- [Serving LLMs efficiently](../../09-ops-and-deployment/efficiently-serving-llms/): the KV cache
