---
status: current
last_reviewed: 2026-11-22
review_every: 6mo
---

# Serving LLMs efficiently

How LLM inference servers get high throughput and low latency from expensive GPUs: caching, batching, quantization, and serving many fine-tuned adapters from one base model.

**Roles:** ML engineer, platform engineer. AI engineers: read *Key concepts* to understand cost and latency.
**Prerequisites:** [How LLMs work](../../02-llm-fundamentals/gen-ai-llms/), PyTorch basics

## Key concepts

- **Two phases.** *Prefill* processes the whole prompt at once (compute-bound) and determines time to first token. *Decode* generates one token at a time (memory-bound) and determines tokens per second.
- **KV cache.** Keep each layer's attention keys and values for tokens already processed, so each new token only computes its own. It trades GPU memory for a large speedup ([Lesson 1](Lesson_1-Text_Generation.ipynb)).
- **Batching.** Serving several requests together uses the GPU far better, but static batches wait for their slowest request ([Lesson 2](Lesson_2-Batching.ipynb)).
- **Continuous batching** adds and removes requests from the running batch at every step, which raises throughput without hurting latency much ([Lesson 3](Lesson_3-Continuous_Batching.ipynb)).
- **Quantization** stores weights in fewer bits, so bigger models or bigger batches fit in memory ([Lesson 4](Lesson_4-Quantization.ipynb)); see [Quantization](../../03-adapting-models/quantization/).
- **LoRA and multi-LoRA.** A LoRA adapter is a tiny add-on to a shared base model. A server can keep one base model in memory and apply a different adapter per request in the same batch, which makes many fine-tuned variants cheap to serve ([Lessons 5–7](Lesson_5-Low-Rank_Adaptation.ipynb)).
- **Use an inference server.** Production systems use engines such as vLLM or LoRAX that implement these techniques (plus paged attention and speculative decoding) instead of hand-written loops.

## Resources

| Resource | Type | Level | Cost |
|---|---|---|---|
| [Efficiently serving LLMs](https://www.deeplearning.ai/short-courses/efficiently-serving-llms/) (Predibase) | Short course | Intermediate | Free when reviewed |
| [vLLM documentation](https://docs.vllm.ai/) | Docs | Intermediate | Free |
| [LoRAX](https://github.com/predibase/lorax) | Server | Intermediate | Free |

## Hands-on

| Lab | What it covers |
|---|---|
| [Lesson 1](Lesson_1-Text_Generation.ipynb) | Token-by-token generation and the KV cache |
| [Lesson 2](Lesson_2-Batching.ipynb) | Batching requests with padding |
| [Lesson 3](Lesson_3-Continuous_Batching.ipynb) | Continuous batching |
| [Lesson 4](Lesson_4-Quantization.ipynb) | Quantizing weights to save memory |
| [Lesson 5](Lesson_5-Low-Rank_Adaptation.ipynb) | How LoRA adapters work |
| [Lesson 6](Lesson_6-Multi-LoRA.ipynb) | Serving many LoRA adapters at once |
| [predibase_lorax.ipynb](predibase_lorax.ipynb) | Prefill vs decode, continuous batching and multi-LoRA with LoRAX |

## Next steps

- [On-device AI](../on-device-ai/): inference on phones and edge hardware
- [LLMOps](../llm-ops/)
