---
status: current
last_reviewed: 2026-11-21
review_every: 6mo
---

# Llama

Llama is Meta's family of open-weights models. You can download, run and fine-tune them yourself, or use them through cloud providers. This folder has three DeepLearning.AI courses with Meta, from prompting Llama 2 to multimodal, long-context Llama 4.

**Roles:** AI engineer, ML engineer.
**Prerequisites:** [Prompt engineering](../../04-building-with-llms/prompt-eng/)

## Key concepts

- **Base vs chat models.** Base models continue text; chat (instruct) models follow instructions. Use chat models unless you're fine-tuning.
- **Prompt formats.** Each generation wraps messages in its own special tokens and roles (system, user, assistant, and for Llama 3.x and 4, tool calls). Libraries usually apply the chat template for you, but it helps to know what the model actually sees.
- **Models are stateless.** For multi-turn chat, you send the whole conversation every time.
- **Multimodal.** Llama 3.2 Vision and Llama 4 accept images: OCR, charts and tables, and image grounding (locating objects).
- **Long context.** Llama 4 can take whole books or codebases in one prompt.
- **Tool calling.** Llama 3.x models know built-in tools (such as Brave search) and custom tools defined in the system prompt.
- **Safety models.** Llama Guard classifies prompts and responses against a safety policy, and works with any model.

## Resources

| Resource | Type | Level | Cost |
|---|---|---|---|
| [Prompt engineering with Llama 2](https://www.deeplearning.ai/short-courses/prompt-engineering-with-llama-2/) | Short course | Beginner | Free when reviewed |
| [Introducing multimodal Llama 3.2](https://www.deeplearning.ai/short-courses/introducing-multimodal-llama-3-2/) | Short course | Beginner | Free when reviewed |
| [Building with Llama 4](https://www.deeplearning.ai/short-courses/building-with-llama-4/) | Short course | Intermediate | Free when reviewed |
| [Meta Llama models on Hugging Face](https://huggingface.co/meta-llama) | Models | All levels | Free |

## Hands-on

| Folder | What the labs cover |
|---|---|
| [llama-2-prompt-eng/](llama-2-prompt-eng/) | Chat vs base models, multi-turn prompts, prompting techniques, comparing Llama models, Code Llama, Llama Guard |
| [llama-3-2-prompt-eng/](llama-3-2-prompt-eng/) | Multimodal prompting, OCR and multi-image use cases, prompt format, tokenization, tool calling, Llama Stack |
| [building-with-llama-4/](building-with-llama-4/) | The Llama API, image reasoning and grounding, prompt format, long context, prompt optimization, synthetic data |

## Next steps

- [Fine-tuning](../../03-adapting-models/finetuning-llms/): the Improving Accuracy labs fine-tune Llama 3
- [Open-source models](../open-source-models/): more open models through Hugging Face
- [Quantization](../../03-adapting-models/quantization/) and [On-device AI](../../09-ops-and-deployment/on-device-ai/): run open models on smaller hardware
