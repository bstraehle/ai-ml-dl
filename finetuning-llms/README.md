---
status: current
last_reviewed: 2026-11-21
review_every: 6mo
---

# Fine-tuning LLMs

Fine-tuning continues training a pretrained model on your own examples so it reliably does a specific task, in a specific format or style. It's the step after prompting and RAG, not instead of them.

**Roles:** ML engineer, AI engineer, data scientist. Product managers: read *Key concepts*.
**Prerequisites:** [How LLMs work](../gen-ai-llms/), Python and PyTorch basics

## Key concepts

- **When to fine-tune.** Try prompting first, then RAG. Fine-tuning teaches *behavior*: a format, a tone, a narrow task, tool use. RAG supplies *knowledge* that changes. Fine-tuning a small model can also cut cost and latency compared with prompting a large one.
- **Where it fits:** pretraining (next-token prediction on web-scale text) gives a base model. Instruction tuning turns it into a chat model, and task fine-tuning specializes it further ([Finetuning LLMs labs 2 and 3](llamini-1/)).
- **Data matters most.** A few hundred to a few thousand high-quality, consistent examples beat a large noisy set. Use one prompt template throughout, and handle tokenization, padding and truncation carefully ([lab 4](llamini-1/04_Data_preparation_lab_student.ipynb)).
- **Parameter-efficient fine-tuning.** LoRA trains small adapter matrices instead of all the weights. QLoRA does the same on a 4-bit [quantized](../quantization/) model, so 7B-class models fit on one GPU.
- **Preference tuning.** Supervised fine-tuning (SFT) shows the model good answers. Direct preference optimization (DPO) then teaches it to prefer better answers over worse ones, without the separate reward model that [RLHF](../rlhf/) needs ([SFT and DPO notebooks](sft-dpo/)).
- **Synthetic data** can bootstrap a training set when real examples are scarce: generate examples with a strong model, filter them, then fine-tune ([Improving accuracy labs](llamini-2/)).
- **Evaluate before and after.** Hold out a test set, measure the task metric, and check that general abilities didn't regress.

## Resources

| Resource | Type | Level | Cost |
|---|---|---|---|
| [Finetuning large language models](https://www.deeplearning.ai/short-courses/finetuning-large-language-models/) (Lamini) | Short course | Beginner | Free when reviewed |
| [Improving accuracy of LLM applications](https://www.deeplearning.ai/short-courses/improving-accuracy-of-llm-applications/) (Lamini, Meta) | Short course | Intermediate | Free when reviewed |
| [Aligning LLMs with direct preference optimization](https://www.youtube.com/watch?v=QXVCqtAZAn4) (DeepLearning.AI, Hugging Face) | Video | Intermediate | Free |
| [Hugging Face TRL](https://huggingface.co/docs/trl/index) | Docs | Intermediate | Free |
| [Unsloth](https://github.com/unslothai/unsloth) | Library | Intermediate | Free |

<details>
<summary>More</summary>

- [Alignment Handbook](https://github.com/huggingface/alignment-handbook) (Hugging Face): recipes for SFT and DPO
- [DPO talk slides](https://docs.google.com/presentation/d/1S8ao40-CdclRU0D2D9FdyN5x8fZL1Iv5) (Google Slides)
- Colab notebooks: [supervised fine-tuning (SFT)](https://colab.research.google.com/drive/1WNSVtM82oknmzL1QrJlNu--yNaWbp6o9) · [preference fine-tuning with DPO](https://colab.research.google.com/drive/1mWiOFBy3zY6OdINEvHN9EPoQ_VIvfFKw)
- [Fine-tune OpenAI models with Weights & Biases](http://wandb.me/openai-colab) (Colab)
- [HF Agents Course: fine-tuning a model for function calling](https://huggingface.co/learn/agents-course/bonus-unit1/introduction)

</details>

## Hands-on

| Folder | What the labs cover |
|---|---|
| [llamini-1/](llamini-1/) | Fine-tuned vs base models, finetuning data, instruction tuning, data preparation, training, evaluation |
| [llamini-2/](llamini-2/) | A Llama 3 SQL agent: diagnosing hallucinations, building an evaluation, generating data and fine-tuning |
| [sft-dpo/](sft-dpo/) | Supervised fine-tuning, then DPO, with Hugging Face tooling |
| [hf-agents-course-function-calling.ipynb](hf-agents-course-function-calling.ipynb) | Fine-tuning Gemma 2 for function calling with LoRA (Hugging Face Agents Course) |

## Next steps

- [RLHF](../rlhf/): the reinforcement-learning route to alignment
- [Quantization](../quantization/): make fine-tuned models smaller and cheaper to run
- [Serving LLMs efficiently](../efficiently-serving-llms/): serve many LoRA adapters on one base model
