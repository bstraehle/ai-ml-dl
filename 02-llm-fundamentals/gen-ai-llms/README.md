---
status: current
last_reviewed: 2026-11-22
review_every: 12mo
---

# Generative AI with LLMs

How large language models work, from tokens and transformers to the lifecycle of an LLM project: picking a model, prompting it, fine-tuning it and aligning it.

**Roles:** everyone building with LLMs. Product managers: the first week is enough.
**Prerequisites:** basic Python; [Deep learning](../../01-foundations/deep-learning/) helps but isn't required

## Key concepts

- **Tokens.** Models read and write tokens (word pieces), not words. Tokenizers differ between models, which affects cost, context length and odd behavior with numbers and rare words ([L2](L2.ipynb)).
- **The transformer** turns tokens into embeddings, mixes them with self-attention, and predicts a probability for every possible next token. Text is generated one token at a time ([L6](L6.ipynb)); see also [Attention in transformers](../attention-in-transformers/).
- **Model types.** Encoder-only models (BERT) understand text, decoder-only models (GPT, Llama) generate it, and encoder-decoder models (T5) transform it.
- **The project lifecycle:** define the use case, choose a model, adapt it (prompt, then RAG, then fine-tune), evaluate, deploy, monitor.
- **The optimization matrix.** Context optimization (what the model needs to know: prompting, then RAG) and model optimization (how it needs to act: fine-tuning) are separate axes, and many systems need both ([diagram](llm-matrix.png), from OpenAI's talk below).
- **In-context learning.** Zero-, one- and few-shot prompts change behavior without training ([Lab 1](Lab_1_summarize_dialogue.ipynb)).
- **Fine-tuning and alignment.** Full or parameter-efficient (LoRA) fine-tuning teaches a task ([Lab 2](Lab_2_fine_tune_generative_ai_model.ipynb)). RLHF with a reward model and PPO steers behavior, for example away from toxic output ([Lab 3](Lab_3_fine_tune_model_to_detoxify_summaries.ipynb)).
- **Scaling laws.** Quality improves predictably with model size, data and compute. Compute-optimal training balances parameters against training tokens.

## Resources

| Resource | Type | Level | Cost |
|---|---|---|---|
| [Generative AI with large language models](https://www.deeplearning.ai/courses/generative-ai-with-llms/) (AWS) | Course | Beginner | Free when reviewed |
| [How transformer LLMs work](https://www.deeplearning.ai/short-courses/how-transformer-llms-work/) | Short course | Beginner | Free when reviewed |
| [Intro to large language models](https://www.youtube.com/watch?v=zjkBMFhNj_g) (Andrej Karpathy) | Video | Beginner | Free |
| [Transformers, the tech behind LLMs](https://www.youtube.com/watch?v=wjZofJX0v4M) (3Blue1Brown) | Video | Beginner | Free |
| [A survey of techniques for maximizing LLM performance](https://www.youtube.com/watch?v=ahnGLM-RC1Y) (OpenAI) | Video | Intermediate | Free |

<details>
<summary>More</summary>

- [Deep dive into LLMs like ChatGPT](https://www.youtube.com/watch?v=7xTGNNLPyMI) (Andrej Karpathy): how models are pretrained, post-trained and why they behave the way they do

</details>

## Hands-on

| Lab | What it covers |
|---|---|
| [L2.ipynb](L2.ipynb) | Comparing the tokenizers of trained LLMs |
| [L6.ipynb](L6.ipynb) | Loading an LLM, generating text and a single token, exploring its architecture |
| [Lab_1_summarize_dialogue.ipynb](Lab_1_summarize_dialogue.ipynb) | Summarizing dialogue with zero-, one- and few-shot prompts |
| [Lab_2_fine_tune_generative_ai_model.ipynb](Lab_2_fine_tune_generative_ai_model.ipynb) | Full and LoRA fine-tuning of FLAN-T5 for summarization |
| [Lab_3_fine_tune_model_to_detoxify_summaries.ipynb](Lab_3_fine_tune_model_to_detoxify_summaries.ipynb) | RLHF with PPO and a toxicity reward model |

## Next steps

- [Prompt engineering](../../04-building-with-llms/prompt-eng/)
- [Reasoning models](../reasoning-models/): models that think before they answer
- [Fine-tuning](../../03-adapting-models/finetuning-llms/)
