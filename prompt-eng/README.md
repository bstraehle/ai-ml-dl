---
status: current
last_reviewed: 2026-11-21
review_every: 6mo
---

# Prompt engineering

Prompt engineering is writing instructions and context so a model reliably does what you want. It's the cheapest way to improve an LLM app, and the first thing to try before RAG or fine-tuning.

**Roles:** everyone who uses LLMs; AI engineers in depth.
**Prerequisites:** none

## Key concepts

- **Be clear and specific.** State the task, the audience, the constraints and what a good answer looks like. Most bad outputs come from vague prompts.
- **Give context.** Include the facts the model needs instead of hoping it knows them. Feeding relevant documents in automatically is [RAG](../advanced-rag/).
- **Use the system prompt** for the role, rules and output format that apply to every turn. User messages carry the specific request.
- **Show, don't only tell.** A few examples (few-shot prompting) pin down format and style better than descriptions do.
- **Structure the prompt.** Separate instructions, context and examples with headings, delimiters or XML-style tags, and ask for a specific output format such as JSON.
- **Let the model reason.** Asking for step-by-step reasoning (chain of thought) helps on multi-step problems. Reasoning models do this on their own, so for them focus on the goal and constraints instead.
- **Iterate with evaluations.** Keep a small set of test inputs, trace every run, and compare prompt versions on the same inputs instead of judging single outputs (the lab traces runs with W&B Weave).

## Resources

| Resource | Type | Level | Cost |
|---|---|---|---|
| [Prompt engineering course notebooks](https://github.com/wandb/edu/tree/main/prompting) (Weights & Biases) | Notebooks | Beginner | Free |
| [OpenAI prompt engineering guide](https://developers.openai.com/api/docs/guides/prompt-engineering) | Docs | Beginner | Free |
| [Anthropic prompt engineering overview](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/overview) | Docs | Beginner | Free |
| [Prompt Engineering Guide](https://www.promptingguide.ai/) | Guide | All levels | Free |
| [Prompt engineering](https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/) (Lilian Weng) | Article | Intermediate | Free |

## Hands-on

| Lab | What it covers |
|---|---|
| [prompt_engineering.ipynb](prompt_engineering.ipynb) | Building a prompting assistant step by step: raw prompts, context, system prompts for inputs and outputs, zero-shot, few-shot and chain of thought, traced with W&B Weave |

More prompting labs elsewhere in the repo:
- [openai/](../openai/): ChatGPT prompt engineering for developers, building systems with ChatGPT, reasoning with o1
- [llama/](../llama/): prompting Llama 2, Llama 3.2 and Llama 4
- [vision-models/](../vision-models/): prompt engineering for vision models

## Next steps

- [Function calling](../function-calling/) and [Structured output](../llm-structured-output/)
- [Advanced RAG](../advanced-rag/): give the model the right context automatically
