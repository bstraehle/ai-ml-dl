---
status: current
last_reviewed: 2026-11-22
review_every: 3mo
---

# Reasoning models

Reasoning models work through a problem step by step (a chain of thought) before they answer. They spend more computation at inference time and do much better on math, coding, planning and multi-step analysis, at the cost of speed and price.

**Roles:** AI engineer, ML engineer. Product managers: read *Key concepts* to know when they're worth it.
**Prerequisites:** [How LLMs work](../gen-ai-llms/), [Prompt engineering](../../04-building-with-llms/prompt-eng/)

## Key concepts

- **Test-time compute.** Instead of only making models bigger, let them think longer. Quality on hard problems rises with the number of reasoning tokens spent.
- **How they're trained.** Mostly reinforcement learning on tasks with checkable answers (math, code, puzzles), which rewards correct final answers and lets useful reasoning emerge. DeepSeek-R1 described this openly, including distilling the reasoning into smaller models.
- **When to use them:** multi-step problems, ambiguous tasks that need planning, code, and checking other models' work. For simple lookups, extraction or chat, a regular model is faster and cheaper.
- **Controls.** APIs expose a reasoning effort or thinking budget. Reasoning tokens are billed even when they're hidden, so leave a generous output limit or the model may run out before it answers.
- **Prompt them differently.** Be simple and direct, give the goal, constraints and what a good answer looks like, and skip "think step by step": they already do. Long few-shot examples can hurt ([prompting lab](../../10-platforms/openai/reasoning-with-o1/L2.ipynb)).
- **Useful patterns.** A reasoning model plans and a cheaper model executes each step ([planning lab](../../10-platforms/openai/reasoning-with-o1/L3.ipynb)). A reasoning model can also improve another model's prompt from evaluation results (metaprompting, [lab](../../10-platforms/openai/reasoning-with-o1/L6.ipynb)).

## Resources

| Resource | Type | Level | Cost |
|---|---|---|---|
| [Reasoning with o1](https://www.deeplearning.ai/short-courses/reasoning-with-o1/) (OpenAI) | Short course | Beginner | Free when reviewed |
| [OpenAI reasoning models guide](https://developers.openai.com/api/docs/guides/reasoning) | Docs | Beginner | Free |
| [Anthropic extended thinking](https://platform.claude.com/docs/en/build-with-claude/extended-thinking) | Docs | Beginner | Free |
| [Understanding reasoning LLMs](https://magazine.sebastianraschka.com/p/understanding-reasoning-llms) (Sebastian Raschka) | Article | Intermediate | Free |
| [DeepSeek-R1 paper](https://arxiv.org/abs/2501.12948) | Paper | Advanced | Free |

<details>
<summary>More</summary>

- [Open-R1](https://github.com/huggingface/open-r1) (Hugging Face): an open reproduction of DeepSeek-R1's training pipeline

</details>

## Hands-on

The [reasoning-with-o1](../../10-platforms/openai/reasoning-with-o1/) labs: prompting principles, planning with a reasoning model and a smaller executor, coding, reasoning over images, and metaprompting.

## Next steps

- [Agentic AI](../../06-agents/agentic-ai/): reasoning models as agent planners
- [LLM evals](../../08-evaluation-and-safety/llm-evals/): check that the extra cost pays off on your task
