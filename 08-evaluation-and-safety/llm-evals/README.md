---
status: current
last_reviewed: 2026-11-22
review_every: 6mo
---

# LLM evals

Evals are the tests of an LLM application: a set of inputs plus a way to score the outputs. Without them, every prompt, model or retrieval change is a guess. With them, you can improve an app steadily and catch regressions before users do.

**Roles:** AI engineer, ML engineer. Product managers: evals are how you define "good enough".
**Prerequisites:** [Prompt engineering](../../04-building-with-llms/prompt-eng/)

## Key concepts

- **Start with error analysis, not metrics.** Read real conversations or traces, write down what went wrong, group the failures, and write evals for the most common ones. Generic scores ("helpfulness 4.2") rarely tell you what to fix.
- **Three kinds of graders:**
  - *Code-based:* assertions on format, length, required fields, forbidden words, valid JSON or SQL. Cheap and deterministic; use them wherever you can.
  - *Reference-based:* compare to a known answer (exact match, similarity), for tasks with one right answer.
  - *LLM-as-judge:* a model grades against a rubric, for open-ended output. Check the judge against a sample of human labels before trusting it, and prefer pass/fail or pairwise comparisons over 1–10 scores.
- **Offline and online.** Offline evals run on a fixed test set before every release. Online evaluation samples production traffic and user feedback, and new failures go back into the test set.
- **Your evals beat public benchmarks.** Leaderboards help shortlist models, but they may not match your task and can be contaminated by training data. Decide with your own examples.
- **RAG and agents need their own evals.** For RAG, score retrieval and generation separately (context relevance, groundedness, answer relevance). For agents, score the steps and tool choices, not only the final answer.
- **Be statistically honest.** Use enough examples, compare versions on the same set, and rerun noisy evals: LLM outputs vary from run to run.

## Resources

| Resource | Type | Level | Cost |
|---|---|---|---|
| [Your AI product needs evals](https://hamel.dev/blog/posts/evals/) (Hamel Husain) | Article | Beginner | Free |
| [OpenAI evals guide](https://developers.openai.com/api/docs/guides/evals) | Docs | Beginner | Free |
| [Evaluating AI agents](https://www.deeplearning.ai/short-courses/evaluating-ai-agents/) | Short course | Intermediate | Free when reviewed |
| [Inspect](https://inspect.aisi.org.uk/) (UK AI Security Institute) | Framework | Intermediate | Free |
| [LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness) (EleutherAI) | Framework | Advanced | Free |

## Hands-on

Eval labs elsewhere in the repo:

| Lab | What it shows |
|---|---|
| [RAG triad of metrics](../../05-rag-and-data/advanced-rag/llamaindex/L2-RAG_Triad_of_metrics.ipynb) | Context relevance, groundedness and answer relevance for RAG |
| [Evaluating AI agents](../../06-agents/agentic-ai/evaluating-ai-agents/) | Tracing, router and skill evals, trajectory evals |
| [Creating an evaluation](../../03-adapting-models/finetuning-llms/llamini-2/Lesson3.ipynb) | Building an eval set for a SQL agent |
| [Model-graded evaluation](../../10-platforms/llama/llama-2-prompt-eng/L5_comparing_llama_models.ipynb) | Using an LLM to grade summaries and reasoning |
| [Hallucination metrics](../quality-safety-llm-apps/L2_Hallucinations.ipynb) | Relevance, self-similarity and LLM self-evaluation |

## Next steps

- [Quality & safety](../quality-safety-llm-apps/): monitoring in production
- [LLMOps](../../09-ops-and-deployment/llm-ops/): run evals in your release pipeline
