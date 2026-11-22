---
status: current
last_reviewed: 2026-11-21
review_every: 6mo
---

# Quality and safety of LLM applications

LLM apps fail in ways traditional software doesn't: they make things up, leak data, produce toxic output, refuse good requests and follow injected instructions. This folder covers measuring those failures in real traffic and acting on them.

**Roles:** AI engineer, ML engineer. Product managers and leaders: read *Key concepts*.
**Prerequisites:** [Prompt engineering](../../04-building-with-llms/prompt-eng/)

## Key concepts

- **Log every prompt and response with metrics.** You can't review LLM output by eye at scale. Compute metrics on each exchange and watch their distributions over time (the labs use whylogs and LangKit).
- **Hallucinations.** Low prompt-response relevance is a warning sign; you can measure it with embeddings, BLEU or BERTScore. Sampling the same prompt several times and checking whether the answers agree (self-similarity) is another, as is asking an LLM to grade the answer ([Lesson 2](L2_Hallucinations.ipynb)).
- **Data leakage.** Detect personal data and secrets in prompts and responses, using patterns (emails, phone and card numbers) and named-entity recognition ([Lesson 3](L3_Data_leakage_and_toxicity.ipynb)).
- **Toxicity** is caught by classifiers on both input and output.
- **Refusals, jailbreaks and prompt injections.** Spot refusals with string matching and sentiment. Spot attacks by their similarity to known jailbreaks, and with injection classifiers ([Lesson 4](L4_Refusals_and_prompt_injections.ipynb)).
- **Passive vs active monitoring.** Passive monitoring analyzes logs and alerts afterwards. Active monitoring is a guardrail that blocks or rewrites a response before the user sees it ([Lesson 5](L5_Passive_and_active_monitoring.ipynb)). Most systems need both.
- **Use a shared checklist.** The OWASP Top 10 for LLM applications names the main risks (prompt injection, sensitive data disclosure, excessive agency, and more). For organizations, the NIST AI Risk Management Framework is the reference for governance.

## Resources

| Resource | Type | Level | Cost |
|---|---|---|---|
| [Quality and safety for LLM applications](https://www.deeplearning.ai/short-courses/quality-safety-llm-applications/) (WhyLabs) | Short course | Beginner | Free when reviewed |
| [OWASP Top 10 for LLM applications](https://genai.owasp.org/llm-top-10/) | Reference | All levels | Free |
| [LangKit](https://github.com/whylabs/langkit) | Library | Intermediate | Free |
| [NIST AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework) | Framework | Leaders | Free |

## Hands-on

| Lab | What it covers |
|---|---|
| [Lesson 1](L1_Overview.ipynb) | Logging prompts and responses with whylogs and LangKit; an overview of the metrics |
| [Lesson 2](L2_Hallucinations.ipynb) | Relevance (BLEU, BERTScore), response self-similarity, LLM self-evaluation |
| [Lesson 3](L3_Data_leakage_and_toxicity.ipynb) | Pattern and entity detection for data leakage; toxicity |
| [Lesson 4](L4_Refusals_and_prompt_injections.ipynb) | Detecting refusals, jailbreaks and prompt injections |
| [Lesson 5](L5_Passive_and_active_monitoring.ipynb) | Passive monitoring and building your own active guardrails |

## Next steps

- [Guardrails](../guardrails/): validate inputs and outputs in code
- [Red teaming](../red-teaming-llm-apps/): attack your app before others do
- [LLMOps](../../09-ops-and-deployment/llm-ops/): run evaluation and monitoring as part of deployment
