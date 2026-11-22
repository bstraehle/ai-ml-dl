---
status: current
last_reviewed: 2026-11-22
review_every: 6mo
---

# Guardrails

Guardrails are checks that run on an LLM app's inputs and outputs and block, fix or flag anything that breaks your rules: hallucinations, off-topic answers, leaked personal data, competitor mentions, unsafe content.

**Roles:** AI engineer. Product managers: read *Key concepts*.
**Prerequisites:** [Quality & safety](../quality-safety-llm-apps/)

## Key concepts

- **Know the failure modes.** RAG apps hallucinate, drift off topic, leak personal data, and say things the business doesn't want said, such as recommending a competitor ([Lesson 1](Lesson_1.ipynb)).
- **Three ways to implement a check** ([diagram](guardrails.png)):
  - *Rules:* regular expressions, keyword lists, pattern matching. Fast and predictable, but brittle.
  - *Small fine-tuned models:* classifiers for topic, toxicity, factuality, or named-entity recognition. Fast and accurate for one job.
  - *A second LLM call* that grades the output. Flexible, but slower and more expensive.

  Real systems combine them.
- **Validators and guards.** In Guardrails AI, a validator checks one thing, and a guard runs validators on input or output and decides what happens on failure: raise, fix, filter or re-ask ([Lesson 3](Lesson_3.ipynb)).
- **Hallucination checks with NLI.** A natural-language-inference model tests whether each sentence of the answer is supported by the retrieved context ([Lessons 4–5](Lesson_4.ipynb)).
- **Topic, PII and competitor checks:** a zero-shot classifier keeps a chatbot on topic, Microsoft Presidio finds and redacts personal data, and entity matching catches competitor names ([Lessons 6–8](Lesson_6.ipynb)).
- **Mind the latency.** Every guard adds time. Run cheap checks first, run checks in parallel, and stream output where you can.

## Resources

| Resource | Type | Level | Cost |
|---|---|---|---|
| [Safe and reliable AI via guardrails](https://www.deeplearning.ai/short-courses/safe-and-reliable-ai-via-guardrails/) (Guardrails AI) | Short course | Intermediate | Free when reviewed |
| [Guardrails AI](https://github.com/guardrails-ai/guardrails) · [Guardrails Hub](https://guardrailsai.com/hub) | Library, validators | Intermediate | Free |
| [NeMo Guardrails](https://github.com/NVIDIA-NeMo/Guardrails) (NVIDIA) | Library | Intermediate | Free |
| [Presidio](https://github.com/data-privacy-stack/presidio) | Library | Intermediate | Free |

## Hands-on

| Lab | What it covers |
|---|---|
| [Lesson 1](Lesson_1.ipynb) | Failure modes in a RAG chatbot |
| [Lesson 3](Lesson_3.ipynb) | Your first validator and guard; running a Guardrails server |
| [Lesson 4](Lesson_4.ipynb) | A hallucination validator using NLI |
| [Lesson 5](Lesson_5.ipynb) | Using the hallucination guard in a chatbot |
| [Lesson 6](Lesson_6.ipynb) | Keeping a chatbot on topic: zero-shot classifiers vs LLMs |
| [Lesson 7](Lesson_7.ipynb) | Preventing PII leaks with Presidio |
| [Lesson 8](Lesson_8.ipynb) | Preventing competitor mentions |

## Next steps

- [Red teaming](../red-teaming-llm-apps/): test whether your guardrails hold
- [AI security](../ai-security/)
