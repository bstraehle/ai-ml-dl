---
status: current
last_reviewed: 2026-11-22
review_every: 6mo
---

# Haystack

Haystack (by deepset) is an open-source Python framework for LLM applications built as explicit pipelines of components: retrievers, prompt builders, generators, routers and your own custom steps.

**Roles:** AI engineer.
**Prerequisites:** [Advanced RAG](../../05-rag-and-data/advanced-rag/)

## Key concepts

- **Components and pipelines.** Each component does one job with typed inputs and outputs; a pipeline connects them into a graph you can inspect and reuse ([L1](Lesson_1.ipynb)).
- **Document stores** hold your indexed documents for retrieval.
- **Customized RAG.** Separate indexing and query pipelines, with your own prompt template ([L2](Lesson_2.ipynb)).
- **Custom components** are plain Python classes, so any API or logic can join a pipeline ([L3](Lesson_3.ipynb)).
- **Branching.** Conditional routes fall back to another path, such as web search, when retrieval finds nothing ([L4](Lesson_4.ipynb)).
- **Loops and agents.** Pipelines can loop until a validator passes (self-reflection) and call functions as tools in a chat agent ([L5](Lesson_5.ipynb), [L6](Lesson_6.ipynb)).

## Resources

| Resource | Type | Level | Cost |
|---|---|---|---|
| [Building AI applications with Haystack](https://www.deeplearning.ai/short-courses/building-ai-applications-with-haystack/) | Short course | Intermediate | Free when reviewed |
| [Haystack documentation](https://docs.haystack.deepset.ai/) | Docs | Intermediate | Free |
| [Haystack on GitHub](https://github.com/deepset-ai/haystack) | Code | Intermediate | Free |

## Hands-on

| Lab | What it covers |
|---|---|
| [Lesson 1](Lesson_1.ipynb) | Components, pipelines and document stores |
| [Lesson 2](Lesson_2.ipynb) | Building a customized RAG pipeline |
| [Lesson 3](Lesson_3.ipynb) | A custom component: a news summarizer |
| [Lesson 4](Lesson_4.ipynb) | Fallbacks with branching pipelines |
| [Lesson 5](Lesson_5.ipynb) | A self-reflecting agent with loops |
| [Lesson 6](Lesson_6.ipynb) | A chat agent with function calling |

## Next steps

- [LangChain](../langchain/) and [Semantic Kernel](../microsoft-semantic-kernel/): other frameworks
- [Agentic AI](../../06-agents/agentic-ai/)
