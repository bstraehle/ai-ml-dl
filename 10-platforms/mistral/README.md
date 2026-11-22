---
status: current
last_reviewed: 2026-11-22
review_every: 6mo
---

# Mistral

Mistral AI makes open-weights and commercial models. This folder covers prompting them, choosing a model size, function calling, RAG and a simple chat UI.

**Roles:** AI engineer.
**Prerequisites:** [Prompt engineering](../../04-building-with-llms/prompt-eng/)

## Key concepts

- **Prompting basics** carry over: classification, extraction, spelling and grammar checks, with structured output where needed ([L2](L2_prompting_capabilities.ipynb)).
- **Choose the smallest model that does the job.** Small models are fast and cheap for simple tasks; larger ones handle harder reasoning. Test on your task, and check the docs for current model names, since lineups change quickly ([L3](L3_model_selection.ipynb)).
- **Function calling** lets the model ask your code for data instead of guessing ([L4](L4_function_calling.ipynb)); see [Function calling](../../04-building-with-llms/function-calling/).
- **RAG from scratch:** fetch a web page, chunk it, embed the chunks, retrieve the closest ones and answer from them ([L5](L5_basic_RAG.ipynb)).
- **Open weights** can run on your own hardware or a cloud provider, which helps with data residency and cost control.

## Resources

| Resource | Type | Level | Cost |
|---|---|---|---|
| [Getting started with Mistral](https://www.deeplearning.ai/short-courses/getting-started-with-mistral/) | Short course | Beginner | Free when reviewed |
| [Mistral documentation](https://docs.mistral.ai/) | Docs | Beginner | Free |

## Hands-on

| Lab | What it covers |
|---|---|
| [L2](L2_prompting_capabilities.ipynb) | Classification, extraction, spelling and grammar |
| [L3](L3_model_selection.ipynb) | Choosing between model sizes |
| [L4](L4_function_calling.ipynb) | Function calling |
| [L5](L5_basic_RAG.ipynb) | Basic RAG over a web article |
| [L6](L6_chat_ui.ipynb) | A chat UI and a RAG UI with Panel |

My Postman collection for the Mistral API is in [projects/rest-apis](../../projects/rest-apis/).

## Next steps

- [Open-source models](../open-source-models/)
- [Advanced RAG](../../05-rag-and-data/advanced-rag/)
