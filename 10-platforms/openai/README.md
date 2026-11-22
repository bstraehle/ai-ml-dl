---
status: current
last_reviewed: 2026-11-22
review_every: 6mo
---

# OpenAI

Building with OpenAI's models: prompting for developers, building multi-step systems with checks and evaluation, writing and coding with Canvas, and using reasoning models.

**Roles:** AI engineer.
**Prerequisites:** basic Python

## Key concepts

- **Two prompting principles:** write clear, specific instructions (delimiters, structured output, conditions, examples), and give the model time to think by asking for steps ([prompt-eng l2](chatgpt-prompt-eng/l2-guidelines.ipynb)).
- **Iterate.** Improve prompts in a loop against examples, rather than hoping for a perfect first prompt ([l3](chatgpt-prompt-eng/l3-iterative-prompt-development.ipynb)).
- **Common tasks:** summarizing, inferring (sentiment, topics), transforming (translation, tone, format) and expanding text, plus a chatbot with a message history.
- **Build systems, not single prompts.** Classify the input, check it with a moderation model, process it with chained prompts, check the output, and evaluate the whole pipeline ([building-systems](chatgpt-building-systems/)).
- **Chat format and tokens.** Messages have roles (system, user, assistant), and cost and limits are counted in tokens ([L1](chatgpt-building-systems/L1_student.ipynb)).
- **Reasoning models** need simpler prompts and are best for planning, coding and analysis; see [Reasoning models](../../02-llm-fundamentals/reasoning-models/) and the [o1 labs](reasoning-with-o1/).
- **Model names change often.** The labs use older models; check the current model list in the docs.

## Resources

| Resource | Type | Level | Cost |
|---|---|---|---|
| [ChatGPT prompt engineering for developers](https://www.deeplearning.ai/short-courses/chatgpt-prompt-engineering-for-developers/) | Short course | Beginner | Free when reviewed |
| [Building systems with the ChatGPT API](https://www.deeplearning.ai/short-courses/building-systems-with-chatgpt/) | Short course | Beginner | Free when reviewed |
| [Reasoning with o1](https://www.deeplearning.ai/short-courses/reasoning-with-o1/) | Short course | Beginner | Free when reviewed |
| [OpenAI API documentation](https://developers.openai.com/api/docs) | Docs | All levels | Free |
| [OpenAI Cookbook](https://developers.openai.com/cookbook) | Examples | Intermediate | Free |

<details>
<summary>More</summary>

- [Collaborative writing and coding with OpenAI Canvas](https://www.deeplearning.ai/short-courses/collaborative-writing-and-coding-with-openai-canvas/)

</details>

## Hands-on

| Folder | What the labs cover |
|---|---|
| [chatgpt-prompt-eng/](chatgpt-prompt-eng/) | Guidelines, iterative development, summarizing, inferring, transforming, expanding, a chatbot |
| [chatgpt-building-systems/](chatgpt-building-systems/) | Chat format and tokens, classification, moderation, chain of thought, chaining prompts, checking outputs, an end-to-end system, evaluation |
| [reasoning-with-o1/](reasoning-with-o1/) | Prompting o1, planning, coding, reasoning with images, metaprompting |
| [access-and-prompt-copies.ipynb](access-and-prompt-copies.ipynb) | Prompts from the Canvas course |

My [OpenAI Space](../../projects/hugging-face/multimodal-ai/) and the OpenAI collection in [projects/rest-apis](../../projects/rest-apis/) show the API in use.

## Next steps

- [Prompt engineering](../../04-building-with-llms/prompt-eng/)
- [Function calling](../../04-building-with-llms/function-calling/)
