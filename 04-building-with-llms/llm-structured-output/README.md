---
status: current
last_reviewed: 2026-11-22
review_every: 6mo
---

# Structured output

Getting LLM output in a shape your code can rely on (JSON that matches a schema, one of a fixed set of choices, a valid phone number) instead of parsing free text and hoping.

**Roles:** AI engineer.
**Prerequisites:** [Prompt engineering](../prompt-eng/), basic Python

## Key concepts

- **Define the shape once.** Describe the output as a schema, usually a Pydantic model, and use it both to request and to validate the response ([L2](L_2.ipynb)).
- **Three ways to get structure:**
  - *Provider structured outputs:* the API guarantees JSON that matches your schema. Simplest when your provider supports it.
  - *Retry-based:* validate the answer and re-ask with the error if it fails (the Instructor library). Works with any model, but costs extra calls and can still fail ([L3](L_3.ipynb)).
  - *Constrained generation:* at each step, mask out tokens that would break the format, so the output is valid by construction (the Outlines library). Needs access to the model's logits, so mostly for open models ([L4](L_4.ipynb)).
- **Beyond JSON.** Constrained generation can enforce any regular expression or grammar: a choice from a list, phone numbers, emails, HTML tags, even legal moves in a game ([L5](L_5.ipynb)).
- **Valid isn't correct.** A schema guarantees shape, not truth. Keep fields simple, use enums where you can, and still check the values.
- **Function calling is related.** A tool's parameter schema is structured output too; see [Function calling](../function-calling/).

## Resources

| Resource | Type | Level | Cost |
|---|---|---|---|
| [Getting structured LLM output](https://www.deeplearning.ai/short-courses/getting-structured-llm-output/) (DotTxt) | Short course | Beginner | Free when reviewed |
| [OpenAI structured outputs guide](https://developers.openai.com/api/docs/guides/structured-outputs) | Docs | Beginner | Free |
| [Instructor](https://python.useinstructor.com/) | Library | Intermediate | Free |
| [Outlines](https://github.com/dottxt-ai/outlines) | Library | Intermediate | Free |

## Hands-on

| Lab | What it covers |
|---|---|
| [L_2.ipynb](L_2.ipynb) | Structured outputs with Pydantic: extracting social media mentions |
| [L_3.ipynb](L_3.ipynb) | Retry-based structured output with Instructor, and when retries fail |
| [L_4.ipynb](L_4.ipynb) | Constrained generation with Outlines and chat templates |
| [L_5.ipynb](L_5.ipynb) | Beyond JSON: choices, phone numbers, emails, HTML tags, tic-tac-toe |

## Next steps

- [Function calling](../function-calling/)
- [Agentic AI](../../06-agents/agentic-ai/): agents depend on reliable structured steps
