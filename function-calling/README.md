---
status: current
last_reviewed: 2026-11-21
review_every: 6mo
---

# Function calling

Function calling (also called tool use) lets an LLM ask your code to do something: look something up, call an API, or write to a database. The model never runs anything itself. It returns a structured request, your code runs it, and you send back the result. Every agent is built on this.

**Roles:** AI engineer. Product managers: read *Key concepts*.
**Prerequisites:** [Prompt engineering](../prompt-eng/)

## Key concepts

- **The loop.** You send the conversation plus tool definitions. The model replies with a tool name and JSON arguments instead of text, your code runs the tool, and you append the result. Repeat until the model answers in text.
- **Tool definitions are prompts.** Each tool has a name, a description and a JSON Schema for its parameters. Clear descriptions and tight schemas (enums, required fields) matter more than anything else for picking the right tool with the right arguments.
- **Parallel and nested calls.** Models can request several independent calls at once, or chain calls where one result feeds the next ([Lesson 2](Lesson_2_Student.ipynb)).
- **Controlling it:** let the model decide (`auto`), force a call (`required` or a specific tool), or turn tools off (`none`).
- **Any API can become a tool.** An OpenAPI spec already describes endpoints and parameters, so it can be turned into tool definitions ([Lesson 3](Lesson_3_Student.ipynb)).
- **Extraction for free.** A "tool" that just takes a schema is a reliable way to pull structured data out of text ([Lesson 4](Lesson_4_Student.ipynb)); see also [Structured output](../llm-structured-output/).
- **Treat it like user input.** Validate arguments, allowlist what tools may do, ask before anything with side effects, and treat tool results as untrusted: they can carry prompt injections.
- **Small models can learn it.** Fine-tuning teaches an open model to call tools reliably ([HF Agents Course notebook](../finetuning-llms/hf-agents-course-function-calling.ipynb)). The Berkeley Function Calling Leaderboard compares models.

## Resources

| Resource | Type | Level | Cost |
|---|---|---|---|
| [Function-calling and data extraction with LLMs](https://www.deeplearning.ai/short-courses/function-calling-and-data-extraction-with-llms/) (Nexusflow) | Short course | Beginner | Free when reviewed |
| [OpenAI function calling guide](https://developers.openai.com/api/docs/guides/function-calling) | Docs | Beginner | Free |
| [Anthropic tool use overview](https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview) | Docs | Beginner | Free |
| [Berkeley Function Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html) | Benchmark | Intermediate | Free |
| [HF Agents Course: fine-tuning a model for function calling](https://huggingface.co/learn/agents-course/bonus-unit1/introduction) | Course unit | Advanced | Free |

## Hands-on

| Lab | What it covers |
|---|---|
| [Lesson 1](Lesson_1_Student.ipynb) | Writing local Python tools and letting the model call them |
| [Lesson 2](Lesson_2_Student.ipynb) | Parallel and nested calls with the NexusRaven model |
| [Lesson 3](Lesson_3_Student.ipynb) | Turning OpenAPI specs into tools for external APIs |
| [Lesson 4](Lesson_4_Student.ipynb) | Structured extraction (addresses) and generating valid JSON |
| [Lesson 5](Lesson_5_Student.ipynb) | Web search tools and chatting with a SQL database |
| [Lesson 6](Lesson_6_Student.ipynb) | A pipeline that extracts features from dialogues into a database |

## Next steps

- [Structured output](../llm-structured-output/): guarantee the shape of a model's answer
- [Agentic AI](../agentic-ai/): put tool calls in a loop
- [MCP](../mcp/): share tools across apps and models
