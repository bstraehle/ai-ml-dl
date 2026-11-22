---
status: current
last_reviewed: 2026-11-22
review_every: 3mo
---

# Computer use

Computer use lets a model operate a computer like a person: it looks at screenshots and sends mouse and keyboard actions. This folder builds up to it with the Anthropic API: messages, images, prompting, prompt caching and tool use.

**Roles:** AI engineer.
**Prerequisites:** [Function calling](../../04-building-with-llms/function-calling/)

## Key concepts

- **The Messages API.** Conversations are lists of user and assistant messages; prefilling the start of the assistant's reply steers its format ([L2](Lesson_2.ipynb)).
- **Multimodal input and streaming.** Messages hold content blocks of text and images, and responses can stream ([L3](Lesson_3.ipynb)).
- **Real-world prompts** are long and structured: role, rules, context in XML-style tags, examples and output format ([L4](Lesson_4.ipynb)).
- **Prompt caching** reuses a long, unchanging prefix (a book, a codebase, a big system prompt) across requests for lower cost and latency ([L5](Lesson_5.ipynb)).
- **Tool use** is the foundation: the model asks for a tool with JSON arguments and your code runs it ([L6](Lesson_6.ipynb)).
- **Computer use is tool use with a screen.** The tools are "take a screenshot", "click", "type" and so on, run in a loop until the task is done. Run it in an isolated virtual machine or container, and watch for prompt injection from what's on screen; see [AI security](../../08-evaluation-and-safety/ai-security/).

## Resources

| Resource | Type | Level | Cost |
|---|---|---|---|
| [Building towards computer use with Anthropic](https://www.deeplearning.ai/short-courses/building-towards-computer-use-with-anthropic/) | Short course | Beginner | Free when reviewed |
| [Computer use tool](https://platform.claude.com/docs/en/agents-and-tools/tool-use/computer-use-tool) (Anthropic docs) | Docs | Intermediate | Free |
| [Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching) (Anthropic docs) | Docs | Intermediate | Free |
| [Anthropic courses](https://github.com/anthropics/courses) | Notebooks | Beginner | Free |

## Hands-on

| Lab | What it covers |
|---|---|
| [Lesson 2](Lesson_2.ipynb) | The Messages API, a simple chatbot, prefilling |
| [Lesson 3](Lesson_3.ipynb) | Images and streaming |
| [Lesson 4](Lesson_4.ipynb) | Consumer vs enterprise-grade prompts |
| [Lesson 5](Lesson_5.ipynb) | Prompt caching with a book |
| [Lesson 6](Lesson_6.ipynb) | Tool use with a fake database |

## Next steps

- [Agentic AI](../agentic-ai/): browser agents and coding agents
- [MCP](../mcp/)
