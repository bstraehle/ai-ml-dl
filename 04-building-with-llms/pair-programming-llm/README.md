---
status: current
last_reviewed: 2026-11-22
review_every: 6mo
---

# Pair programming with LLMs

Using an LLM as a coding partner: writing, improving, testing, debugging and explaining code. The same habits apply to today's AI coding assistants and coding agents.

**Roles:** AI engineer, ML engineer, any developer.
**Prerequisites:** basic Python

## Key concepts

- **Reusable prompt templates** keep coding requests consistent: role, task, the code, and the output you want ([L2](L2-template.ipynb)).
- **Common scenarios:** improve or simplify existing code, write tests, make code more efficient, and debug errors ([L3](L3-pair_program.ipynb)).
- **Paying down technical debt.** LLMs are good at explaining unfamiliar or complex code and writing documentation for it ([L4](L4-technical_debt.ipynb)).
- **Give context.** The more of the codebase, requirements and constraints the model sees, the better the result. Coding agents gather this context themselves by reading files and running commands.
- **Review everything.** Generated code can be subtly wrong, insecure or out of date. Run it, test it and read it before you merge it.
- **Work in small steps.** Ask for one change at a time, check it, then continue.

## Resources

| Resource | Type | Level | Cost |
|---|---|---|---|
| [Pair programming with a large language model](https://www.deeplearning.ai/short-courses/pair-programming-llm/) (Google) | Short course | Beginner | Free when reviewed |
| [Gemini API documentation](https://aistudio.google.com/docs) | Docs | Beginner | Free |

## Hands-on

| Lab | What it covers |
|---|---|
| [L1](L1-palm.ipynb) | Getting started with the API and its models |
| [L2](L2-template.ipynb) | String templates for coding prompts |
| [L3](L3-pair_program.ipynb) | Improving, simplifying, testing, optimizing and debugging code |
| [L4](L4-technical_debt.ipynb) | Explaining and documenting complex code |

The labs use Google's PaLM API, which Google has since replaced with the Gemini API.

## Next steps

- [Agentic AI](../../06-agents/agentic-ai/): the Windsurf and Replit coding-agent courses
- [AI security](../../08-evaluation-and-safety/ai-security/): risks of agents that run code
