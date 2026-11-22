---
status: current
last_reviewed: 2026-11-22
review_every: 12mo
---

# AI-powered game

Building a text adventure game with an LLM: generating a consistent world, running the story loop, moderating content, and tracking game state as JSON.

**Roles:** AI engineer.
**Prerequisites:** [Prompt engineering](../prompt-eng/), [Structured output](../llm-structured-output/)

## Key concepts

- **Hierarchical generation.** Generate the world, then its kingdoms, then towns and characters, feeding each level's output into the next prompt, so details stay consistent ([L1](L1.ipynb)).
- **The game loop.** Each player action plus the story so far goes to the model, which narrates what happens next; a Gradio UI makes it playable ([L2](L2.ipynb)).
- **Moderation.** Llama Guard checks player input and model output against a content policy you define, such as "safe" or "moderate" violence for a game ([L3](L3.ipynb)).
- **Structured game mechanics.** Ask the model for JSON (for example, inventory changes) so code, not prose, tracks the game state ([L4](L4.ipynb)).
- **Save state.** The generated world is saved as JSON ([Kyropeia.json](Kyropeia.json)) so a game can be reloaded.

## Resources

| Resource | Type | Level | Cost |
|---|---|---|---|
| [Building an AI-powered game](https://www.deeplearning.ai/short-courses/building-an-ai-powered-game/) (Together AI, AI Dungeon) | Short course | Beginner | Free when reviewed |
| [Together AI documentation](https://docs.together.ai/) | Docs | Beginner | Free |

## Hands-on

| Lab | What it covers |
|---|---|
| [L1](L1.ipynb) | Generating a world, kingdoms and towns |
| [L2](L2.ipynb) | A playable game with a Gradio UI and an action loop |
| [L3](L3.ipynb) | Moderating content with Llama Guard and custom policies |
| [L4](L4.ipynb) | JSON game mechanics: an inventory detector |

The notebooks save worlds to a `../shared_data/` folder from the course environment; change that path to run them locally.

## Next steps

- [Guardrails](../../08-evaluation-and-safety/guardrails/)
- [Gradio](../gradio/)
