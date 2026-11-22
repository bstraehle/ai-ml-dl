---
status: current
last_reviewed: 2026-11-22
review_every: 12mo
---

# Gradio

Gradio builds a web UI for a model or AI app in a few lines of Python, which makes it the quickest way to demo, test and share what you build. All my [Hugging Face Spaces](../../projects/hugging-face/) use it.

**Roles:** AI engineer, ML engineer, data scientist.
**Prerequisites:** basic Python

## Key concepts

- **`gr.Interface`** wraps a function: declare the input and output components (text, image, audio) and Gradio builds the page ([L1](L1_NLP_tasks_with_a_simple_interface.ipynb)).
- **Any model behind it:** a local pipeline, a hosted inference endpoint, or an LLM API ([L2](L2_Image_captioning_app.ipynb), [L3](L3_Image_generation_app.ipynb)).
- **`gr.Blocks`** gives full control over layout and lets you chain several models in one app, such as captioning an image and then generating a new one from the caption ([L4](L4_Describe_and_generate_game.ipynb)).
- **`gr.Chatbot`** handles chat history and streaming for conversational apps ([L5](L5_Chat_with_any_LLM.ipynb)).
- **Share it.** `share=True` gives a temporary public link; Hugging Face Spaces host Gradio apps permanently. Gradio apps can also act as MCP servers for agents.

## Resources

| Resource | Type | Level | Cost |
|---|---|---|---|
| [Building generative AI applications with Gradio](https://www.deeplearning.ai/short-courses/building-generative-ai-applications-with-gradio/) (Hugging Face) | Short course | Beginner | Free when reviewed |
| [Gradio quickstart](https://gradio.app/guides/quickstart) | Docs | Beginner | Free |
| [Hugging Face Spaces](https://huggingface.co/docs/hub/spaces) | Docs | Beginner | Free |

## Hands-on

| Lab | What it builds |
|---|---|
| [L1](L1_NLP_tasks_with_a_simple_interface.ipynb) | Summarization and named-entity apps with `gr.Interface` |
| [L2](L2_Image_captioning_app.ipynb) | An image captioning app |
| [L3](L3_Image_generation_app.ipynb) | An image generation app |
| [L4](L4_Describe_and_generate_game.ipynb) | A describe-and-generate game with `gr.Blocks` |
| [L5](L5_Chat_with_any_LLM.ipynb) | A chat app with `gr.Chatbot` |

The labs call hosted inference endpoints from the course environment, so they need your own endpoints or models to run elsewhere.

## Next steps

- [My Hugging Face Spaces](../../projects/hugging-face/): real Gradio apps
- [MCP](../../06-agents/mcp/)
