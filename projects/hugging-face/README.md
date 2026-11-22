# Hugging Face Spaces

Source code for my public [Hugging Face Spaces](https://huggingface.co/bstraehle), mirrored from the Hub.
Don't edit these folders here: change the Space, then run `python .github/scripts/sync_hf_spaces.py`.

| Space | What it does | Source |
|---|---|---|
| [Advanced RAG](https://huggingface.co/spaces/bstraehle/advanced-rag) | Advanced RAG with pre- & post-retrieval processing | [advanced-rag/](advanced-rag/) |
| [Agentic AI](https://huggingface.co/spaces/bstraehle/agentic-ai) | Agentic AI with LangChain & LlamaIndex | [agentic-ai/](agentic-ai/) |
| [Amazon Bedrock](https://huggingface.co/spaces/bstraehle/bedrock) | Amazon Bedrock SDK with Anthropic Claude Opus 5.5 model | [bedrock/](bedrock/) |
| [Stockfish Chess Engine - MCP Server](https://huggingface.co/spaces/bstraehle/chess) | Evaluates chess positions | [chess/](chess/) |
| [General AI Assistant - MCP Client & Server](https://huggingface.co/spaces/bstraehle/gaia) | Solves GAIA Benchmark & Humanity's Last Exam problems (benchmark files aren't mirrored; see the Space) | [gaia/](gaia/) |
| [Multi-Agent AI - Deep Research](https://huggingface.co/spaces/bstraehle/multi-agent-ai-crewai-deep-research) | Multi-Agent AI with crewAI | [multi-agent-ai-crewai-deep-research/](multi-agent-ai-crewai-deep-research/) |
| [Multi-Agent AI - Chess](https://huggingface.co/spaces/bstraehle/multi-agent-ai-langgraph-chess) | Multi-Agent AI with LangGraph | [multi-agent-ai-langgraph-chess/](multi-agent-ai-langgraph-chess/) |
| [Multi-Agent AI - Deep Research](https://huggingface.co/spaces/bstraehle/multi-agent-ai-langgraph-deep-research) | Multi-Agent AI with LangGraph | [multi-agent-ai-langgraph-deep-research/](multi-agent-ai-langgraph-deep-research/) |
| [Multi-Agent AI - Chess](https://huggingface.co/spaces/bstraehle/multi-agent-ai-microsoft-chess) | Multi-Agent AI with Microsoft Agent Framework | [multi-agent-ai-microsoft-chess/](multi-agent-ai-microsoft-chess/) |
| [Multi-Agent AI - Coding](https://huggingface.co/spaces/bstraehle/multi-agent-ai-microsoft-coding) | Multi-Agent AI with Microsoft Agent Framework | [multi-agent-ai-microsoft-coding/](multi-agent-ai-microsoft-coding/) |
| [OpenAI](https://huggingface.co/spaces/bstraehle/multimodal-ai) | OpenAI SDK with GPT-6.2 Sol model | [multimodal-ai/](multimodal-ai/) |
| [Google Vertex AI](https://huggingface.co/spaces/bstraehle/vertex-ai) | Google Gen AI SDK on Vertex AI with Gemini 4 Argon model | [vertex-ai/](vertex-ai/) |

**gaia:** Dependabot reports litellm advisories for `gaia/requirements.txt`. litellm stays at 1.80.0 because stagehand 0.5.x (the last release that the crewai-tools Stagehand browser tool supports) caps it there. Most advisories concern the LiteLLM Proxy server, which this Space doesn't run; it uses litellm only as a library.
