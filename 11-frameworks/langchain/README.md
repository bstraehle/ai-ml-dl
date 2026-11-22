---
status: current
last_reviewed: 2026-11-22
review_every: 6mo
---

# LangChain

LangChain is an open-source framework (Python and JavaScript) for LLM applications: models, prompts, output parsers, retrieval and tools behind common interfaces. LangGraph, from the same team, adds stateful agent workflows.

**Roles:** AI engineer.
**Prerequisites:** [Prompt engineering](../../04-building-with-llms/prompt-eng/)

## Key concepts

- **Building blocks:** chat models, prompt templates and output parsers that turn text into structured data ([app-dev L1](app-dev/L1-Model_prompt_parser.ipynb)).
- **LCEL** (LangChain Expression Language) composes steps with `|` into chains that support streaming and batching ([functions L2](functions-tools-agents/L2-lcel-student.ipynb), [JS L1](langchain-js/L1_student.ipynb)).
- **Chat with your data:** load documents, split them, embed them into a vector store, retrieve, and answer with sources ([chat-with-data](chat-with-data/)).
- **Better retrieval:** maximal marginal relevance for diverse results, metadata filters, and self-querying ([chat-with-data 04](chat-with-data/04_retrieval.ipynb)).
- **Memory** keeps conversation history; conversational retrieval rewrites follow-up questions so they make sense on their own ([chat-with-data 06](chat-with-data/06_chat.ipynb)).
- **Functions, tools and agents:** Pydantic models become function definitions for tagging, extraction and routing, and an agent loop picks tools ([functions-tools-agents](functions-tools-agents/)).
- **Evaluate** chains with generated test sets and LLM grading ([app-dev L5](app-dev/L5-Evaluation.ipynb)).
- **APIs move fast.** The courses use older class names (`LLMChain`, `ConversationBufferMemory`); check the current docs, and use LangGraph for new agent work.

## Resources

| Resource | Type | Level | Cost |
|---|---|---|---|
| [LangChain for LLM application development](https://www.deeplearning.ai/short-courses/langchain-for-llm-application-development/) | Short course | Beginner | Free when reviewed |
| [LangChain: chat with your data](https://www.deeplearning.ai/short-courses/langchain-chat-with-your-data/) | Short course | Beginner | Free when reviewed |
| [Functions, tools and agents with LangChain](https://www.deeplearning.ai/short-courses/functions-tools-agents-langchain/) | Short course | Intermediate | Free when reviewed |
| [Build LLM apps with LangChain.js](https://www.deeplearning.ai/short-courses/build-llm-apps-with-langchain-js/) | Short course | Beginner | Free when reviewed |
| [LangChain documentation](https://docs.langchain.com/) | Docs | All levels | Free |

<details>
<summary>More</summary>

- [LangChain on YouTube](https://www.youtube.com/@LangChain)

</details>

## Hands-on

| Folder | What the labs cover |
|---|---|
| [app-dev/](app-dev/) | Models, prompts and parsers; memory; chains; Q&A over documents; evaluation; agents |
| [chat-with-data/](chat-with-data/) | Loading, splitting, embeddings and vector stores, retrieval, question answering, chat |
| [functions-tools-agents/](functions-tools-agents/) | OpenAI functions, LCEL, function calling, tagging and extraction, tools and routing, a conversational agent |
| [langchain-js/](langchain-js/) | Building blocks, loading data, vector stores, Q&A, conversational Q&A, shipping as a web API |

More LangChain and LangGraph labs are in [Agentic AI](../../06-agents/agentic-ai/).

## Next steps

- [Agentic AI](../../06-agents/agentic-ai/): LangGraph agents
- [Advanced RAG](../../05-rag-and-data/advanced-rag/)
