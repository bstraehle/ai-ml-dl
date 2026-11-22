---
status: current
last_reviewed: 2026-11-21
review_every: 3mo
---

# Agentic AI

An agent is an LLM application where the model decides what happens next: which tool to call, whether to keep going, or which agent to hand off to. This folder has the agent courses I took, their labs, and my own multi-agent projects.

**Roles:** AI engineer, ML engineer. Product managers: read *Key concepts*.
**Prerequisites:** [Function calling](../../04-building-with-llms/function-calling/), [Advanced RAG](../../05-rag-and-data/advanced-rag/)

## Key concepts

- **Agency is a spectrum.** Use the lowest level that solves the problem. Each step up adds flexibility, but also cost, latency and ways to fail.

  | Level | The LLM's output... | Example |
  |---|---|---|
  | Simple processor | has no effect on program flow | `process(llm_response)` |
  | Router | picks a branch | `if llm_decision(): a() else: b()` |
  | Tool call | picks a function and its arguments | `run(llm_tool, llm_args)` |
  | Multi-step agent | decides whether to keep looping | `while llm_should_continue(): step()` |
  | Multi-agent | starts other agents | `if llm_trigger(): run_agent()` |

  (From the Hugging Face smolagents course; slides: [agency levels](agency-levels.png), [the road to more agency](agency-levels-2.png).)
- **Workflows before agents.** Workflows follow code paths you define; agents choose their own. Anthropic's common workflow patterns are prompt chaining, routing, parallelization, orchestrator-workers and evaluator-optimizer. Reach for a free-running agent only when the steps can't be known in advance.
- **Four design patterns:** reflection, tool use, planning and multi-agent collaboration. The [AutoGen labs](autogen/) cover each one.
- **Code agents** write and run code as their actions, instead of emitting JSON tool calls. That composes better and can create new tools on the fly ([smolagents labs](hugging-face/)). It also needs a sandbox.
- **Memory.** Short-term memory is the conversation and run state. Long-term memory is semantic (facts), episodic (past experiences) or procedural (learned instructions); see the [LangMem](langmem/) labs. Letta treats the context window like an operating system's RAM and pages memory in and out ([Letta labs](letta/)).
- **Context engineering** decides what goes into the context window at each step: instructions, tool definitions, retrieved documents, memory and history. It often matters more than the wording of the prompt.
- **Protocols:** [MCP](../mcp/) connects agents to tools and data; agent-to-agent protocols such as ACP connect agents to each other ([ACP labs](acp/)).
- **Evaluate the path, not only the answer.** Trace every run. Score router decisions, individual skills and the whole trajectory, as well as the final output ([evaluation labs](evaluating-ai-agents/)).

## Resources

Top picks (DeepLearning.AI short courses were free to enroll when this page was reviewed):

| Resource | Type | Level | Cost |
|---|---|---|---|
| [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents/) (Anthropic) | Article | Beginner | Free |
| [AI Agents in LangGraph](https://www.deeplearning.ai/short-courses/ai-agents-in-langgraph/) | Short course | Intermediate | Free |
| [Building code agents with Hugging Face smolagents](https://www.deeplearning.ai/short-courses/building-code-agents-with-hugging-face-smolagents/) | Short course | Intermediate | Free |
| [Long-term agentic memory with LangGraph](https://www.deeplearning.ai/short-courses/long-term-agentic-memory-with-langgraph/) | Short course | Intermediate | Free |
| [Evaluating AI agents](https://www.deeplearning.ai/short-courses/evaluating-ai-agents/) | Short course | Intermediate | Free |

<details>
<summary>More courses and reading</summary>

**Frameworks**
- AutoGen: [AI agentic design patterns with AutoGen](https://www.deeplearning.ai/short-courses/ai-agentic-design-patterns-with-autogen/) · [GitHub](https://github.com/microsoft/autogen/)
- crewAI: [Multi AI agent systems](https://www.deeplearning.ai/short-courses/multi-ai-agent-systems-with-crewai/) · [Practical multi AI agents and advanced use cases](https://www.deeplearning.ai/short-courses/practical-multi-ai-agents-and-advanced-use-cases-with-crewai/)
- DSPy: [Build and optimize agentic apps](https://www.deeplearning.ai/short-courses/dspy-build-optimize-agentic-apps/)
- LangChain and LangGraph: [Building your own database agent](https://www.deeplearning.ai/short-courses/building-your-own-database-agent/) · [LangGraph (Python) playlist](https://www.youtube.com/playlist?list=PLfaIDFEXuae16n2TWUkKq5PgJ0w6Pkwtg)
- Letta: [LLMs as operating systems: agent memory](https://www.deeplearning.ai/short-courses/llms-as-operating-systems-agent-memory/)
- LlamaIndex: [Building agentic RAG](https://www.deeplearning.ai/short-courses/building-agentic-rag-with-llamaindex/) · [Event-driven agentic document workflows](https://www.deeplearning.ai/short-courses/event-driven-agentic-document-workflows/)
- Amazon Bedrock: [Serverless agentic workflows](https://www.deeplearning.ai/short-courses/serverless-agentic-workflows-with-amazon-bedrock/)

**Protocols**
- [ACP: agent communication protocol](https://www.deeplearning.ai/short-courses/acp-agent-communication-protocol/) (IBM)

**Agent applications**
- [Building AI browser agents](https://www.deeplearning.ai/short-courses/building-ai-browser-agents/)
- [Building AI voice agents for production](https://www.deeplearning.ai/short-courses/building-ai-voice-agents-for-production/)
- [Build apps with Windsurf's AI coding agents](https://www.deeplearning.ai/short-courses/build-apps-with-windsurfs-ai-coding-agents/)
- [Vibe coding 101 with Replit](https://www.deeplearning.ai/short-courses/vibe-coding-101-with-replit/)

**Reading**
- [Context Engineering Guide](https://docs.google.com/document/d/1JU8w-E4LlseFZm-ag22GSBU5A2rp2nb7iFGBNAbFL7k/) (Google Doc)

</details>

## Hands-on

| Folder | What the labs build |
|---|---|
| [autogen/](autogen/) | Multi-agent conversation, sequential chats, reflection, tool use (chess), coding, planning |
| [crew-ai/beginner/](crew-ai/beginner/) | Crews for research and writing, customer support, outreach, event planning, financial analysis |
| [crew-ai/advanced/](crew-ai/advanced/) | Project planning, sales pipeline, support analytics, content at scale, crews for production |
| [langgraph/](langgraph/) | A ReAct agent from scratch, LangGraph components, agentic search, persistence, human in the loop, an essay writer |
| [langchain/](langchain/) | Agents over CSV files and SQL databases, function calling, the Assistants API |
| [llamaindex/agentic-rag/](llamaindex/agentic-rag/) | Router engine, tool calling, reasoning loop, multi-document agent |
| [llamaindex/event-driven-agentic-workflows/](llamaindex/event-driven-agentic-workflows/) | Event-driven workflows with RAG, form parsing, human in the loop, voice |
| [hugging-face/](hugging-face/) | smolagents code agents, a sandboxed interpreter, tracing, search tools |
| [dspy/](dspy/) | Signatures and modules, MLflow tracing, optimizing an agent with DSPy |
| [langmem/](langmem/) | An email assistant with semantic, then episodic, then procedural memory |
| [letta/](letta/) | Self-editing memory from scratch, agentic RAG with external memory, multi-agent orchestration |
| [acp/](acp/) | ACP servers and clients, sequential and router chaining, adding MCP |
| [evaluating-ai-agents/](evaluating-ai-agents/) | Tracing, router and skill evaluations, trajectory evaluations |
| [realavatar/](realavatar/) | Voice agent components and latency optimization |

## My projects

Each links to the live Space and its source in [hugging-face/](../../projects/hugging-face/).

- [General AI Assistant](https://huggingface.co/spaces/bstraehle/gaia): a multi-agent system with MCP client and server that solves GAIA benchmark and Humanity's Last Exam problems ([source](../../projects/hugging-face/gaia/))
- [Agentic AI](https://huggingface.co/spaces/bstraehle/agentic-ai): agentic RAG with LangChain and LlamaIndex ([source](../../projects/hugging-face/agentic-ai/))
- Deep research: [crewAI](https://huggingface.co/spaces/bstraehle/multi-agent-ai-crewai-deep-research) ([source](../../projects/hugging-face/multi-agent-ai-crewai-deep-research/)) · [LangGraph](https://huggingface.co/spaces/bstraehle/multi-agent-ai-langgraph-deep-research) ([source](../../projects/hugging-face/multi-agent-ai-langgraph-deep-research/))
- Chess: [LangGraph](https://huggingface.co/spaces/bstraehle/multi-agent-ai-langgraph-chess) ([source](../../projects/hugging-face/multi-agent-ai-langgraph-chess/)) · [Microsoft Agent Framework](https://huggingface.co/spaces/bstraehle/multi-agent-ai-microsoft-chess) ([source](../../projects/hugging-face/multi-agent-ai-microsoft-chess/))
- Coding: [Microsoft Agent Framework](https://huggingface.co/spaces/bstraehle/multi-agent-ai-microsoft-coding) ([source](../../projects/hugging-face/multi-agent-ai-microsoft-coding/))

## Next steps

- [MCP](../mcp/): connect agents to tools and data in a standard way
- [Quality & safety](../../08-evaluation-and-safety/quality-safety-llm-apps/) and [Red teaming](../../08-evaluation-and-safety/red-teaming-llm-apps/): make agents trustworthy
- [LLMOps](../../09-ops-and-deployment/llm-ops/): run them in production
