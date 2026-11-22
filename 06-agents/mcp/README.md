---
status: current
last_reviewed: 2026-11-21
review_every: 3mo
---

# Model Context Protocol (MCP)

MCP is an open protocol for connecting LLM applications to tools and data. You write an integration once as an MCP server, and any MCP-capable app (a chat app, IDE or agent) can use it. Anthropic introduced it in November 2024, and it is now supported across most major AI tools.

**Roles:** AI engineer. Product managers and architects: read *Key concepts*.
**Prerequisites:** [Function calling](../../04-building-with-llms/function-calling/)

## Key concepts

- **Hosts, clients, servers.** The *host* is the LLM application. It runs one *client* per connection, and each client talks to a *server* that provides capabilities. Messages are JSON-RPC 2.0.
- **What servers offer:** *tools* (functions the model can call), *resources* (data the user or model can read) and *prompts* (reusable templates the user picks).
- **Transports:** *stdio* for local servers launched by the host, and *Streamable HTTP* for remote servers. The course's remote-server lesson also shows the older SSE transport.
- **MCP vs function calling.** Function calling is how one model calls one app's functions. MCP standardizes discovering and sharing those functions, so tools are reusable across apps and models.
- **Security.** Tools can run arbitrary code, and tool descriptions from untrusted servers can carry prompt injections. Only connect servers you trust, ask the user before calling tools, and give servers the least access they need.
- **Agent-to-agent is a different layer.** MCP connects an agent to tools and data. Protocols like ACP connect agents to each other; see [Agentic AI](../agentic-ai/).

## Resources

| Resource | Type | Level | Cost |
|---|---|---|---|
| [MCP: Build rich-context AI apps with Anthropic](https://www.deeplearning.ai/short-courses/mcp-build-rich-context-ai-apps-with-anthropic/) | Short course | Beginner | Free when reviewed |
| [MCP documentation](https://modelcontextprotocol.io/) | Docs | All levels | Free |
| [MCP specification (latest)](https://modelcontextprotocol.io/specification/latest) | Spec | Advanced | Free |
| [MCP reference servers](https://github.com/modelcontextprotocol/servers) | Code | Intermediate | Free |
| [Hugging Face MCP course](https://huggingface.co/learn/mcp-course) | Course | Beginner | Free |

## Hands-on

| Lab | What it builds |
|---|---|
| [L3](L3.ipynb) | A chatbot with tools, the starting point before MCP |
| [L4](L4.ipynb) | An MCP server that exposes the chatbot's tools |
| [L5](L5.ipynb) | An MCP client that connects the chatbot to the server |
| [L6](L6.ipynb) | Connecting the chatbot to reference servers (for example filesystem and fetch) |
| [L7](L7.ipynb) | Adding prompt and resource features |
| [L9](L9.ipynb) | Creating and deploying a remote server |

Related: [agentic-ai/acp/L9](../agentic-ai/acp/L9.ipynb) adds MCP to an agent that's served over ACP.

## My projects

- [Stockfish chess engine MCP server](https://huggingface.co/spaces/bstraehle/chess): evaluates chess positions ([source](../../projects/hugging-face/chess/))
- [General AI Assistant](https://huggingface.co/spaces/bstraehle/gaia): an MCP client and server for multi-agent problem solving ([source](../../projects/hugging-face/gaia/))

## Next steps

- [Agentic AI](../agentic-ai/): agents that use MCP tools
- [Red teaming](../../08-evaluation-and-safety/red-teaming-llm-apps/): test tool-using apps for prompt injection
