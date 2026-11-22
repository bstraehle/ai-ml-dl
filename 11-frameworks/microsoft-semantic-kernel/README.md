---
status: current
last_reviewed: 2026-11-22
review_every: 6mo
---

# Semantic Kernel

Microsoft's open-source SDK for adding AI to apps in Python, C# and Java: wrap prompts and code as functions, group them into plugins, add memory, and let a planner combine them.

**Roles:** AI engineer, especially in .NET and Azure shops.
**Prerequisites:** [Prompt engineering](../../04-building-with-llms/prompt-eng/)

## Key concepts

- **The kernel** is the central object that holds your AI services (models), plugins and settings, like a stove that everything cooks on ([L1](L1_student.ipynb)).
- **Semantic and native functions.** A semantic function is a prompt template; a native function is ordinary code. Both are called the same way, so they mix freely ([L2](L2_student.ipynb)).
- **Plugins** group related functions so they can be reused and shared ([L3](L3_student.ipynb), [L4](L4_student.ipynb)).
- **Memory.** Store text as embeddings and retrieve it by similarity, the building block of RAG ([L5](L5_student.ipynb)).
- **Planners** let the model pick and chain functions to reach a goal. Newer versions do this with function calling instead ([L6](L6_student.ipynb)).
- **From notebook to app.** The Chat Copilot sample connects these pieces in a full web app ([L7](L7_student.ipynb)).
- **Where it's heading.** Microsoft's newer Agent Framework builds on Semantic Kernel and AutoGen for agents; my [multi-agent Spaces](../../projects/hugging-face/) use it.

## Resources

| Resource | Type | Level | Cost |
|---|---|---|---|
| [How business thinkers can start building AI plugins with Semantic Kernel](https://www.deeplearning.ai/short-courses/microsoft-semantic-kernel/) | Short course | Beginner | Free when reviewed |
| [Semantic Kernel documentation](https://learn.microsoft.com/en-us/semantic-kernel/overview/) | Docs | Beginner | Free |
| [Semantic Kernel on GitHub](https://github.com/microsoft/semantic-kernel/) | Code | Intermediate | Free |

<details>
<summary>More</summary>

- [Chat Copilot](https://github.com/microsoft/chat-copilot/): a sample chat app built on Semantic Kernel
- [Microsoft Learn collection](https://learn.microsoft.com/en-us/collections/1n31t57k7k6r85/)

</details>

## Hands-on

| Lab | What it covers |
|---|---|
| [L1](L1_student.ipynb) | Setting up a kernel |
| [L2](L2_student.ipynb) | Semantic and native functions (a SWOT analysis) |
| [L3](L3_student.ipynb) | Organizing functions into plugins |
| [L4](L4_student.ipynb) | Chaining plugins to solve a business problem |
| [L5](L5_student.ipynb) | Memory with embeddings |
| [L6](L6_student.ipynb) | Planners |
| [L7](L7_student.ipynb) | Connecting to the Chat Copilot backend |
| [dotnet.ipynb](dotnet.ipynb), [Program.cs](Program.cs) | Chat completion and a summarize function in C# |

## Next steps

- [Agentic AI](../../06-agents/agentic-ai/)
- [LangChain](../langchain/) and [Haystack](../haystack/): other frameworks
