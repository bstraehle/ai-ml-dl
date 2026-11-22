---
status: current
last_reviewed: 2026-11-22
review_every: 3mo
---

# AI security

Securing applications built on LLMs and agents. The core problem is new: models follow instructions found in any text they read, so attackers can steer them through documents, web pages, emails and tool results, not just through the chat box.

**Roles:** AI engineer, security engineer, architect. Leaders: read *Key concepts*.
**Prerequisites:** [Function calling](../../04-building-with-llms/function-calling/), [Red teaming](../red-teaming-llm-apps/)

## Key concepts

- **Prompt injection is the central risk.** *Direct* injection comes from the user. *Indirect* injection hides instructions in content the app retrieves or a tool returns. There's no complete fix today, so design for it rather than hoping a filter catches everything.
- **Watch the dangerous combination.** An agent that has access to private data, reads untrusted content, and can send data out (email, web requests, links) can be tricked into leaking that data. Simon Willison calls this the "lethal trifecta". Remove at least one leg.
- **Defense in depth:**
  - Give tools the least privilege they need, and scope credentials per user.
  - Keep trusted instructions separate from untrusted content, and treat model output as untrusted input to the next step.
  - Require human approval for consequential actions: payments, deletions, outgoing messages.
  - Don't put secrets in prompts, and assume system prompts will leak.
  - Log and monitor tool calls, and rate-limit them.
- **Agents add new risks:** excessive agency, tool misuse, memory poisoning, and untrusted MCP servers whose tool descriptions can carry instructions ([MCP](../../06-agents/mcp/)).
- **RAG needs access control.** Retrieve only documents the current user is allowed to see, or the model will happily summarize them for anyone.
- **The supply chain matters too:** model files that run code when loaded (such as pickle), poisoned training or fine-tuning data, and ordinary dependency vulnerabilities.
- **Use shared frameworks** to structure threat modeling: the OWASP Top 10 for LLM applications, OWASP's agentic AI threats, MITRE ATLAS, and Google's Secure AI Framework (SAIF).

## Resources

| Resource | Type | Level | Cost |
|---|---|---|---|
| [OWASP Top 10 for LLM applications](https://genai.owasp.org/llm-top-10/) | Reference | All levels | Free |
| [OWASP agentic AI threats and mitigations](https://genai.owasp.org/resource/agentic-ai-threats-and-mitigations/) | Reference | Intermediate | Free |
| [Prompt injection series](https://simonwillison.net/series/prompt-injection/) (Simon Willison) | Articles | All levels | Free |
| [MITRE ATLAS](https://atlas.mitre.org/) | Knowledge base | Advanced | Free |
| [Secure AI Framework (SAIF)](https://saif.google/) (Google) | Framework | Leaders | Free |

## Hands-on

- [Red teaming labs](../red-teaming-llm-apps/): prompt injection, attack libraries, automated scans
- [Guardrails labs](../guardrails/): input and output validation, PII redaction
- [Refusals and prompt injections lab](../quality-safety-llm-apps/L4_Refusals_and_prompt_injections.ipynb): detecting jailbreaks and injections

## Next steps

- [AI governance](../ai-governance/): policies, risk management and regulation
- [Agentic AI](../../06-agents/agentic-ai/)
