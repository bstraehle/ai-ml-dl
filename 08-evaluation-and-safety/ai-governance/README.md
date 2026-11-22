---
status: current
last_reviewed: 2026-11-22
review_every: 6mo
---

# AI governance

How organizations decide which AI systems to build or buy, assess and document their risks, keep humans in control, and comply with regulation. Governance is an ongoing process, not a one-time sign-off.

**Roles:** product manager, business leader, legal and compliance, AI engineer.
**Prerequisites:** [Generative AI for everyone](../../02-llm-fundamentals/gen-ai-for-everyone/)

## Key concepts

- **Risk-based by default.** Classify each use case by its potential for harm. An internal writing assistant needs light controls. Systems that affect hiring, credit, health, safety or legal rights need much more: testing, documentation, human oversight and monitoring.
- **Know what you have.** Keep an inventory of AI systems in use (built and bought), with an owner, purpose, data used and risk level for each.
- **Frameworks give the structure:**
  - The NIST AI Risk Management Framework organizes the work into *Govern, Map, Measure* and *Manage*.
  - ISO/IEC 42001 is a certifiable standard for an AI management system, in the style of ISO 27001 for security.
  - The OECD AI Principles are the internationally agreed values most frameworks build on.
- **Regulation is arriving.** The EU AI Act sorts AI systems into risk tiers (prohibited practices, high-risk systems, transparency obligations, minimal risk) and adds rules for general-purpose AI models. It entered into force in 2024 and applies in stages, including to companies outside the EU that serve EU users. Other jurisdictions are moving too, so check the current status for your markets.
- **Practical artifacts:** model and system cards, data provenance records, impact assessments, evaluation results ([LLM evals](../llm-evals/)), human oversight procedures, an incident response plan, and vendor assessments for third-party models.
- **It's cross-functional.** Product, engineering, security, legal and the business share the decisions. Engineers supply the evidence: evals, red-team results and monitoring.

## Resources

| Resource | Type | Level | Cost |
|---|---|---|---|
| [NIST AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework) | Framework | All levels | Free |
| [EU AI Act explorer](https://artificialintelligenceact.eu/) (Future of Life Institute) | Reference | All levels | Free |
| [OECD AI Principles](https://oecd.ai/en/ai-principles) | Principles | Beginner | Free |

## Hands-on

There are no labs for this topic. To produce the evidence governance needs, see [LLM evals](../llm-evals/), [Red teaming](../red-teaming-llm-apps/) and [Quality & safety](../quality-safety-llm-apps/).

## Next steps

- [AI security](../ai-security/)
- [Carbon-aware computing](../../09-ops-and-deployment/carbon-aware-computing/): the environmental side of responsible AI
