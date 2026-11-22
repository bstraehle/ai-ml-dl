---
status: current
last_reviewed: 2026-11-22
review_every: 6mo
---

# Red teaming LLM apps

Red teaming is attacking your own LLM app before others do: trying to make it leak data, ignore its instructions, say harmful things or go down, then fixing what you find.

**Roles:** AI engineer, security engineer. Product managers: read *Key concepts*.
**Prerequisites:** [Quality & safety](../quality-safety-llm-apps/)

## Key concepts

- **What can go wrong:** bias and stereotypes, sensitive information disclosure, service disruption (very long or looping outputs), hallucinations, and harmful content ([L1](L1_Overview_of_LLM_vulnerabilities.ipynb)).
- **Classic attacks:** exploiting text completion, biased or leading prompts, direct prompt injection ("ignore previous instructions"), and gray-box attacks that use knowledge of the system prompt ([L2](L2_Red_teaming_LLMs.ipynb)).
- **Scale it.** Manual probing finds the first issues. Libraries of known attack prompts and automated scanners (Giskard) cover far more ground ([L3](L3_Red_teaming_at_scale.ipynb)).
- **Use LLMs against LLMs.** One model generates adversarial inputs for your app's domain, and another judges whether the responses broke the rules ([L4](L4_Red_teaming_LLMs_with_LLMs.ipynb)).
- **Make it a process.** Run a full assessment in rounds, fix, and re-test; keep the attacks as a regression suite that runs before every release ([L5](L5_A_full_red_teaming_assessment.ipynb)).
- **Agents raise the stakes.** Once an app can call tools, a successful injection can take actions, not just produce text. Test indirect injection through documents, web pages and tool results too; see [AI security](../ai-security/).

## Resources

| Resource | Type | Level | Cost |
|---|---|---|---|
| [Red teaming LLM applications](https://www.deeplearning.ai/short-courses/red-teaming-llm-applications/) (Giskard) | Short course | Beginner | Free when reviewed |
| [Promptfoo red teaming guide](https://www.promptfoo.dev/docs/red-team/) | Docs, tool | Intermediate | Free |
| [PyRIT](https://github.com/Azure/PyRIT) (Microsoft) | Tool | Advanced | Free |
| [Giskard](https://github.com/Giskard-AI/giskard-oss) | Tool | Intermediate | Free |

## Hands-on

| Lab | What it covers |
|---|---|
| [L1](L1_Overview_of_LLM_vulnerabilities.ipynb) | Bias, information disclosure, service disruption, hallucinations |
| [L2](L2_Red_teaming_LLMs.ipynb) | Bypassing safeguards: completion tricks, biased prompts, prompt injection |
| [L3](L3_Red_teaming_at_scale.ipynb) | Manual testing, prompt libraries and Giskard's LLM scan |
| [L4](L4_Red_teaming_LLMs_with_LLMs.ipynb) | Generating and evaluating adversarial inputs with LLMs |
| [L5](L5_A_full_red_teaming_assessment.ipynb) | A full red-teaming assessment in rounds |

## Next steps

- [AI security](../ai-security/): the wider threat picture, including agents
- [Guardrails](../guardrails/): defenses to put in place
