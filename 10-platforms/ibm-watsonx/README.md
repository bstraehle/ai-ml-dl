---
status: current
last_reviewed: 2026-11-22
review_every: 12mo
---

# IBM watsonx

IBM's AI platform: watsonx.ai for models and development, and watsonx Assistant for chatbots, which IBM now offers within watsonx Orchestrate. This folder has the chatbots and the Flask app I built in IBM's applied AI courses.

**Roles:** AI engineer, solutions architect.
**Prerequisites:** basic Python and web development

## Key concepts

- **Conversational design.** A chatbot is built from intents (what the user wants), entities (details such as dates or products) and dialog or actions (how the bot responds and what it asks next).
- **Actions vs dialog.** Newer watsonx Assistant bots use actions (step-by-step flows), older ones use a dialog tree; the flower-shop bot has both.
- **Integrate with a backend.** A bot can call your own web service for data it doesn't have, as the student-advisor bot does with a small Flask server.
- **Wrap AI behind an API.** A web app can call an NLP model (here, emotion detection) through a clean Python package, with unit tests and linting.
- **Platform names change.** IBM's assistant product now redirects to watsonx Orchestrate; the concepts above still apply.

## Resources

| Resource | Type | Level | Cost |
|---|---|---|---|
| [Building AI powered chatbots without programming](https://www.coursera.org/learn/building-ai-powered-chatbots/) (IBM) | Course | Beginner | Free to audit when reviewed |
| [IBM Applied AI professional certificate](https://www.coursera.org/professional-certificates/applied-artifical-intelligence-ibm-watson-ai/) | Certificate | Beginner | Paid |
| [IBM watsonx.ai](https://www.ibm.com/products/watsonx-ai) | Product | Intermediate | Free trial |

## Hands-on

| Folder | What's inside |
|---|---|
| [chatbots/](chatbots/) | Banking, flower-shop and student-advisor bots (exported skills and web pages) |
| [chatbots/student-advisor/](chatbots/student-advisor/) | A Flask backend the bot calls |
| [emotion-detection-web-app/](emotion-detection-web-app/) | A Flask app around an emotion detection package, with tests |

The [conversational AI magic quadrant image](magic-quadrant-conversational-ai.png) shows the vendor landscape at the time I took the course. My Postman collection for watsonx.ai is in [projects/rest-apis](../../projects/rest-apis/).

## Next steps

- [Prompt engineering](../../04-building-with-llms/prompt-eng/)
- [Agentic AI](../../06-agents/agentic-ai/): chatbots that use tools
