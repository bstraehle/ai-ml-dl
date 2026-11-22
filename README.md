# AI, ML & Deep Learning: learning resources, labs and projects

Courses I've worked through, the labs that came with them, my notes, and my own projects. It covers math and classic machine learning through to LLMs, RAG and agents. Pick a learning path for your role, or browse by topic.

## Learning paths by role

Each path is an ordered list of topics in this repo. Skip what you already know. Not sure which role fits? See [AI careers](12-careers/ai-careers/).

### AI engineer: building LLM apps and agents

1. [How LLMs work](02-llm-fundamentals/gen-ai-llms/), then [Prompt engineering](04-building-with-llms/prompt-eng/)
2. [Function calling](04-building-with-llms/function-calling/) and [Structured output](04-building-with-llms/llm-structured-output/)
3. [Embedding models](05-rag-and-data/embedding-models/), then [Advanced RAG](05-rag-and-data/advanced-rag/) and [Unstructured data](05-rag-and-data/unstructured-data-for-llm-apps/)
4. [Agentic AI](06-agents/agentic-ai/) and [MCP](06-agents/mcp/)
5. [LLM evals](08-evaluation-and-safety/llm-evals/), then [Quality & safety](08-evaluation-and-safety/quality-safety-llm-apps/), [Guardrails](08-evaluation-and-safety/guardrails/), [Red teaming](08-evaluation-and-safety/red-teaming-llm-apps/) and [AI security](08-evaluation-and-safety/ai-security/)
6. [LLMOps](09-ops-and-deployment/llm-ops/) and [Serving LLMs efficiently](09-ops-and-deployment/efficiently-serving-llms/)
7. Build something: see [my projects](#my-projects) for examples

### ML engineer: training, adapting and deploying models

1. [Mathematics](01-foundations/mathematics/), [Machine learning](01-foundations/machine-learning/), [Deep learning](01-foundations/deep-learning/)
2. [Attention in transformers](02-llm-fundamentals/attention-in-transformers/) and [Pretraining LLMs](02-llm-fundamentals/pretraining-llms/)
3. [Fine-tuning](03-adapting-models/finetuning-llms/), [RLHF](03-adapting-models/rlhf/), [Quantization](03-adapting-models/quantization/)
4. [ML engineering for production (MLOps)](09-ops-and-deployment/machine-learning-eng/) and [Experiment tracking with W&B](09-ops-and-deployment/wandb/)
5. [Serving LLMs efficiently](09-ops-and-deployment/efficiently-serving-llms/), [On-device AI](09-ops-and-deployment/on-device-ai/), [Federated learning](09-ops-and-deployment/federated-learning/)
6. A managed platform: [Amazon SageMaker](10-platforms/amazon-sagemaker/), [Google Vertex AI](10-platforms/google-vertex-ai/) or [Amazon Bedrock](10-platforms/amazon-bedrock/)

### Data scientist: from data to models

1. [Mathematics](01-foundations/mathematics/), then [Machine learning](01-foundations/machine-learning/)
2. [Practical data science](01-foundations/practical-data-science/) and [Kaggle competitions](01-foundations/kaggle/)
3. [Deep learning](01-foundations/deep-learning/) and [Embedding models](05-rag-and-data/embedding-models/)
4. [Experiment tracking with W&B](09-ops-and-deployment/wandb/)
5. Next: the first steps of the AI engineer path, to use LLMs in your analysis

### Product manager or business leader: what's possible and what it takes

1. [Generative AI for everyone](02-llm-fundamentals/gen-ai-for-everyone/)
2. [How LLMs work](02-llm-fundamentals/gen-ai-llms/), the first week is enough
3. The *Key concepts* sections of [Advanced RAG](05-rag-and-data/advanced-rag/), [Agentic AI](06-agents/agentic-ai/) and [MCP](06-agents/mcp/)
4. Measuring quality: [LLM evals](08-evaluation-and-safety/llm-evals/)
5. Risks, rules and costs: [AI governance](08-evaluation-and-safety/ai-governance/), [AI security](08-evaluation-and-safety/ai-security/), [Quality & safety](08-evaluation-and-safety/quality-safety-llm-apps/), [Carbon-aware computing](09-ops-and-deployment/carbon-aware-computing/)

## Browse by topic

Topics are grouped into numbered folders, roughly in learning order. Each folder has its own index.

| Area | Topics |
|---|---|
| [Foundations](01-foundations/) | [Mathematics](01-foundations/mathematics/) · [Machine learning](01-foundations/machine-learning/) · [Deep learning](01-foundations/deep-learning/) · [Practical data science](01-foundations/practical-data-science/) · [Kaggle](01-foundations/kaggle/) |
| [How LLMs work](02-llm-fundamentals/) | [Generative AI for everyone](02-llm-fundamentals/gen-ai-for-everyone/) · [Generative AI with LLMs](02-llm-fundamentals/gen-ai-llms/) · [Attention in transformers](02-llm-fundamentals/attention-in-transformers/) · [Pretraining LLMs](02-llm-fundamentals/pretraining-llms/) · [Reasoning models](02-llm-fundamentals/reasoning-models/) |
| [Adapting models](03-adapting-models/) | [Fine-tuning](03-adapting-models/finetuning-llms/) · [RLHF](03-adapting-models/rlhf/) · [Quantization](03-adapting-models/quantization/) |
| [Building with LLMs](04-building-with-llms/) | [Prompt engineering](04-building-with-llms/prompt-eng/) · [Function calling](04-building-with-llms/function-calling/) · [Structured output](04-building-with-llms/llm-structured-output/) · [Pair programming with LLMs](04-building-with-llms/pair-programming-llm/) · [Gradio](04-building-with-llms/gradio/) · [AI-powered game](04-building-with-llms/ai-powered-game/) |
| [RAG and data](05-rag-and-data/) | [Embedding models](05-rag-and-data/embedding-models/) · [Advanced RAG](05-rag-and-data/advanced-rag/) · [Unstructured data](05-rag-and-data/unstructured-data-for-llm-apps/) |
| [Agents](06-agents/) | [Agentic AI](06-agents/agentic-ai/) · [MCP](06-agents/mcp/) · [Computer use](06-agents/anthropic-computer-use/) |
| [Multimodal](07-multimodal/) | [Vision models](07-multimodal/vision-models/) · [Diffusion models](07-multimodal/diffusion-models/) |
| [Evaluation, safety and governance](08-evaluation-and-safety/) | [LLM evals](08-evaluation-and-safety/llm-evals/) · [Quality & safety](08-evaluation-and-safety/quality-safety-llm-apps/) · [Guardrails](08-evaluation-and-safety/guardrails/) · [Red teaming](08-evaluation-and-safety/red-teaming-llm-apps/) · [AI security](08-evaluation-and-safety/ai-security/) · [AI governance](08-evaluation-and-safety/ai-governance/) |
| [Ops and deployment](09-ops-and-deployment/) | [LLMOps](09-ops-and-deployment/llm-ops/) · [MLOps](09-ops-and-deployment/machine-learning-eng/) · [Serving LLMs](09-ops-and-deployment/efficiently-serving-llms/) · [On-device AI](09-ops-and-deployment/on-device-ai/) · [Federated learning](09-ops-and-deployment/federated-learning/) · [Carbon-aware computing](09-ops-and-deployment/carbon-aware-computing/) · [Weights & Biases](09-ops-and-deployment/wandb/) |
| [Platforms and model providers](10-platforms/) | [Open-source models (Hugging Face)](10-platforms/open-source-models/) · [Amazon Bedrock](10-platforms/amazon-bedrock/) · [Amazon SageMaker](10-platforms/amazon-sagemaker/) · [Google Vertex AI](10-platforms/google-vertex-ai/) · [IBM watsonx](10-platforms/ibm-watsonx/) · [OpenAI](10-platforms/openai/) · [Mistral](10-platforms/mistral/) · [Llama](10-platforms/llama/) |
| [Frameworks](11-frameworks/) | [LangChain](11-frameworks/langchain/) · [Haystack](11-frameworks/haystack/) · [Semantic Kernel](11-frameworks/microsoft-semantic-kernel/) |
| [Careers](12-careers/) | [AI careers](12-careers/ai-careers/) |
| [My projects](projects/) | [Hugging Face apps](projects/hugging-face/) · [Gen AI REST APIs](projects/rest-apis/) |

## Resources I recommend

**Courses**
- [DeepLearning.AI courses](https://www.deeplearning.ai/courses/): most topics in this repo started here
- [Weights & Biases course notebooks](https://github.com/wandb/edu): RAG, LLM apps, fine-tuning and MLOps
- [IBM Applied AI professional certificate](https://www.coursera.org/professional-certificates/applied-artifical-intelligence-ibm-watson-ai/) on Coursera
- [OpenAI Academy](https://academy.openai.com/)
- [Andrej Karpathy on YouTube](https://www.youtube.com/@AndrejKarpathy): neural networks and LLMs from scratch
- [Sebastian Raschka](https://sebastianraschka.com/): [Build a Large Language Model (From Scratch)](https://github.com/rasbt/LLMs-from-scratch) code and videos, and the [Ahead of AI](https://magazine.sebastianraschka.com/) newsletter

**Books**
- [Generative AI on AWS: Building Context-Aware Multimodal Reasoning Applications](https://github.com/generative-ai-on-aws/generative-ai-on-aws) by Chris Fregly, Antje Barth and Shelbee Eigenbrode (O'Reilly), with code and notebooks for every chapter
- [Conversational AI: Chatbots that work](https://www.manning.com/books/conversational-ai) by Andrew R. Freed (Manning)

**Notebooks**
- [Hugging Face Open-Source AI Cookbook](https://huggingface.co/learn/cookbook/index/)

## More places to learn

Free or low-cost learning from AI companies, course platforms and universities. I haven't taken all of these, but they're widely used and well regarded.

**From AI companies**
- [Anthropic Academy](https://academy.claude.com/): courses on building with Claude, agents and MCP
- [Microsoft: Generative AI for Beginners](https://github.com/microsoft/generative-ai-for-beginners), [AI Agents for Beginners](https://github.com/microsoft/ai-agents-for-beginners) and [ML for Beginners](https://github.com/microsoft/ML-For-Beginners): free lesson-by-lesson courses on GitHub
- [NVIDIA Deep Learning Institute](https://www.nvidia.com/en-us/training/): self-paced and instructor-led courses on deep learning, generative AI and GPUs
- [Google Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course) and [Google Skills](https://www.skills.google/) (Google Cloud training)
- [AWS Skill Builder](https://skillbuilder.aws/) and [IBM SkillsBuild](https://skillsbuild.org/)
- [Hugging Face Learn](https://huggingface.co/learn): courses on LLMs, agents, MCP, diffusion and more

**Course platforms and practice**
- [Coursera: DeepLearning.AI](https://www.coursera.org/partners/deeplearning-ai): the longer specializations, with certificates
- [Kaggle](https://www.kaggle.com/learn): free micro-courses, datasets and competitions to practice on
- [fast.ai](https://course.fast.ai/): practical deep learning, top down

**University courses**
- [Stanford CS336: Language Modeling from Scratch](https://stanford-cs336.github.io/)
- [Stanford CS231n: Deep Learning for Computer Vision](https://cs231n.stanford.edu/)
- [MIT 6.S191: Introduction to Deep Learning](https://introtodeeplearning.com/)

**Staying current**
- [The Batch](https://www.deeplearning.ai/the-batch/) (DeepLearning.AI): weekly AI news
- [Ahead of AI](https://magazine.sebastianraschka.com/) (Sebastian Raschka): research explained

## My projects

- [Hugging Face apps](projects/hugging-face/): source code for my 12 [Hugging Face Spaces](https://hf.co/bstraehle), mirrored from the Hub: a GAIA benchmark agent with MCP, agentic and advanced RAG, multi-agent deep research, chess and coding (crewAI, LangGraph, Microsoft Agent Framework), a Stockfish MCP server, and SDK demos for Amazon Bedrock, OpenAI and Vertex AI
- [Gen AI REST APIs](projects/rest-apis/): Postman collections for 51 generative AI providers, exported from my public [Generative AI APIs workspace](https://www.postman.com/ai-engineer/generative-ai-apis/)

## How this repo stays current

A [monthly maintenance issue](https://github.com/bstraehle/ai-ml-dl/issues?q=label%3Amaintenance) lists broken links and topics that are due for review. Each topic README in the new format records when it was last reviewed. Spotted something outdated, or know a better resource? Please [open an issue](https://github.com/bstraehle/ai-ml-dl/issues/new).
