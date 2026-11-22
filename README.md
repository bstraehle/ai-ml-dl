# AI, ML & Deep Learning: learning resources, labs and projects

Courses I've worked through, the labs that came with them, my notes, and my own projects. It covers math and classic machine learning through to LLMs, RAG and agents. Pick a learning path for your role, or browse by topic.

> **Being reorganized.** Topics whose README starts with a review date (for example [Agentic AI](agentic-ai/), [Advanced RAG](advanced-rag/) and [MCP](mcp/)) are in the new format: key concepts, top resources, hands-on labs and next steps. The rest are still plain link lists and are being updated.

## Learning paths by role

Each path is an ordered list of topics in this repo. Skip what you already know.

### AI engineer: building LLM apps and agents

1. [How LLMs work](gen-ai-llms/), then [Prompt engineering](openai/)
2. [Function calling](function-calling/) and [Structured output](llm-structured-output/)
3. [Embedding models](embedding-models/), then [Advanced RAG](advanced-rag/) and [Unstructured data](unstructured-data-for-llm-apps/)
4. [Agentic AI](agentic-ai/) and [MCP](mcp/)
5. [Quality & safety](quality-safety-llm-apps/), [Guardrails](guardrails/), [Red teaming](red-teaming-llm-apps/)
6. [LLMOps](llm-ops/) and [Serving LLMs efficiently](efficiently-serving-llms/)
7. Build something: see [my projects](#my-projects) for examples

### ML engineer: training, adapting and deploying models

1. [Mathematics](mathematics/), [Machine learning](machine-learning/), [Deep learning](deep-learning/)
2. [Attention in transformers](attention-in-transformers/) and [Pretraining LLMs](pretraining-llms/)
3. [Fine-tuning](finetuning-llms/), [RLHF](rlhf/), [Quantization](quantization/)
4. [ML engineering for production (MLOps)](machine-learning-eng/) and [Experiment tracking with W&B](wandb/)
5. [Serving LLMs efficiently](efficiently-serving-llms/), [On-device AI](on-device-ai/), [Federated learning](federated-learning/)
6. A managed platform: [Amazon SageMaker](amazon-sagemaker/), [Google Vertex AI](google-vertex-ai/) or [Amazon Bedrock](amazon-bedrock/)

### Data scientist: from data to models

1. [Mathematics](mathematics/), then [Machine learning](machine-learning/)
2. [Practical data science](practical-data-science/) and [Kaggle competitions](kaggle/)
3. [Deep learning](deep-learning/) and [Embedding models](embedding-models/)
4. [Experiment tracking with W&B](wandb/)
5. Next: the first steps of the AI engineer path, to use LLMs in your analysis

### Product manager or business leader: what's possible and what it takes

1. [Generative AI for everyone](gen-ai-for-everyone/)
2. [How LLMs work](gen-ai-llms/), the first week is enough
3. The *Key concepts* sections of [Advanced RAG](advanced-rag/), [Agentic AI](agentic-ai/) and [MCP](mcp/)
4. Risks and costs: [Quality & safety](quality-safety-llm-apps/), [Red teaming](red-teaming-llm-apps/), [Carbon-aware computing](carbon-aware-computing/)

## Browse by topic

| Area | Topics |
|---|---|
| Foundations | [Mathematics](mathematics/) · [Machine learning](machine-learning/) · [Deep learning](deep-learning/) · [Practical data science](practical-data-science/) · [Kaggle](kaggle/) |
| How LLMs work | [Generative AI for everyone](gen-ai-for-everyone/) · [Generative AI with LLMs](gen-ai-llms/) · [Attention in transformers](attention-in-transformers/) · [Pretraining LLMs](pretraining-llms/) |
| Adapting models | [Fine-tuning](finetuning-llms/) · [RLHF](rlhf/) · [Quantization](quantization/) |
| Building with LLMs | [Prompt engineering](prompt-eng/) · [Function calling](function-calling/) · [Structured output](llm-structured-output/) · [Pair programming with LLMs](pair-programming-llm/) · [Gradio](gradio/) · [AI-powered game](ai-powered-game/) |
| RAG and data | [Embedding models](embedding-models/) · [Advanced RAG](advanced-rag/) · [Unstructured data](unstructured-data-for-llm-apps/) |
| Agents | [Agentic AI](agentic-ai/) · [MCP](mcp/) · [Computer use](anthropic-computer-use/) |
| Multimodal | [Vision models](vision-models/) · [Diffusion models](diffusion-models/) |
| Evaluation and safety | [Quality & safety](quality-safety-llm-apps/) · [Guardrails](guardrails/) · [Red teaming](red-teaming-llm-apps/) |
| Ops and deployment | [LLMOps](llm-ops/) · [MLOps](machine-learning-eng/) · [Serving LLMs](efficiently-serving-llms/) · [On-device AI](on-device-ai/) · [Federated learning](federated-learning/) · [Carbon-aware computing](carbon-aware-computing/) · [Weights & Biases](wandb/) |
| Platforms and model providers | [Amazon Bedrock](amazon-bedrock/) · [Amazon SageMaker](amazon-sagemaker/) · [Google Vertex AI](google-vertex-ai/) · [IBM watsonx](ibm-watsonx/) · [OpenAI](openai/) · [Mistral](mistral/) · [Llama](llama/) |
| Frameworks | [LangChain](langchain/) · [Haystack](haystack/) · [Semantic Kernel](microsoft-semantic-kernel/) |
| My projects | [Hugging Face apps](hugging-face/) · [Gen AI REST APIs](rest-apis/) |

## Resources I recommend

**Courses**
- [DeepLearning.AI courses](https://www.deeplearning.ai/courses/): most topics in this repo started here
- [Weights & Biases course notebooks](https://github.com/wandb/edu): RAG, LLM apps, fine-tuning and MLOps
- [IBM Applied AI professional certificate](https://www.coursera.org/professional-certificates/applied-artifical-intelligence-ibm-watson-ai/) on Coursera
- [OpenAI Academy](https://academy.openai.com/)
- [Andrej Karpathy on YouTube](https://www.youtube.com/@AndrejKarpathy): neural networks and LLMs from scratch

**Books**
- [Generative AI on AWS: Building Context-Aware Multimodal Reasoning Applications](https://github.com/generative-ai-on-aws/generative-ai-on-aws) by Chris Fregly, Antje Barth and Shelbee Eigenbrode (O'Reilly), with code and notebooks for every chapter
- [Conversational AI: Chatbots that work](https://www.manning.com/books/conversational-ai) by Andrew R. Freed (Manning)

**Notebooks and lists**
- [Hugging Face Open-Source AI Cookbook](https://huggingface.co/learn/cookbook/index/)
- [Awesome Generative AI Guide: free GenAI courses](https://github.com/aishwaryanr/awesome-generative-ai-guide?tab=readme-ov-file#book-list-of-free-genai-courses)

## My projects

- [Hugging Face apps](hugging-face/): source code for my [Hugging Face Spaces](https://hf.co/bstraehle): RAG, agentic RAG, multi-agent systems (AutoGen, crewAI, LangGraph), a GAIA benchmark agent, and fine-tuning
- [Gen AI REST APIs](rest-apis/): a [Postman workspace](https://www.postman.com/bstraehle/workspace/generative-ai-llm-rest-apis/overview) for calling LLM provider APIs directly
- [Kaggle](https://www.kaggle.com/bstraehle): tabular classification and regression competitions

## How this repo stays current

A [monthly maintenance issue](https://github.com/bstraehle/ai-ml-dl/issues?q=label%3Amaintenance) lists broken links and topics that are due for review. Each topic README in the new format records when it was last reviewed. Spotted something outdated, or know a better resource? Please [open an issue](https://github.com/bstraehle/ai-ml-dl/issues/new).
