---
status: current
last_reviewed: 2026-11-22
review_every: 6mo
---

# Google Vertex AI

Google Cloud's platform for building with Google's models (Gemini, embeddings) and with your own: text generation, embeddings, semantic search and multimodal prompting.

**Roles:** AI engineer, ML engineer.
**Prerequisites:** a Google Cloud project, basic Python

> Vertex AI's documentation now redirects to Google's Gemini Enterprise Agent Platform.

## Key concepts

- **Embeddings capture meaning.** Similar texts get nearby vectors, which you can visualize and cluster ([L1](L1-embeddings-api-intro.ipynb), [L3](L3-visualizing-embeddings.ipynb)).
- **Applications of embeddings:** clustering, anomaly detection and classification, shown on Stack Overflow questions ([L4](L4-applications-of-embeddings.ipynb)).
- **Text generation parameters.** Temperature, top-k and top-p control how varied the output is ([L5](L5-text-generation.ipynb)).
- **Semantic search Q&A.** Embed documents, find the nearest ones to a question with approximate nearest-neighbor search, and generate an answer from them ([L6](L6-semantic-search.ipynb)).
- **Multimodal prompting.** Gemini models take text, images and video together.
- **Project setup.** Vertex AI needs a project, billing and enabled APIs ([L7](L7-optional.ipynb)).

## Resources

| Resource | Type | Level | Cost |
|---|---|---|---|
| [Understanding and applying text embeddings](https://www.deeplearning.ai/short-courses/google-cloud-vertex-ai/) (Google Cloud) | Short course | Beginner | Free when reviewed |
| [Large multimodal model prompting with Gemini](https://www.deeplearning.ai/short-courses/large-multimodal-model-prompting-with-gemini/) (Google Cloud) | Short course | Beginner | Free when reviewed |
| [Gemini Enterprise Agent Platform documentation](https://docs.cloud.google.com/gemini-enterprise-agent-platform) | Docs | Intermediate | Free |
| [Gemini API documentation](https://aistudio.google.com/docs) | Docs | Beginner | Free |

## Hands-on

| Lab | What it covers |
|---|---|
| [L1](L1-embeddings-api-intro.ipynb) | Getting started with text embeddings |
| [L3](L3-visualizing-embeddings.ipynb) | Visualizing embeddings |
| [L4](L4-applications-of-embeddings.ipynb) | Clustering, anomaly detection and classification with embeddings |
| [L5](L5-text-generation.ipynb) | Text generation and its parameters |
| [L6](L6-semantic-search.ipynb) | A semantic search Q&A system |
| [L7](L7-optional.ipynb) | Google Cloud setup |

My [Google Vertex AI Space](../../projects/hugging-face/vertex-ai/) and the Gemini collection in [projects/rest-apis](../../projects/rest-apis/) show the APIs in use.

## Next steps

- [Embedding models](../../05-rag-and-data/embedding-models/)
- [Carbon-aware computing](../../09-ops-and-deployment/carbon-aware-computing/): Vertex AI training in low-carbon regions
