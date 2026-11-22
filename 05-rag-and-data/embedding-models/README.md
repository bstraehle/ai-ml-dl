---
status: current
last_reviewed: 2026-11-22
review_every: 12mo
---

# Embedding models

Embedding models turn text (or images) into vectors, so that similar meanings end up close together. They power semantic search, RAG, clustering, deduplication and recommendations.

**Roles:** AI engineer, ML engineer, data scientist.
**Prerequisites:** [How LLMs work](../../02-llm-fundamentals/gen-ai-llms/)

## Key concepts

- **From words to context.** Word embeddings (Word2Vec, GloVe) give each word one vector. Contextual models (BERT) give the same word different vectors depending on the sentence ([Lesson 2](Lesson_2.ipynb)).
- **Token vs sentence embeddings.** Averaging BERT's token vectors makes a poor sentence embedding. Sentence models such as SBERT are trained to produce one good vector per text ([Lesson 3](Lesson_3.ipynb)).
- **Dual encoders vs cross-encoders.** A dual (bi-)encoder embeds query and document separately, so documents can be embedded once and searched fast. A cross-encoder reads both together: more accurate, too slow for search, ideal for reranking the top results.
- **Contrastive training.** Dual encoders learn by pulling matching pairs together and pushing other in-batch examples apart, which turns into a cross-entropy loss over a similarity matrix ([Lesson 4](Lesson_4.ipynb)).
- **Asymmetric search.** Questions and answers look different. Some models use separate query and passage encoders or prefixes for this ([Lesson 5](Lesson_5.ipynb)).
- **Choosing a model.** Compare on a benchmark close to your task (the MTEB leaderboard), then test on your own data. Also weigh dimensions (storage), maximum input length, languages and cost.

## Resources

| Resource | Type | Level | Cost |
|---|---|---|---|
| [Embedding models: from architecture to implementation](https://www.deeplearning.ai/short-courses/embedding-models-from-architecture-to-implementation/) (Vectara) | Short course | Intermediate | Free when reviewed |
| [Sentence Transformers documentation](https://sbert.net/) | Docs | Intermediate | Free |
| [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard) | Benchmark | All levels | Free |

## Hands-on

| Lab | What it covers |
|---|---|
| [Lesson 2](Lesson_2.ipynb) | GloVe and Word2Vec, contextual BERT embeddings, cross-encoders |
| [Lesson 3](Lesson_3.ipynb) | Token vs sentence embeddings, SBERT and dual encoders |
| [Lesson 4](Lesson_4.ipynb) | Training a dual encoder with a contrastive loss |
| [Lesson 5](Lesson_5.ipynb) | Inference with separate question and answer encoders |

## Next steps

- [Advanced RAG](../advanced-rag/): use embeddings for retrieval, with reranking and hybrid search
- [Unstructured data](../unstructured-data-for-llm-apps/): prepare documents for embedding
