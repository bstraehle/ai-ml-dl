---
status: current
last_reviewed: 2026-11-21
review_every: 6mo
---

# Advanced RAG

Retrieval-augmented generation (RAG) answers questions from your own data. It retrieves relevant passages and puts them in the prompt, so the model grounds its answer in them instead of relying on what it memorized. This folder covers the full pipeline, from search basics to knowledge graphs, multimodal RAG and evaluation.

**Roles:** AI engineer, ML engineer, data scientist. Product managers: read *Key concepts*.
**Prerequisites:** [Embedding models](../embedding-models/), [Prompt engineering](../openai/)

## Key concepts

- **The pipeline:** ingest, chunk, embed, index, retrieve, (rerank), generate. Most quality problems come from the first two steps and from retrieval, not from the LLM.
- **Search types.** Keyword search (BM25) matches exact terms, dense search matches meaning, and hybrid search combines both. Hybrid is usually the safest default ([Weaviate](weaviate/), [Pinecone](pinecone/) and [Cohere](cohere/) labs).
- **Vector indexes** trade a little accuracy for a lot of speed using approximate nearest-neighbour search (for example HNSW). Vector quantization shrinks memory further ([Qdrant labs](qdrant/)).
- **Improving retrieval.** Query expansion and rewriting, cross-encoder reranking and embedding adaptors ([Chroma labs](chroma/)). Also sentence-window and auto-merging retrieval, which return a larger surrounding context than the matched chunk ([LlamaIndex labs](llamaindex/)). Metadata filters and boosting help too ([MongoDB labs](mongodb-atlas-vector-search/)).
- **Knowledge graphs** store entities and relationships, so you can answer questions that span many documents ("which managers invest in companies X and Y?") ([Neo4j labs](neo4j/)).
- **Multimodal RAG** retrieves images and video frames alongside text, and answers with a vision-language model ([Weaviate multimodal](weaviate-multimodal/), [video RAG](multimodal-rag-videos/)).
- **Long context vs RAG.** Long-context models can take whole documents, but RAG is still cheaper, faster and easier to keep current. Many systems combine the two ([AI21 Labs](ai21-labs/)).
- **Evaluate with the RAG triad:** context relevance (did we retrieve the right passages?), groundedness (is the answer supported by them?) and answer relevance (does it answer the question?) ([LlamaIndex labs](llamaindex/)).
- **Run it as a pipeline.** Ingestion is a scheduled data pipeline with retries and monitoring, not a one-off notebook ([Astronomer / Airflow labs](astronomer/)).

## Resources

Top picks (DeepLearning.AI short courses were free to enroll when this page was reviewed):

| Resource | Type | Level | Cost |
|---|---|---|---|
| [Vector databases: from embeddings to applications](https://www.deeplearning.ai/short-courses/vector-databases-embeddings-applications/) (Weaviate) | Short course | Beginner | Free |
| [Advanced retrieval for AI](https://www.deeplearning.ai/short-courses/advanced-retrieval-for-ai/) (Chroma) | Short course | Intermediate | Free |
| [Building and evaluating advanced RAG](https://www.deeplearning.ai/short-courses/building-evaluating-advanced-rag/) (LlamaIndex, TruEra) | Short course | Intermediate | Free |
| [Knowledge graphs for RAG](https://www.deeplearning.ai/short-courses/knowledge-graphs-rag/) (Neo4j) | Short course | Intermediate | Free |
| [RAG++ course notebooks](https://github.com/wandb/edu/tree/main/rag-advanced) (Weights & Biases) | Notebooks | Advanced | Free |

<details>
<summary>More courses and videos</summary>

**Search and vector databases**
- [Large language models with semantic search](https://www.deeplearning.ai/short-courses/large-language-models-semantic-search/) (Cohere)
- [Building applications with vector databases](https://www.deeplearning.ai/short-courses/building-applications-vector-databases/) (Pinecone)
- [Retrieval optimization: from tokenization to vector quantization](https://www.deeplearning.ai/short-courses/retrieval-optimization-from-tokenization-to-vector-quantization/) (Qdrant)
- [Prompt compression and query optimization](https://www.deeplearning.ai/short-courses/prompt-compression-and-query-optimization/) (MongoDB)

**Multimodal and long context**
- [Building multimodal search and RAG](https://www.deeplearning.ai/short-courses/building-multimodal-search-and-rag/) (Weaviate)
- [Multimodal RAG: chat with videos](https://www.deeplearning.ai/short-courses/multimodal-rag-chat-with-videos/) (Intel)
- [Build long-context AI apps with Jamba](https://www.deeplearning.ai/short-courses/build-long-context-ai-apps-with-jamba/) (AI21 Labs)

**Production**
- [Orchestrating workflows for GenAI applications](https://www.deeplearning.ai/short-courses/orchestrating-workflows-for-genai-applications/) (Astronomer)

**Videos**
- [LangChain v0.1.0 launch: retrieval](https://www.youtube.com/watch?v=yK64dws8f6A) (LangChain)
- [Multi-modal retrieval augmented generation with LlamaIndex](https://www.youtube.com/watch?v=35RlrrgYDyU) (LlamaIndex)
- [Introduction to AI agents with Milvus and LlamaIndex](https://www.youtube.com/watch?v=YL9zey9NLlw) (Zilliz) · [cookbooks](https://github.com/ytang07/ai_agents_cookbooks/tree/main/)

</details>

## Hands-on

| Folder | What the labs cover |
|---|---|
| [cohere/](cohere/) | Keyword search, embeddings, dense retrieval, reranking, generating answers |
| [weaviate/](weaviate/) | Embeddings, kNN and approximate nearest neighbours, sparse, dense and hybrid search, multilingual RAG |
| [pinecone/](pinecone/) | Semantic search, RAG, recommender systems, hybrid search, facial similarity, anomaly detection |
| [chroma/](chroma/) | Where vector search fails, query expansion, cross-encoder reranking, embedding adaptors |
| [llamaindex/](llamaindex/) | An advanced RAG pipeline, the RAG triad, sentence-window and auto-merging retrieval |
| [qdrant/](qdrant/) | Embedding models, tokenizers, measuring relevance, HNSW tuning, vector quantization |
| [mongodb-atlas-vector-search/](mongodb-atlas-vector-search/) | Vector search, metadata filters, projections, boosting, prompt compression |
| [neo4j/](neo4j/) | Cypher, building a knowledge graph from SEC filings, chatting with the graph |
| [weaviate-multimodal/](weaviate-multimodal/) | Multimodal search, large multimodal models, multimodal RAG, recommenders |
| [multimodal-rag-videos/](multimodal-rag-videos/) | Multimodal embeddings, preprocessing video, retrieval, vision-language models |
| [ai21-labs/](ai21-labs/) | Jamba prompting, tool calling, long-context prompting |
| [astronomer/](astronomer/) | Turning a RAG notebook into a scheduled, parameterized and fault-tolerant Airflow pipeline |

## My projects

- [RAG with LangChain and LlamaIndex](../hugging-face/rag/)
- [Advanced RAG with pre- and post-retrieval processing](../hugging-face/advanced-rag/)

## Next steps

- [Agentic AI](../agentic-ai/): let an agent decide when and what to retrieve (agentic RAG)
- [Unstructured data](../unstructured-data-for-llm-apps/): parse PDFs, tables and images for ingestion
- [Quality & safety](../quality-safety-llm-apps/): monitor RAG apps in production
