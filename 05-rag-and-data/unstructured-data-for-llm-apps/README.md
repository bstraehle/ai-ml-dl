---
status: current
last_reviewed: 2026-11-22
review_every: 12mo
---

# Unstructured data for LLM apps

Most company knowledge lives in PDFs, slides, web pages, emails and scans. Before an LLM app can use it, the content has to be extracted, cleaned, split and labeled. This step decides much of a RAG system's quality.

**Roles:** AI engineer, data scientist.
**Prerequisites:** [Embedding models](../embedding-models/)

## Key concepts

- **Normalize first.** Convert every format (HTML, PowerPoint, PDF, EPUB) into the same list of typed elements: titles, narrative text, list items, tables, images ([Lesson 2](Lesson_2_Student.ipynb)).
- **Metadata is a feature.** Keep the page number, section, source and element type with each chunk. It enables filtering ("only chapter 3"), citations and better ranking ([Lesson 3](Lesson_3_Student.ipynb)).
- **Chunk by structure, not by character count.** Splitting at section and title boundaries keeps related text together. Fixed-size chunks often cut ideas in half.
- **PDFs and images need layout detection.** Document layout models, or vision-language models, find titles, columns and tables in scans and complex PDFs, where plain text extraction scrambles the reading order ([Lesson 4](Lesson_4_Student.ipynb)).
- **Tables are special.** Extract them as HTML or structured data rather than flattened text, or the model loses the rows and columns ([Lesson 5](Lesson_5_Student.ipynb)).
- **Filter noise.** Drop headers, footers, page numbers and boilerplate before embedding ([Lesson 6](Lesson_6_Student.ipynb)).

## Resources

| Resource | Type | Level | Cost |
|---|---|---|---|
| [Preprocessing unstructured data for LLM applications](https://www.deeplearning.ai/short-courses/preprocessing-unstructured-data-for-llm-applications/) (Unstructured) | Short course | Beginner | Free when reviewed |
| [Unstructured](https://github.com/Unstructured-IO/unstructured) | Library | Intermediate | Free |
| [Docling](https://github.com/docling-project/docling) | Library | Intermediate | Free |

## Hands-on

| Lab | What it covers |
|---|---|
| [Lesson 2](Lesson_2_Student.ipynb) | Normalizing HTML, PowerPoint and PDF content into elements |
| [Lesson 3](Lesson_3_Student.ipynb) | Metadata extraction and chunking by chapter; loading into a vector database |
| [Lesson 4](Lesson_4_Student.ipynb) | Preprocessing PDFs and images with document layout detection |
| [Lesson 5](Lesson_5_Student.ipynb) | Extracting tables |
| [Lesson 6](Lesson_6_Student.ipynb) | Building a RAG bot over a PDF, slides and a README |

## Next steps

- [Advanced RAG](../advanced-rag/)
- [LLM evals](../../08-evaluation-and-safety/llm-evals/): measure whether better preprocessing improved answers
