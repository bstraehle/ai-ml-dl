---
status: current
last_reviewed: 2026-11-22
review_every: 12mo
---

# Open-source models (Hugging Face)

Using open models from the Hugging Face Hub for text, speech and vision tasks with a few lines of code, and sharing them as apps.

**Roles:** AI engineer, ML engineer, data scientist.
**Prerequisites:** basic Python

## Key concepts

- **The Hub** hosts open models, datasets and demo apps. A model card describes what a model does, how it was trained, its limits and its license.
- **Pipelines** wrap model loading, preprocessing and postprocessing behind one call per task: chat, translation, summarization, embeddings, speech, vision ([labs](#hands-on)).
- **Pick the smallest model that works.** Bigger models are better but need more memory and are slower. Check the hardware you have and try a small model first.
- **Check the license.** "Open" ranges from permissive (Apache 2.0, MIT) to licenses with use restrictions. Read it before using a model commercially.
- **Audio needs the right sampling rate.** Resample audio to what the model was trained on ([L5](L5_Zero-Shot_Audio_Classification.ipynb)).
- **Zero-shot models** like CLIP classify against labels you choose at run time, with no training ([L13](L13_Zero_Shot_Image_Classification.ipynb)).
- **Share as an app** with Gradio on Hugging Face Spaces ([L14](L14_deployment.ipynb)).

## Resources

| Resource | Type | Level | Cost |
|---|---|---|---|
| [Open source models with Hugging Face](https://www.deeplearning.ai/short-courses/open-source-models-hugging-face) | Short course | Beginner | Free when reviewed |
| [Transformers pipelines](https://huggingface.co/docs/transformers/main_classes/pipelines) | Docs | Beginner | Free |
| [Hugging Face LLM course](https://huggingface.co/learn/llm-course) | Course | Intermediate | Free |

## Hands-on

| Lab | Task |
|---|---|
| [L2](L2_NLP.ipynb) | Chatbot |
| [L3](L3_Translation_and_Summarization.ipynb) | Translation and summarization |
| [L4](L4_Sentence_Embeddings.ipynb) | Sentence embeddings |
| [L5](L5_Zero-Shot_Audio_Classification.ipynb) | Zero-shot audio classification |
| [L6](L6_Automatic_Speech_Recognition.ipynb) | Speech recognition, with a Gradio app |
| [L7](L7_Text_to_Speech.ipynb) | Text to speech |
| [L8](L8_object_detection.ipynb) | Object detection |
| [L9](L9_segmentation.ipynb) | Segmentation with SAM, and depth estimation |
| [L10](L10_image_retrieval.ipynb) | Image-text retrieval |
| [L11](L11_image_captioning.ipynb) | Image captioning |
| [L12](L12_visual_q_and_a.ipynb) | Visual question answering |
| [L13](L13_Zero_Shot_Image_Classification.ipynb) | Zero-shot image classification |
| [L14](L14_deployment.ipynb) | Deploying to Hugging Face Spaces |

## Next steps

- [Llama](../llama/) and [Mistral](../mistral/): open-weights model families
- [Gradio](../../04-building-with-llms/gradio/)
