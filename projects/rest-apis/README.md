# Generative AI REST APIs

Postman collections for 51 generative AI providers, exported from the public [Generative AI APIs](https://www.postman.com/ai-engineer/generative-ai-apis/) workspace. Most start with a Get Started request that has a saved example response from the real API.

To use one, fork it from the workspace, or import its JSON file into Postman. Collections read API keys from your Postman Vault (`{{vault:<provider>_api_key}}`), so no keys are stored in them. Local-only APIs need no key.

Don't edit these files here: change the workspace, then run `python .github/scripts/sync_postman.py`.

| Collection | Requests | File |
|---|---|---|
| Alibaba Cloud API (Aliyun) | 23 | [alibaba-cloud-api-aliyun.postman_collection.json](alibaba-cloud-api-aliyun.postman_collection.json) |
| Amazon Bedrock API | 63 | [amazon-bedrock-api.postman_collection.json](amazon-bedrock-api.postman_collection.json) |
| Anthropic API | 29 | [anthropic-api.postman_collection.json](anthropic-api.postman_collection.json) |
| AssemblyAI API | 4 | [assemblyai-api.postman_collection.json](assemblyai-api.postman_collection.json) |
| Black Forest Labs API (FLUX) | 4 | [black-forest-labs-api-flux.postman_collection.json](black-forest-labs-api-flux.postman_collection.json) |
| Cartesia API | 5 | [cartesia-api.postman_collection.json](cartesia-api.postman_collection.json) |
| Celeris API ⚡ | 6 | [celeris-api.postman_collection.json](celeris-api.postman_collection.json) |
| Cerebras API ⚡ | 5 | [cerebras-api.postman_collection.json](cerebras-api.postman_collection.json) |
| Cohere API | 45 | [cohere-api.postman_collection.json](cohere-api.postman_collection.json) |
| Deepgram API | 28 | [deepgram-api.postman_collection.json](deepgram-api.postman_collection.json) |
| DeepInfra API | 91 | [deepinfra-api.postman_collection.json](deepinfra-api.postman_collection.json) |
| DeepSeek API | 9 | [deepseek-api.postman_collection.json](deepseek-api.postman_collection.json) |
| Docker Model Runner API (Localhost) | 15 | [docker-model-runner-api-localhost.postman_collection.json](docker-model-runner-api-localhost.postman_collection.json) |
| Eden AI API | 2 | [eden-ai-api.postman_collection.json](eden-ai-api.postman_collection.json) |
| ElevenLabs API | 11 | [elevenlabs-api.postman_collection.json](elevenlabs-api.postman_collection.json) |
| fal.ai API | 15 | [fal.ai-api.postman_collection.json](fal.ai-api.postman_collection.json) |
| Fireworks AI API | 23 | [fireworks-ai-api.postman_collection.json](fireworks-ai-api.postman_collection.json) |
| Google Gemini API & Vertex AI API | 70 | [google-gemini-api-vertex-ai-api.postman_collection.json](google-gemini-api-vertex-ai-api.postman_collection.json) |
| Groq API ⚡ | 13 | [groq-api.postman_collection.json](groq-api.postman_collection.json) |
| Hugging Face API | 64 | [hugging-face-api.postman_collection.json](hugging-face-api.postman_collection.json) |
| IBM watsonx.ai API | 72 | [ibm-watsonx.ai-api.postman_collection.json](ibm-watsonx.ai-api.postman_collection.json) |
| InclusionAI API (Ant Ling) | 7 | [inclusionai-api-ant-ling.postman_collection.json](inclusionai-api-ant-ling.postman_collection.json) |
| Luma AI API | 5 | [luma-ai-api.postman_collection.json](luma-ai-api.postman_collection.json) |
| Meta Model API | 25 | [meta-model-api.postman_collection.json](meta-model-api.postman_collection.json) |
| Microsoft Foundry API (Azure OpenAI) | 9 | [microsoft-foundry-api-azure-openai.postman_collection.json](microsoft-foundry-api-azure-openai.postman_collection.json) |
| MiniMax API | 13 | [minimax-api.postman_collection.json](minimax-api.postman_collection.json) |
| Mistral AI API | 61 | [mistral-ai-api.postman_collection.json](mistral-ai-api.postman_collection.json) |
| Moonshot AI API (Kimi) | 9 | [moonshot-ai-api-kimi.postman_collection.json](moonshot-ai-api-kimi.postman_collection.json) |
| Nebius AI API | 17 | [nebius-ai-api.postman_collection.json](nebius-ai-api.postman_collection.json) |
| Novita API | 32 | [novita-api.postman_collection.json](novita-api.postman_collection.json) |
| NVIDIA AI API | 22 | [nvidia-ai-api.postman_collection.json](nvidia-ai-api.postman_collection.json) |
| Ollama API (Localhost & Cloud) | 24 | [ollama-api-localhost-cloud.postman_collection.json](ollama-api-localhost-cloud.postman_collection.json) |
| OpenAI API | 127 | [openai-api.postman_collection.json](openai-api.postman_collection.json) |
| OpenRouter API | 77 | [openrouter-api.postman_collection.json](openrouter-api.postman_collection.json) |
| Perplexity AI API | 6 | [perplexity-ai-api.postman_collection.json](perplexity-ai-api.postman_collection.json) |
| Reflection AI API | 4 | [reflection-ai-api.postman_collection.json](reflection-ai-api.postman_collection.json) |
| Reka AI API | 6 | [reka-ai-api.postman_collection.json](reka-ai-api.postman_collection.json) |
| Replicate API | 23 | [replicate-api.postman_collection.json](replicate-api.postman_collection.json) |
| Runway API | 5 | [runway-api.postman_collection.json](runway-api.postman_collection.json) |
| SambaNova API ⚡ | 7 | [sambanova-api.postman_collection.json](sambanova-api.postman_collection.json) |
| SiliconFlow API | 14 | [siliconflow-api.postman_collection.json](siliconflow-api.postman_collection.json) |
| Stability AI API | 11 | [stability-ai-api.postman_collection.json](stability-ai-api.postman_collection.json) |
| StepFun API | 5 | [stepfun-api.postman_collection.json](stepfun-api.postman_collection.json) |
| Tencent Hy API (TokenHub) | 26 | [tencent-hy-api-tokenhub.postman_collection.json](tencent-hy-api-tokenhub.postman_collection.json) |
| Together AI API | 63 | [together-ai-api.postman_collection.json](together-ai-api.postman_collection.json) |
| TypeSafe AI API | 6 | [typesafe-ai-api.postman_collection.json](typesafe-ai-api.postman_collection.json) |
| Upstage API (Solar) | 9 | [upstage-api-solar.postman_collection.json](upstage-api-solar.postman_collection.json) |
| Voyage AI API (MongoDB) | 30 | [voyage-ai-api-mongodb.postman_collection.json](voyage-ai-api-mongodb.postman_collection.json) |
| xAI API | 28 | [xai-api.postman_collection.json](xai-api.postman_collection.json) |
| Xiaomi MiMo API | 6 | [xiaomi-mimo-api.postman_collection.json](xiaomi-mimo-api.postman_collection.json) |
| Z.ai API (GLM) | 8 | [z.ai-api-glm.postman_collection.json](z.ai-api-glm.postman_collection.json) |
