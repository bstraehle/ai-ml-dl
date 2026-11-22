---
status: current
last_reviewed: 2026-11-22
review_every: 6mo
---

# Amazon Bedrock

Amazon Bedrock is AWS's managed API for foundation models from many providers, including Amazon, Anthropic, Meta and Mistral. You call models without managing servers, and connect them to other AWS services.

**Roles:** AI engineer, solutions architect.
**Prerequisites:** an AWS account, basic Python

## Key concepts

- **One API, many models.** The Converse API uses the same request shape for every chat model. The older InvokeModel API takes each provider's own request format.
- **Model access and regions.** Models are enabled per account and region. Newer models often run through cross-region inference profiles.
- **Streaming** returns tokens as they're generated, for responsive apps ([streaming notebook](gen-ai-txt-stream.ipynb)).
- **Beyond text:** image generation models are called the same way ([image notebook](gen-ai-img.ipynb)).
- **Serverless apps.** Combine Bedrock with Amazon Transcribe, S3 and Lambda for event-driven pipelines, such as summarizing every audio file uploaded to a bucket ([serverless labs](serverless-llm-apps/)).
- **Log and secure it.** Turn on invocation logging, and use IAM to control who can call which model.

## Resources

| Resource | Type | Level | Cost |
|---|---|---|---|
| [Serverless LLM apps with Amazon Bedrock](https://www.deeplearning.ai/short-courses/serverless-llm-apps-amazon-bedrock/) | Short course | Beginner | Free when reviewed |
| [What is Amazon Bedrock?](https://docs.aws.amazon.com/bedrock/latest/userguide/what-is-bedrock.html) | Docs | Beginner | Free |
| [Converse API](https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html) | Docs | Intermediate | Free |
| [Amazon Bedrock workshop](https://github.com/aws-samples/amazon-bedrock-workshop) | Labs | Intermediate | Free (AWS usage billed) |

## Hands-on

| Lab | What it covers |
|---|---|
| [gen-ai-txt.ipynb](gen-ai-txt.ipynb) | A first text generation call |
| [gen-ai-txt-stream.ipynb](gen-ai-txt-stream.ipynb) | Streaming responses |
| [gen-ai-img.ipynb](gen-ai-img.ipynb) | Generating an image |
| [serverless-llm-apps/](serverless-llm-apps/) | First generations, summarizing an audio file, logging, a Lambda function, and event-driven generation |

The serverless labs 4 and 5 used a prebuilt Lambda layer (a 25 MB zip) that was removed from the repo; build a layer with the packages in [serverless-llm-apps/1/requirements.txt](serverless-llm-apps/1/requirements.txt) (boto3 and jinja2) instead.

My [Amazon Bedrock Space](../../projects/hugging-face/bedrock/) and the Bedrock collection in [projects/rest-apis](../../projects/rest-apis/) show the API in use.

## Next steps

- [Amazon SageMaker AI](../amazon-sagemaker/): train and host your own models
- [Agentic AI](../../06-agents/agentic-ai/): the serverless agentic workflows course
