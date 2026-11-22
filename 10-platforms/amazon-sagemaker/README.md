---
status: current
last_reviewed: 2026-11-21
review_every: 12mo
---

# Amazon SageMaker AI

Amazon SageMaker AI (formerly Amazon SageMaker) is AWS's managed service for training, tuning and deploying machine learning models. You bring a training script; it runs the script on the instances you choose and hosts the trained model behind an endpoint.

**Roles:** ML engineer, data scientist.
**Prerequisites:** [Deep learning](../../01-foundations/deep-learning/), an AWS account

## Key concepts

- **Training jobs.** Your script runs in a managed container on the instance type you pick (for example a GPU), reads data from S3, and writes the model back to S3. You pay only while the job runs.
- **Hyperparameters as arguments.** SageMaker passes hyperparameters to your script as command-line arguments, so the same script runs in the notebook and as a training job ([mnist-train-cnn.py](mnist-train-cnn.py)).
- **Endpoints** host a model for real-time inference and bill by the hour while they run. Delete them when you're done.
- **Notebooks for exploration.** The same notebook environment is useful for looking inside models, as in the transformer lab below.
- **Higher-level options:** for generative AI, [Amazon Bedrock](../amazon-bedrock/) gives API access to foundation models without managing instances.

## Resources

| Resource | Type | Level | Cost |
|---|---|---|---|
| [What is Amazon SageMaker AI?](https://docs.aws.amazon.com/sagemaker/latest/dg/whatis.html) | Docs | Beginner | Free |
| [Train and deploy Keras models on Amazon SageMaker](https://aws.amazon.com/blogs/machine-learning/train-and-deploy-keras-models-with-tensorflow-and-apache-mxnet-on-amazon-sagemaker/) | Blog post | Intermediate | Free |
| [exBERT](https://huggingface.co/spaces/exbert-project/exbert) | Interactive tool | Intermediate | Free |

## Hands-on

| Lab | What it covers |
|---|---|
| [1-mnist-tune-cnn.ipynb](1-mnist-tune-cnn.ipynb) | Building a Keras convolutional neural network for MNIST in a notebook |
| [2-mnist-train-deploy-cnn.ipynb](2-mnist-train-deploy-cnn.ipynb) | Uploading data to S3, training on a GPU instance, deploying an endpoint (based on the blog post above) |
| [mnist-train-cnn.py](mnist-train-cnn.py) | The training script, with hyperparameters passed in by SageMaker |
| [Transformers_MLCourse_SageMaker.ipynb](Transformers_MLCourse_SageMaker.ipynb) | Tokenizers, positional encoding and self-attention in BERT, then GPT-2 |

## Next steps

- [ML engineering for production (MLOps)](../../09-ops-and-deployment/machine-learning-eng/)
- [Amazon Bedrock](../amazon-bedrock/) and [Google Vertex AI](../google-vertex-ai/)
