---
status: current
last_reviewed: 2026-11-22
review_every: 12mo
---

# MLOps: machine learning engineering for production

Getting a model out of a notebook and keeping it working: scoping, data pipelines, modeling pipelines, deployment and monitoring.

**Roles:** ML engineer, platform engineer, data scientist.
**Prerequisites:** [Machine learning](../../01-foundations/machine-learning/)

> I took the four-course MLOps specialization. Its specialization page now redirects to the first course only.

## Key concepts

- **The ML project lifecycle:** scoping, data, modeling, deployment, then back to data as the world changes. The model is a small part of a production ML system.
- **Data is the product.** Define labels consistently, validate data on every run, track lineage, and treat data issues as bugs.
- **Data and concept drift.** Inputs change (data drift) and so does the relationship between inputs and outputs (concept drift); monitoring catches both.
- **Pipelines.** Express data validation, feature engineering, training and evaluation as automated, reproducible pipeline steps.
- **Model analysis.** Look beyond one metric: slice performance by group, check fairness and robustness, and explain predictions.
- **Deployment patterns:** shadow mode, canary releases and A/B tests reduce the risk of a new model; serving can be online, batch or on device.
- **Monitor and retrain.** Track software metrics (latency, errors) and model metrics (prediction distribution, accuracy where labels arrive), and retrain when they degrade.

## Resources

| Resource | Type | Level | Cost |
|---|---|---|---|
| [Machine Learning Engineering for Production (MLOps)](https://www.deeplearning.ai/courses/machine-learning-engineering-for-production-mlops/) (DeepLearning.AI) | Course | Intermediate | Free to audit when reviewed |
| [Made With ML](https://madewithml.com/) | Course | Intermediate | Free |
| [Designing Machine Learning Systems](https://github.com/chiphuyen/dmls-book) (Chip Huyen) | Book resources | Intermediate | Free resources, paid book |
| [Rules of machine learning](https://developers.google.com/machine-learning/guides/rules-of-ml) (Google) | Guide | Intermediate | Free |

## Hands-on

The specialization's labs run on its own platform. In this repo, see [LLMOps](../llm-ops/) and [Amazon SageMaker AI](../../10-platforms/amazon-sagemaker/) for deployment labs.

## Next steps

- [LLMOps](../llm-ops/): the same practices for LLM applications
- [Weights & Biases](../wandb/): experiment tracking
