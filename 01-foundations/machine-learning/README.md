---
status: current
last_reviewed: 2026-11-22
review_every: 12mo
---

# Machine learning

The core ideas of machine learning: learning from data with regression, classification, neural networks, trees, clustering and recommenders. Most later topics build on these.

**Roles:** ML engineer, data scientist; AI engineers benefit too.
**Prerequisites:** [Mathematics](../mathematics/), basic Python

## Key concepts

- **Supervised learning** maps inputs to known outputs. *Regression* predicts numbers, *classification* predicts categories. Linear and logistic regression are the starting point.
- **Cost functions and gradient descent.** Training means minimizing a cost (how wrong the model is) by repeatedly stepping the parameters downhill along the gradient.
- **Overfitting and regularization.** A model that memorizes the training set fails on new data. Regularization, more data and simpler models help, and a held-out validation set shows when it's happening.
- **Neural networks** stack layers of simple units to learn non-linear patterns; frameworks like TensorFlow and PyTorch compute the gradients for you.
- **Diagnose before you tune.** High bias (underfitting) and high variance (overfitting) need different fixes. Error analysis on misclassified examples shows where to spend effort.
- **Trees and ensembles.** Decision trees, random forests and gradient-boosted trees (XGBoost) are often the strongest choice for tabular data.
- **Unsupervised learning** finds structure without labels: k-means clustering and anomaly detection.
- **Recommenders and reinforcement learning:** collaborative filtering and content-based recommendation, and agents that learn from rewards.

## Resources

| Resource | Type | Level | Cost |
|---|---|---|---|
| [Machine Learning Specialization](https://www.deeplearning.ai/courses/machine-learning-specialization/) (DeepLearning.AI, Stanford) | Course | Beginner | Free to audit when reviewed |
| [scikit-learn user guide](https://scikit-learn.org/stable/user_guide.html) | Docs | Beginner | Free |
| [StatQuest](https://www.youtube.com/@statquest) (Josh Starmer) | Videos | Beginner | Free |
| [Rules of machine learning](https://developers.google.com/machine-learning/guides/rules-of-ml) (Google) | Guide | Intermediate | Free |

## Hands-on

The specialization's labs run on its own platform. In this repo, practice on real data with the [Kaggle](../kaggle/) competitions.

## Next steps

- [Deep learning](../deep-learning/)
- [Practical data science](../practical-data-science/) and [Kaggle](../kaggle/)
