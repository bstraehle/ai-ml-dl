---
status: current
last_reviewed: 2026-11-22
review_every: 12mo
---

# Mathematics for machine learning

The linear algebra, calculus, and probability and statistics that machine learning is built on, with labs in Python and NumPy.

**Roles:** ML engineer, data scientist, research engineer.
**Prerequisites:** high-school algebra, basic Python

## Key concepts

- **Linear algebra.** Data is stored as vectors and matrices. Systems of linear equations, matrix multiplication and linear transformations describe what models compute; eigenvalues and eigenvectors underlie techniques like PCA.
- **Calculus.** Derivatives and gradients say how a cost changes when parameters change. *Loss* measures the error on one example, *cost* averages it over the training set.
- **Optimization.** Gradient descent repeatedly steps the parameters downhill; Newton's method uses second derivatives to converge faster. Linear regression and classification are both optimization problems.
- **Probability.** Distributions, conditional probability and Bayes' rule explain uncertainty and puzzles like the Monty Hall and birthday problems.
- **Statistics.** Summary statistics describe data, the central limit theorem explains why averages behave predictably, and hypothesis tests (A/B testing) decide whether a difference is real.

## Notes

Regression with a perceptron (C2_W3_Lab_1):

```python
for i in range(0, num_iterations):
    # Forward propagation. Inputs: "X, parameters". Outputs: "Y_hat".
    Y_hat = forward_propagation(X, parameters)
    # Cost function. Inputs: "Y_hat, Y". Outputs: "cost".
    cost = compute_cost(Y_hat, Y)
    # Backpropagation. Inputs: "Y_hat, X, Y". Outputs: "grads".
    grads = backward_propagation(Y_hat, X, Y)
    # Gradient descent parameter update. Inputs: "parameters, grads, learning_rate". Outputs: "parameters".
    parameters = update_parameters(parameters, grads, learning_rate)
    # Print the cost every iteration.
    print ("Cost after iteration %i: %f" %(i, cost))
```

Classification with a perceptron (C2_W3_Lab_2):

```python
for i in range(0, num_iterations):
    # Forward propagation. Inputs: "X, parameters". Outputs: "A".
    A = forward_propagation(X, parameters)
    # Cost function. Inputs: "A, Y". Outputs: "cost".
    cost = compute_cost(A, Y)
    # Backpropagation. Inputs: "A, X, Y". Outputs: "grads".
    grads = backward_propagation(A, X, Y)
    # Gradient descent parameter update. Inputs: "parameters, grads, learning_rate". Outputs: "parameters".
    parameters = update_parameters(parameters, grads, learning_rate)
    # Print the cost every iteration.
    print ("Cost after iteration %i: %f" %(i, cost))
```

## Resources

| Resource | Type | Level | Cost |
|---|---|---|---|
| [Mathematics for Machine Learning and Data Science](https://www.deeplearning.ai/courses/mathematics-for-machine-learning-and-data-science-specialization/) (DeepLearning.AI) | Course | Beginner | Free to audit when reviewed |
| [Essence of linear algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) (3Blue1Brown) | Videos | Beginner | Free |
| [Mathematics for Machine Learning](https://mml-book.github.io/) (Deisenroth, Faisal, Ong) | Book | Intermediate | Free |
| [StatQuest](https://www.youtube.com/@statquest) (Josh Starmer) | Videos | Beginner | Free |

## Hands-on

| Course | Labs and assignments |
|---|---|
| 1. Linear algebra | [NumPy basics](C1_W1_Lab_1.ipynb), linear systems with [2](C1_W1_Lab_2.ipynb) and [3](C1_W2_Lab_1.ipynb) variables, [systems of equations](C1_W2_Assignment.ipynb), [vector operations](C1_W3_Lab_1.ipynb), [matrix multiplication](C1_W3_Lab_2.ipynb), [linear transformations](C1_W3_Lab_3.ipynb), [a single-perceptron network](C1_W3_Assignment.ipynb), [eigenvalues and eigenvectors](C1_W4_Assignment.ipynb) |
| 2. Calculus | [Differentiation in Python](C2_W1_Lab_1.ipynb), [optimizing a cost function](C2_W1_Assignment.ipynb), gradient descent in [one](C2_W2_Lab_1.ipynb) and [two](C2_W2_Lab_2.ipynb) variables, [linear regression with gradient descent](C2_W2_Assignment.ipynb), perceptron [regression](C2_W3_Lab_1.ipynb) and [classification](C2_W3_Lab_2.ipynb), [Newton's method](C2_W3_Lab_3.ipynb), [a two-layer network](C2_W3_Assignment.ipynb) |
| 3. Probability and statistics | [Monty Hall](C3_W1_Lab_1.ipynb), [birthday problems](C3_W1_Lab_2.ipynb), [probability distributions](C3_W1_Assignment.ipynb), [summary statistics](C3_W2_Lab_1.ipynb), [loaded dice](C3_W2_Assignment.ipynb), [central limit theorem](C3_W3_Lab_1.ipynb), [A/B testing](C3_W4_Assignment.ipynb) |

## Next steps

- [Machine learning](../machine-learning/)
