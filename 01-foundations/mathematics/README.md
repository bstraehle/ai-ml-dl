https://www.deeplearning.ai/courses/mathematics-for-machine-learning-and-data-science-specialization/  

## Machine Learning Motivation

### Linear Algebra

- Matrix Operations

### Calculus
- Derivatives
- Gradients
- Optimization
- Loss Functions (Single Training Example) and Cost Functions (All Training Examples)
- Gradient Descent
- Linear Regression
- Classification

### Probability & Statistics

### Notes

Regression with Perceptron (C2_W3_Lab_1)

```
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

Classification with Perceptron (C2_W3_Lab_2)  

```
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
