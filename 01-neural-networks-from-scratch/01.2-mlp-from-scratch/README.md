# Multi-Layer Perceptron (MLP) From Scratch – XOR Classification

## Overview

This project implements a fully connected neural network (Multi-Layer Perceptron) from scratch using only NumPy.

The goal is to understand:

- Why a single neuron is not enough
- How hidden layers introduce non-linearity
- How forward propagation works
- How backpropagation works
- How gradients flow through multiple layers

The network is trained to solve the XOR problem.

---

# 1️⃣ The XOR Problem

XOR truth table:

| x1 | x2 | XOR |
|----|----|-----|
| 0  | 0  | 0   |
| 0  | 1  | 1   |
| 1  | 0  | 1   |
| 1  | 1  | 0   |

XOR is **not linearly separable**.

This means:
A single straight line cannot separate the classes.

Therefore:
A single neuron (logistic regression) cannot solve XOR.

This is why we need hidden layers.

---

# 2️⃣ Network Architecture

The architecture used:

Input (2 features)
↓
Hidden Layer (ReLU activation)
↓
Output Layer (Sigmoid activation)

Mathematically:

Z1 = X · W1 + b1  
A1 = ReLU(Z1)  

Z2 = A1 · W2 + b2  
A2 = Sigmoid(Z2)

Where:

- W1, b1 → Hidden layer parameters
- W2, b2 → Output layer parameters
- A2 → Final predicted probability

---

# 3️⃣ Forward Propagation

Forward propagation computes predictions.

Step 1 — Linear transformation (hidden layer):

Z1 = XW1 + b1  

Step 2 — Apply non-linearity:

A1 = ReLU(Z1)

ReLU function:

ReLU(z) = max(0, z)

It introduces non-linearity.

Without it, multiple layers would collapse into one linear transformation.

Step 3 — Output layer:

Z2 = A1W2 + b2  
A2 = Sigmoid(Z2)

Sigmoid function:

Sigmoid(z) = 1 / (1 + exp(-z))

This converts output into a probability between 0 and 1.

---

# 4️⃣ Why Activation Functions Are Necessary

If we removed ReLU:

Z1 = XW1 + b1  
Z2 = A1W2 + b2  

This becomes:

Z2 = X(W1W2) + (combined bias)

Which is still just a linear function.

Therefore:
Multiple linear layers without activation = single linear layer.

Activation functions allow networks to learn complex patterns.

---

# 5️⃣ Loss Function – Binary Cross Entropy (BCE)

Used for binary classification.

BCE formula:

Loss = - mean( y * log(A2) + (1 - y) * log(1 - A2) )

Why BCE?

- Measures how wrong predicted probabilities are
- Penalizes confident wrong predictions heavily
- Works perfectly with sigmoid activation

---

# 6️⃣ Backpropagation (Core Concept)

Backpropagation computes gradients for all parameters.

It uses the chain rule from calculus.

Forward pass:
Compute prediction.

Backward pass:
Propagate error backward through layers.

---

## Output Layer Gradient

For sigmoid + BCE:

dZ2 = A2 - y  

dW2 = (1/m) * A1ᵀ · dZ2  
db2 = (1/m) * sum(dZ2)

This is similar to logistic regression.

---

## Hidden Layer Gradient

Loss depends on hidden layer through W2.

Using chain rule:

dA1 = dZ2 · W2ᵀ  

For ReLU derivative:

ReLU'(z) = 1 if z > 0  
           0 otherwise  

So:

dZ1 = dA1 * (Z1 > 0)

Then:

dW1 = (1/m) * Xᵀ · dZ1  
db1 = (1/m) * sum(dZ1)

This is backpropagation.

Error flows backward.

---

# 7️⃣ Parameter Update (Gradient Descent)

Parameters are updated as:

W = W - learning_rate * dW  
b = b - learning_rate * db  

This repeats for many epochs.

---

# 8️⃣ Key Concepts Learned

- Why XOR requires hidden layers
- How non-linearity enables complex decision boundaries
- How forward propagation works
- How backpropagation applies the chain rule
- How gradients flow layer-by-layer
- Why activation functions are essential
- Why sigmoid + BCE work well together

---

# 9️⃣ Big Picture

Linear Regression:
Single linear transformation.

Logistic Regression:
Linear transformation + sigmoid.

MLP:
Linear → Activation → Linear → Activation.

Deep learning is simply stacking transformations with non-linearities.

---

# 1️⃣0️⃣ Mental Model

Forward pass:
Make prediction.

Backward pass:
Assign blame layer-by-layer.

Update:
Adjust weights slightly.

Repeat.

---

This project builds the mathematical foundation required before moving to deep learning frameworks like PyTorch.
