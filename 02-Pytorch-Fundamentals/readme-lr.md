# Linear Regression in PyTorch — Understanding the Training Workflow

## Overview

This project rebuilds linear regression using PyTorch.

The goal is NOT just to train a model, but to deeply understand:

- Tensors
- Autograd
- Loss functions
- Optimizers
- The PyTorch training loop

This connects the manual NumPy implementation to real deep learning workflows.

---

# 1️⃣ What Is Being Solved?

We generate synthetic data following:

    y = 3x + noise

The model must learn the relationship between `x` and `y`.

---

# 2️⃣ Model Structure

We use:

    nn.Linear(1, 1)

This represents:

    y = XW + b

Where:

- W = weight
- b = bias

PyTorch automatically:
- Initializes them randomly
- Sets requires_grad=True
- Tracks gradients

---

# 3️⃣ Key Components of a PyTorch Training Loop

Every PyTorch training loop follows this structure:

1. Forward Pass
2. Compute Loss
3. Zero Gradients
4. Backward Pass
5. Update Parameters

This pattern applies to ALL deep learning models.

---

# 4️⃣ Forward Pass

    y_pred = model(X)

The model computes:

    y_pred = XW + b

During this step:
- PyTorch builds a computation graph automatically.
- Every operation is recorded for gradient computation.

---

# 5️⃣ Loss Function

We use:

    nn.MSELoss()

MSE = mean((y_pred - y_true)^2)

This measures how wrong the predictions are.

Loss produces a single scalar value.

---

# 6️⃣ Backward Pass (Autograd)

Calling:

    loss.backward()

Does the following:

- Applies chain rule automatically
- Computes gradients of loss w.r.t model parameters
- Stores gradients inside:
  
      parameter.grad

Important:
Backward DOES NOT update parameters.
It only computes gradients.

---

# 7️⃣ Why We Call optimizer.zero_grad()

In PyTorch:

Gradients accumulate.

That means:
If we do backward multiple times without clearing gradients,
they will be added together.

So before computing new gradients, we reset them:

    optimizer.zero_grad()

---

# 8️⃣ Optimizer (SGD)

We use:

    optim.SGD(model.parameters(), lr=0.1)

SGD stands for Stochastic Gradient Descent.

It performs:

    parameter = parameter - learning_rate * gradient

Important:
The optimizer does NOT compute gradients.
It only updates parameters using gradients already stored.

---

# 9️⃣ Parameter Update

    optimizer.step()

This updates:

- Weight
- Bias

Using the gradients computed during backward().

---

# 🔟 Gradient Flow Summary

Forward:
    Build computation graph

Loss:
    Compute error

Backward:
    Compute gradients via chain rule

Step:
    Update parameters

Repeat.

---

# 1️⃣1️⃣ Inspecting Learned Parameters

After training:

    for name, param in model.named_parameters():
        print(name, param.data)

The learned weight should be close to 3.

---

# 1️⃣2️⃣ Big Picture

Manual NumPy version:

    Compute gradients manually
    Update weights manually

PyTorch version:

    loss.backward() → computes gradients
    optimizer.step() → updates weights

Same math.
Cleaner implementation.

---

# 1️⃣3️⃣ Mental Model

Think of PyTorch as:

- A calculator that remembers how you computed something
- Automatically applies chain rule
- Stores gradients
- Lets you update parameters easily

No magic.
Just automated calculus.

---

# Final Takeaway

If you understand:

- requires_grad
- computation graph
- backward()
- zero_grad()
- optimizer.step()
