# PyTorch Tensor Basics & Autograd — From Scratch Understanding

## Overview

This module builds a deep understanding of:

- What a tensor is
- Tensor shapes and operations
- requires_grad
- Computation graphs
- Backpropagation
- Gradient accumulation
- How PyTorch computes gradients automatically

This is the foundation before building neural networks in PyTorch.

---

# 1️⃣ What Is a Tensor?

A tensor is a multi-dimensional array.

It is similar to a NumPy array but with two additional superpowers:

1. Automatic differentiation (autograd)
2. GPU support

Examples:

Scalar:
```
torch.tensor(5.0)
```

Vector:
```
torch.tensor([1.0, 2.0, 3.0])
```

Matrix:
```
torch.tensor([[1.0, 2.0],
              [3.0, 4.0]])
```

---

# 2️⃣ Tensor Properties

Every tensor has:

- Shape
- Data type (dtype)
- Device (CPU or GPU)
- requires_grad flag

Example:

```
x = torch.tensor([[1.0, 2.0]])
print(x.shape)
print(x.dtype)
print(x.device)
```

---

# 3️⃣ Basic Tensor Operations

Addition:
```
a + b
```

Multiplication:
```
a * b
```

Matrix multiplication:
```
torch.matmul(A, B)
```

All operations are vectorized and optimized.

---

# 4️⃣ requires_grad Explained

When we define:

```
w = torch.tensor(2.0, requires_grad=True)
```

We are telling PyTorch:

Track operations on this tensor so we can compute gradients later.

Without this flag, PyTorch behaves like NumPy.

---

# 5️⃣ Computation Graph

When we compute:

```
y = w * x + b
```

PyTorch builds a graph of operations.

Each operation stores:

- What was done
- How to compute its derivative

This is called a dynamic computation graph.

---

# 6️⃣ Backward Pass

Calling:

```
y.backward()
```

Does the following:

1. Starts from y
2. Applies chain rule
3. Propagates gradients backward
4. Stores gradients in `.grad`

Example:

If:
```
y = w * x + b
x = 3
```

Then:

```
dy/dw = 3
dy/db = 1
```

And PyTorch stores:

```
w.grad = 3
b.grad = 1
```

---

# 7️⃣ Gradient Accumulation

Calling backward multiple times accumulates gradients.

Example:

```
y.backward()
y.backward()
```

Gradients double.

That is why we must reset gradients before every training step.

---

# 8️⃣ Leaf vs Non-Leaf Tensors

Leaf tensors:
- Created directly by user
- Have requires_grad=True
- Store gradients in `.grad`

Intermediate tensors:
- Created by operations
- Have grad_fn
- Do NOT store gradients unless explicitly asked

---

# 9️⃣ Mental Model

Forward pass:
Compute prediction.

Backward pass:
Apply chain rule automatically.

Update:
Adjust parameters using gradients.

---

This understanding removes the “magic” from PyTorch.
