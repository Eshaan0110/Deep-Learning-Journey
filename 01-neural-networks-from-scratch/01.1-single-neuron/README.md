# 01.1 — Single Neuron: Logistic Regression

The simplest possible classifier: one neuron with a sigmoid activation and binary cross-entropy loss.

## The Neuron Model

```
x ──► [w·x + b] ──► sigmoid ──► p ∈ (0, 1) ──► classify
```

### Forward pass

```
z = w·x + b          (linear transform)
p = σ(z) = 1 / (1 + e^{-z})   (sigmoid squashes to probability)
```

### Loss: Binary Cross-Entropy

```
L = -[ y·log(p) + (1-y)·log(1-p) ]
```

BCE punishes wrong, confident predictions much more than wrong, uncertain ones. This is why it converges faster than MSE for classification.

### Gradients

Because BCE + sigmoid compose cleanly, the gradient simplifies to:

```
∂L/∂w = (p - y) · x
∂L/∂b = (p - y)
```

No chain-rule complexity — the sigmoid's derivative cancels with BCE's derivative.

### Decision Boundary

The boundary is where z = 0, i.e. `x = -b / w`. Points to the right predict class 1; to the left, class 0.

## Run

```bash
python logistic_regression.py
```

The script generates linearly separable data (`y = 1 if x > 0`), trains for 1000 epochs, and plots the learned decision boundary.

## Why not MSE for classification?

MSE loss on a sigmoid output produces very flat gradients when the prediction is wrong but confident (sigmoid saturates near 0 or 1). BCE is derived from maximum likelihood estimation and gives strong gradients exactly where they are needed most.
