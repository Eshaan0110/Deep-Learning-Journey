# 2.2 MLP in PyTorch — From Scratch to nn.Module

## What We're Building

In module 01.2 we built an MLP by hand: manual weight matrices, manual backpropagation, global variables. It worked, but it doesn't scale. Real deep learning code uses PyTorch's `nn.Module` system.

This module rebuilds the same idea — a multi-layer perceptron for a binary classification problem — using PyTorch's abstractions:

| Manual (01.2)              | PyTorch (2.2)                      |
|----------------------------|------------------------------------|
| `W1 = np.random.randn(...)` | `nn.Linear(in, out)`              |
| Manual backprop loop       | `loss.backward()`                  |
| `W -= lr * dW`             | `optimizer.step()`                 |
| Only training data         | Train / validation split           |
| SGD only                   | Adam optimizer                     |
| No saving                  | `torch.save` / `load_state_dict`   |

---

## The Problem: Concentric Circles

XOR was a proof of concept — 4 data points. Here we use a harder dataset: 500 points arranged in two concentric circles.

```
Class 0 (inner circle):  radius ≈ 0.5
Class 1 (outer circle):  radius ≈ 1.2
With Gaussian noise added to both
```

No single line can separate inner from outer. The MLP must learn a curved, non-linear decision boundary.

---

## nn.Module: The Right Way to Define Models

```python
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
```

### Why inherit from nn.Module?

1. **Parameter tracking**: `model.parameters()` automatically finds every weight and bias in the network, no matter how deep. You don't have to list W1, b1, W2, b2 manually.

2. **`train()` / `eval()` modes**: Some layers (BatchNorm, Dropout) behave differently during training vs inference. Calling `model.train()` or `model.eval()` switches all layers at once.

3. **`state_dict()`**: A dictionary of all learned parameters. Lets you save checkpoints and resume training.

4. **`nn.Sequential`**: Chains layers together so `forward()` just calls them in order. Clean and readable.

---

## Adam Optimizer vs SGD

In module 2.1 we used SGD (Stochastic Gradient Descent):

```
W = W - lr * grad
```

Adam is an adaptive optimizer. It keeps a running average of:
- The gradient itself (momentum)
- The squared gradient (RMSprop-style scaling)

```
m = β1 * m + (1 - β1) * grad          # 1st moment: gradient direction
v = β2 * v + (1 - β2) * grad²          # 2nd moment: gradient magnitude
W = W - lr * m / (√v + ε)              # step scaled by gradient history
```

**Why this matters:**
- SGD treats all parameters with the same learning rate.
- Adam adapts per-parameter. Parameters with consistently large gradients take smaller steps; parameters with small gradients take larger steps.
- Adam typically converges faster and is less sensitive to the initial learning rate.

For most practical deep learning, Adam (lr=0.001 or 0.01) is the default starting point.

---

## Train / Validation Split

```python
perm = np.random.permutation(n)
split = int(0.8 * n)
train_idx, val_idx = perm[:split], perm[split:]
```

### Why hold out a validation set?

Training loss always decreases — the model is literally optimizing it. But that doesn't mean the model is getting better at generalization.

**Overfitting**: The model memorizes training data instead of learning the underlying pattern. Training loss drops, but validation loss rises.

Watching both curves tells you:
- If val loss tracks train loss → model is generalizing well
- If val loss stops improving while train loss keeps dropping → overfitting

In this module you'll see both curves converge to ~99% accuracy, confirming the model learned the actual circle structure.

---

## Training Loop Pattern

```python
for epoch in range(epochs):
    # --- Train ---
    model.train()
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # --- Evaluate ---
    model.eval()
    with torch.no_grad():
        y_pred_val = model(X_val)
        loss_val = criterion(y_pred_val, y_val)
```

Key differences from module 2.1:
- `model.train()` / `model.eval()` toggle layer behavior
- `torch.no_grad()` disables the computation graph during evaluation — saves memory and speeds up inference
- `optimizer.zero_grad()` must come before `loss.backward()` (or after `optimizer.step()`) — PyTorch accumulates gradients

---

## Saving and Loading Models

```python
# Save: only the learned parameters, not the architecture
torch.save(model.state_dict(), "mlp_circles.pth")

# Load: must recreate the architecture first
reloaded = MLP(input_dim=2, hidden_dims=[16, 8], output_dim=1)
reloaded.load_state_dict(torch.load("mlp_circles.pth", weights_only=True))
reloaded.eval()
```

**Why `state_dict()` and not `torch.save(model)`?**

Saving the full model object pickles the class definition. If you rename or move the class, the file becomes unloadable. `state_dict()` saves only the tensor values — portable and version-safe.

---

## Expected Output

```
Train: torch.Size([400, 2])  |  Val: torch.Size([100, 2])

Model architecture:
MLP(
  (net): Sequential(
    (0): Linear(in_features=2, out_features=16, bias=True)
    (1): ReLU()
    (2): Linear(in_features=16, out_features=8, bias=True)
    (3): ReLU()
    (4): Linear(in_features=8, out_features=1, bias=True)
    (5): Sigmoid()
  )
)
Total parameters: 185

Epoch   0 | Train Loss: 0.7041 | Val Loss: 0.6977 | Train Acc: 0.500 | Val Acc: 0.510
Epoch  50 | Train Loss: 0.1832 | Val Loss: 0.1801 | Train Acc: 0.935 | Val Acc: 0.940
Epoch 100 | Train Loss: 0.0521 | Val Loss: 0.0489 | Train Acc: 0.988 | Val Acc: 0.990
...
Final val accuracy: 99.0%
```

---

## What You Learned

- `nn.Module` as the standard way to define PyTorch models
- `nn.Sequential` for stacking layers cleanly
- Adam optimizer and why it adapts per-parameter
- Train/validation split to detect overfitting
- `model.train()` vs `model.eval()` and when to use each
- `torch.no_grad()` for inference efficiency
- Saving and loading model weights with `state_dict()`
- How a 2-layer MLP learns a non-linear curved decision boundary

---

## Mental Model

```
XOR (01.2 from scratch)
  → tiny 4-sample dataset
  → manual weight updates
  → proof of concept

Circles (2.2 PyTorch)
  → realistic 500-sample dataset
  → nn.Module + Adam + train/val
  → production-style training loop
  → model checkpointing
```

Next: 03-Convolutional-Networks — when input data has spatial structure (images), convolution layers replace fully-connected ones.
