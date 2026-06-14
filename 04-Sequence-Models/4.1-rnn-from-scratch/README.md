# 04.1 — Vanilla RNN from Scratch

A complete recurrent neural network built in NumPy: forward pass, Backpropagation Through Time (BPTT), gradient clipping, and sine wave prediction.

## The RNN Cell

At every time step t, the RNN updates a hidden state vector:

```
h_t = tanh(x_t · W_xh + h_{t-1} · W_hh + b_h)
y   = h_T · W_hy + b_y          (output from last step)
```

- `x_t` — input at step t (scalar here, generally a vector)
- `h_t` — hidden state: the RNN's "memory" of everything seen so far
- `W_hh` — recurrent weight matrix: lets the network pass information across steps
- `tanh` — squashes the hidden state to [-1, 1], preventing unbounded growth

## Backpropagation Through Time (BPTT)

Training means unrolling the RNN across all T time steps and running standard backprop through the unrolled graph:

```
Loss = (y_pred - y_true)²

∂L/∂W_hy = h_T · ∂L/∂y
∂L/∂h_T  = W_hy · ∂L/∂y

For t = T-1 … 0:
    δ_t  = (1 - h_t²) · ∂L/∂h_t        ← tanh derivative
    ∂L/∂W_xh += x_t · δ_t
    ∂L/∂W_hh += h_{t-1} · δ_t
    ∂L/∂h_{t-1} = W_hh · δ_t           ← gradient to previous step
```

## Vanishing Gradients

Each step multiplies the gradient by `W_hh · (1 - h²)`. Over long sequences, this product shrinks toward zero — the network stops learning from distant inputs. This is the **vanishing gradient problem**.

Gradient clipping (`clip = 1.0`) prevents the opposite: **exploding gradients** where the product grows uncontrollably. Both are symptoms of repeated matrix multiplication across time.

LSTM (next module) solves vanishing gradients with gated additive updates instead of multiplicative ones.

## Task: Sine Wave Next-Step Prediction

Given 20 previous values of sin(t), predict sin(t+1).

- Sequence length: 20
- Hidden size: 32
- Optimizer: SGD with gradient clipping
- Training samples: ~240

## Run

```bash
python rnn.py
```

Expected: val MSE drops below 0.01 by epoch 200. Plot saved as `rnn_results.png`.
