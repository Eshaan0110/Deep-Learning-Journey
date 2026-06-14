# 04.2 — LSTM in PyTorch

Same task as 04.1 (sine wave next-step prediction), now using PyTorch's `nn.LSTM`. The LSTM's gated cell state solves the vanishing gradient problem that limits vanilla RNNs on longer sequences.

## Why LSTM?

A vanilla RNN propagates gradients through repeated matrix multiplication:

```
∂L/∂h_0 ∝ W_hh^T × W_hh^T × … × W_hh^T   (T times)
```

If the largest eigenvalue of `W_hh` < 1, this product → 0 (vanishing). If > 1, → ∞ (exploding).

LSTM replaces this with a **cell state** `c_t` that accumulates information via **addition**, not multiplication:

```
c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
```

The gradient of `c_t` with respect to `c_{t-1}` is `f_t` (the forget gate). When `f_t ≈ 1`, the gradient flows back undiluted — the LSTM can remember across hundreds of steps.

## LSTM Cell Equations

```
f_t = σ(W_f · [h_{t-1}, x_t] + b_f)      forget gate  (what to erase from c)
i_t = σ(W_i · [h_{t-1}, x_t] + b_i)      input gate   (what to write to c)
g_t = tanh(W_g · [h_{t-1}, x_t] + b_g)   candidate    (what to write)
o_t = σ(W_o · [h_{t-1}, x_t] + b_o)      output gate  (what to expose as h)

c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t          cell state update
h_t = o_t ⊙ tanh(c_t)                     hidden state
```

Four gate weight matrices vs. one in a vanilla RNN — that's why LSTMs have ~4× more parameters.

## Architecture

```
Input (batch, 20, 1)
     ↓
nn.LSTM(input_size=1, hidden_size=32)
     ↓
Last hidden state h_T  (batch, 32)
     ↓
nn.Linear(32, 1)
     ↓
Predicted next value  (batch, 1)
```

## PyTorch API Notes

```python
lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
output, (h_n, c_n) = lstm(x)
# output : (batch, seq_len, hidden_size) — every time step's h_t
# h_n    : (num_layers, batch, hidden_size) — final hidden state
# c_n    : (num_layers, batch, hidden_size) — final cell state
```

## Run

```bash
python lstm_timeseries.py
```

Plots saved: `lstm_timeseries_results.png`, `lstm_hidden_states.png`.
