import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# ============================================================
# Dataset: sine wave next-step prediction
# ============================================================

T      = np.linspace(0, 6 * np.pi, 300, dtype=np.float32)
signal = np.sin(T)

SEQ_LEN = 20

X_all, y_all = [], []
for i in range(len(signal) - SEQ_LEN):
    X_all.append(signal[i : i + SEQ_LEN])
    y_all.append(signal[i + SEQ_LEN])

X_all = np.array(X_all, dtype=np.float32)  # (N, seq_len)
y_all = np.array(y_all, dtype=np.float32)  # (N,)

split   = int(0.8 * len(X_all))
X_train = X_all[:split]
y_train = y_all[:split]
X_val   = X_all[split:]
y_val   = y_all[split:]

# ============================================================
# RNN parameters
# ============================================================

INPUT_SIZE  = 1   # one scalar per time step
HIDDEN_SIZE = 32
OUTPUT_SIZE = 1


def init_params(hidden_size=HIDDEN_SIZE, scale=0.05):
    rng = np.random.default_rng(0)
    return {
        "W_xh": rng.normal(0, scale, (INPUT_SIZE, hidden_size)).astype(np.float32),
        "W_hh": rng.normal(0, scale, (hidden_size, hidden_size)).astype(np.float32),
        "b_h":  np.zeros(hidden_size, dtype=np.float32),
        "W_hy": rng.normal(0, scale, (hidden_size, OUTPUT_SIZE)).astype(np.float32),
        "b_y":  np.zeros(OUTPUT_SIZE, dtype=np.float32),
    }


# ============================================================
# Forward pass
# ============================================================

def rnn_forward(x_seq, params):
    """
    x_seq : (seq_len,) array of scalars
    Returns scalar prediction and list of hidden states h_0 … h_T.

    RNN cell:  h_t = tanh(x_t * W_xh + h_{t-1} @ W_hh + b_h)
    Output  :  y   = h_T @ W_hy + b_y
    """
    W_xh, W_hh, b_h = params["W_xh"], params["W_hh"], params["b_h"]
    W_hy, b_y = params["W_hy"], params["b_y"]

    hidden_size = W_hh.shape[0]
    h = np.zeros(hidden_size, dtype=np.float32)
    hs = [h.copy()]  # hs[0] = h_0 (initial state)

    for x_t in x_seq:
        # x_t is scalar; W_xh[0] is the weight vector for input dim 0
        h = np.tanh(x_t * W_xh[0] + h @ W_hh + b_h)
        hs.append(h.copy())

    y_pred = (hs[-1] @ W_hy + b_y).item()
    return y_pred, hs


# ============================================================
# Backpropagation Through Time (BPTT)
# ============================================================

def rnn_backward(x_seq, y_true, y_pred, hs, params, clip=1.0):
    """
    Computes gradients for MSE loss via BPTT.
    Returns grad dict and scalar loss.

    Key idea: unroll the RNN and backprop through each time step,
    accumulating dW_xh, dW_hh, db_h as we go backward.
    """
    W_hh = params["W_hh"]
    W_hy = params["W_hy"]

    loss = float((y_pred - y_true) ** 2)

    # ── Output layer ──────────────────────────────────────────
    # dL/dy_pred = 2*(y_pred - y_true)   [MSE gradient]
    dy = np.array([2.0 * (y_pred - y_true)], dtype=np.float32)  # (1,)

    dW_hy = hs[-1].reshape(-1, 1) * dy         # (hidden_size, 1)
    db_y  = dy.copy()                           # (1,)

    # Gradient flowing back into the last hidden state
    dh = (dy @ W_hy.T).squeeze()               # (hidden_size,)

    # ── Recurrent layers ──────────────────────────────────────
    dW_xh = np.zeros_like(params["W_xh"])
    dW_hh = np.zeros_like(W_hh)
    db_h  = np.zeros_like(params["b_h"])

    for t in reversed(range(len(x_seq))):
        h_t    = hs[t + 1]   # hidden state after processing step t
        h_prev = hs[t]        # hidden state before processing step t
        x_t    = x_seq[t]    # input scalar at step t

        # Gradient through tanh: d/dz tanh(z) = 1 - tanh(z)^2
        dtanh = (1.0 - h_t ** 2) * dh          # (hidden_size,)

        dW_xh[0] += x_t * dtanh                # (hidden_size,)
        dW_hh    += np.outer(h_prev, dtanh)     # (hidden_size, hidden_size)
        db_h     += dtanh                       # (hidden_size,)

        # Gradient for previous hidden state
        dh = dtanh @ W_hh.T                     # (hidden_size,)

    grads = {
        "W_xh": dW_xh, "W_hh": dW_hh, "b_h": db_h,
        "W_hy": dW_hy, "b_y": db_y,
    }
    for g in grads.values():
        np.clip(g, -clip, clip, out=g)  # gradient clipping prevents explosion

    return grads, loss


# ============================================================
# SGD update
# ============================================================

def sgd_update(params, grads, lr):
    for k in params:
        params[k] -= lr * grads[k]


# ============================================================
# Training
# ============================================================

if __name__ == "__main__":
    params = init_params()
    lr     = 0.005
    epochs = 300

    train_losses, val_losses = [], []

    print(f"{'Epoch':>5} | {'Train MSE':>9} | {'Val MSE':>8}")
    print("-" * 32)

    for epoch in range(epochs):
        idx        = np.random.permutation(len(X_train))
        total_loss = 0.0

        for i in idx:
            y_pred, hs = rnn_forward(X_train[i], params)
            grads, loss = rnn_backward(X_train[i], float(y_train[i]), y_pred, hs, params)
            sgd_update(params, grads, lr)
            total_loss += loss

        train_losses.append(total_loss / len(X_train))

        val_loss = 0.0
        for i in range(len(X_val)):
            y_p, _ = rnn_forward(X_val[i], params)
            val_loss += (y_p - float(y_val[i])) ** 2
        val_losses.append(val_loss / len(X_val))

        if epoch % 50 == 0:
            print(f"{epoch:>5} | {train_losses[-1]:>9.5f} | {val_losses[-1]:>8.5f}")

    print(f"\nFinal val MSE: {val_losses[-1]:.5f}")

    # ============================================================
    # Visualisation 1: loss curves
    # ============================================================

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(train_losses, label="Train MSE")
    axes[0].plot(val_losses,   label="Val MSE")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE")
    axes[0].set_title("RNN Training Loss (Sine Wave Prediction)")
    axes[0].legend()

    # ============================================================
    # Visualisation 2: predictions vs ground truth on val set
    # ============================================================

    preds = [rnn_forward(X_val[i], params)[0] for i in range(len(X_val))]
    t_val = T[split + SEQ_LEN:]

    axes[1].plot(t_val, y_val,          label="Ground truth", alpha=0.8)
    axes[1].plot(t_val, preds,          label="RNN prediction", linestyle="--")
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("sin(t)")
    axes[1].set_title("Next-step Sine Wave Prediction")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("rnn_results.png", dpi=100)
    print("Plot saved → rnn_results.png")
