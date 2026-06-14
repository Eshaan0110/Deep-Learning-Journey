import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)
np.random.seed(42)

# ============================================================
# Dataset: noisy sine wave next-step prediction
# (Same task as 4.1 — compare LSTM vs vanilla RNN)
# ============================================================

T      = np.linspace(0, 6 * np.pi, 300, dtype=np.float32)
signal = (np.sin(T) + 0.1 * np.random.randn(len(T))).astype(np.float32)

SEQ_LEN = 20

X_all, y_all = [], []
for i in range(len(signal) - SEQ_LEN):
    X_all.append(signal[i : i + SEQ_LEN])
    y_all.append(signal[i + SEQ_LEN])

X_all = np.array(X_all, dtype=np.float32)
y_all = np.array(y_all, dtype=np.float32)

split   = int(0.8 * len(X_all))
X_train = torch.tensor(X_all[:split]).unsqueeze(-1)   # (N, seq_len, 1)
y_train = torch.tensor(y_all[:split]).unsqueeze(-1)   # (N, 1)
X_val   = torch.tensor(X_all[split:]).unsqueeze(-1)
y_val   = torch.tensor(y_all[split:]).unsqueeze(-1)

print(f"Train: {X_train.shape}  |  Val: {X_val.shape}")

# ============================================================
# Model: LSTM → Linear head
# ============================================================

class LSTMPredictor(nn.Module):
    """
    Wraps nn.LSTM for scalar sequence-to-scalar prediction.

    LSTM cell equations (per time step):
        f_t = σ(W_f · [h_{t-1}, x_t] + b_f)     forget gate
        i_t = σ(W_i · [h_{t-1}, x_t] + b_i)     input gate
        g_t = tanh(W_g · [h_{t-1}, x_t] + b_g)  candidate cell
        o_t = σ(W_o · [h_{t-1}, x_t] + b_o)     output gate

        c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t         cell state (memory)
        h_t = o_t ⊙ tanh(c_t)                    hidden state

    The cell state c_t uses ADDITION, not multiplication, so gradients
    flow back through time without vanishing — unlike vanilla RNN.
    """

    def __init__(self, hidden_size=32, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,    # input shape: (batch, seq_len, features)
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, seq_len, 1)
        # lstm_out: (batch, seq_len, hidden_size) — all hidden states
        # h_n: (num_layers, batch, hidden_size) — last hidden state
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]   # use final-step hidden state
        return self.head(last_hidden)      # (batch, 1)


model     = LSTMPredictor(hidden_size=32, num_layers=1)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

total_params = sum(p.numel() for p in model.parameters())
print(f"\nModel parameters: {total_params}")
print(f"Model:\n{model}\n")

# ============================================================
# DataLoaders
# ============================================================

train_ds     = TensorDataset(X_train, y_train)
val_ds       = TensorDataset(X_val,   y_val)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False)

# ============================================================
# Training loop
# ============================================================

epochs = 150
train_losses, val_losses = [], []

print(f"{'Epoch':>5} | {'Train MSE':>9} | {'Val MSE':>8}")
print("-" * 32)

for epoch in range(1, epochs + 1):
    model.train()
    total, n = 0.0, 0
    for X_b, y_b in train_loader:
        pred = model(X_b)
        loss = criterion(pred, y_b)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total += loss.item() * len(y_b)
        n     += len(y_b)
    train_losses.append(total / n)

    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for X_b, y_b in val_loader:
            pred  = model(X_b)
            total += criterion(pred, y_b).item() * len(y_b)
            n     += len(y_b)
    val_losses.append(total / n)

    if epoch % 25 == 0:
        print(f"{epoch:>5} | {train_losses[-1]:>9.5f} | {val_losses[-1]:>8.5f}")

print(f"\nFinal val MSE: {val_losses[-1]:.5f}")

# ============================================================
# Save model
# ============================================================

torch.save(model.state_dict(), "lstm_timeseries.pth")
print("Model saved → lstm_timeseries.pth")

# ============================================================
# Visualisation 1: loss curves
# ============================================================

model.eval()
with torch.no_grad():
    val_preds = model(X_val).squeeze().numpy()

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(train_losses, label="Train MSE")
axes[0].plot(val_losses,   label="Val MSE")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("MSE")
axes[0].set_title("LSTM Training Loss (Sine Wave Prediction)")
axes[0].legend()

# ============================================================
# Visualisation 2: predictions vs ground truth
# ============================================================

t_val     = T[split + SEQ_LEN:]
y_val_np  = y_val.squeeze().numpy()

axes[1].plot(t_val, y_val_np,   label="Ground truth", alpha=0.8)
axes[1].plot(t_val, val_preds,  label="LSTM prediction", linestyle="--")
axes[1].set_xlabel("t")
axes[1].set_ylabel("signal")
axes[1].set_title("Next-step Prediction (Noisy Sine Wave)")
axes[1].legend()

plt.tight_layout()
plt.savefig("lstm_timeseries_results.png", dpi=100)
print("Plot saved → lstm_timeseries_results.png")

# ============================================================
# Visualisation 3: LSTM gates analysis on one sequence
# ============================================================

sample = X_val[:1]   # (1, seq_len, 1)

# Register hooks to capture gate activations
gate_activations = {}

def hook_factory(name):
    def hook(module, input, output):
        # output is (lstm_out, (h_n, c_n))
        # We capture the cell state trajectory from lstm_out
        gate_activations[name] = output
    return hook

handle = model.lstm.register_forward_hook(hook_factory("lstm"))

model.eval()
with torch.no_grad():
    _ = model(sample)

handle.remove()

lstm_out_sample = gate_activations["lstm"][0].squeeze().numpy()   # (seq_len, hidden_size)
c_n_sample      = gate_activations["lstm"][1][1].squeeze().numpy()  # (hidden_size,)

fig2, ax = plt.subplots(figsize=(10, 4))
im = ax.imshow(lstm_out_sample.T, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xlabel("Time step")
ax.set_ylabel("Hidden unit")
ax.set_title("LSTM Hidden State Activations Across Time (first val sequence)")
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig("lstm_hidden_states.png", dpi=100)
print("Plot saved → lstm_hidden_states.png")
