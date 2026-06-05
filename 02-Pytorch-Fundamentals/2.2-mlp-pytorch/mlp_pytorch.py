import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)

# ============================================================
# Dataset: two concentric circles (not linearly separable)
# ============================================================

def make_circles(n=500, noise=0.08):
    n_half = n // 2
    theta = np.linspace(0, 2 * np.pi, n_half)
    inner = np.column_stack([0.5 * np.cos(theta), 0.5 * np.sin(theta)])
    outer = np.column_stack([1.2 * np.cos(theta), 1.2 * np.sin(theta)])
    X = np.vstack([inner, outer]) + np.random.randn(n, 2) * noise
    y = np.array([0] * n_half + [1] * n_half, dtype=np.float32)
    return X.astype(np.float32), y


X_np, y_np = make_circles()

# 80 / 20 train-val split
n = len(X_np)
perm = np.random.permutation(n)
split = int(0.8 * n)
train_idx, val_idx = perm[:split], perm[split:]

X_train = torch.tensor(X_np[train_idx])
y_train = torch.tensor(y_np[train_idx]).unsqueeze(1)
X_val   = torch.tensor(X_np[val_idx])
y_val   = torch.tensor(y_np[val_idx]).unsqueeze(1)

print(f"Train: {X_train.shape}  |  Val: {X_val.shape}")

# ============================================================
# Model: MLP via nn.Module
# ============================================================

class MLP(nn.Module):
    """Fully-connected network with configurable hidden layers."""

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


model     = MLP(input_dim=2, hidden_dims=[16, 8], output_dim=1)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

print(f"\nModel architecture:\n{model}")
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}")

# ============================================================
# Helpers
# ============================================================

def binary_accuracy(y_true, y_pred):
    preds = (y_pred >= 0.5).float()
    return (preds == y_true).float().mean().item()

# ============================================================
# Training loop
# ============================================================

epochs = 300
train_losses, val_losses = [], []
train_accs,   val_accs   = [], []

for epoch in range(epochs):
    # --- train step ---
    model.train()
    y_pred_train = model(X_train)
    loss_train   = criterion(y_pred_train, y_train)

    optimizer.zero_grad()
    loss_train.backward()
    optimizer.step()

    # --- eval step (no gradient tracking needed) ---
    model.eval()
    with torch.no_grad():
        y_pred_val = model(X_val)
        loss_val   = criterion(y_pred_val, y_val)

    train_losses.append(loss_train.item())
    val_losses.append(loss_val.item())
    train_accs.append(binary_accuracy(y_train, y_pred_train.detach()))
    val_accs.append(binary_accuracy(y_val, y_pred_val))

    if epoch % 50 == 0:
        print(f"Epoch {epoch:3d} | "
              f"Train Loss: {loss_train.item():.4f} | "
              f"Val Loss: {loss_val.item():.4f} | "
              f"Train Acc: {train_accs[-1]:.3f} | "
              f"Val Acc: {val_accs[-1]:.3f}")

print(f"\nFinal val accuracy: {val_accs[-1]*100:.1f}%")

# ============================================================
# Save & reload with state_dict
# ============================================================

torch.save(model.state_dict(), "mlp_circles.pth")
print("Model saved → mlp_circles.pth")

reloaded = MLP(input_dim=2, hidden_dims=[16, 8], output_dim=1)
reloaded.load_state_dict(torch.load("mlp_circles.pth", weights_only=True))
reloaded.eval()
print("Model reloaded successfully")

# ============================================================
# Visualisation
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 1. Decision boundary
xx, yy = np.meshgrid(np.linspace(-2, 2, 300), np.linspace(-2, 2, 300))
grid   = torch.tensor(np.c_[xx.ravel(), yy.ravel()].astype(np.float32))
with torch.no_grad():
    Z = reloaded(grid).numpy().reshape(xx.shape)

axes[0].contourf(xx, yy, Z, levels=50, cmap="RdBu", alpha=0.7)
axes[0].scatter(X_np[y_np == 0, 0], X_np[y_np == 0, 1],
                c="blue", s=10, label="Class 0 (inner)")
axes[0].scatter(X_np[y_np == 1, 0], X_np[y_np == 1, 1],
                c="red",  s=10, label="Class 1 (outer)")
axes[0].set_title("Learned Decision Boundary")
axes[0].legend(loc="upper right", fontsize=8)

# 2. Loss curves
axes[1].plot(train_losses, label="Train")
axes[1].plot(val_losses,   label="Val")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("BCE Loss")
axes[1].set_title("Loss Curves")
axes[1].legend()

# 3. Accuracy curves
axes[2].plot(train_accs, label="Train")
axes[2].plot(val_accs,   label="Val")
axes[2].set_xlabel("Epoch")
axes[2].set_ylabel("Accuracy")
axes[2].set_ylim(0, 1.05)
axes[2].set_title("Accuracy Curves")
axes[2].legend()

plt.tight_layout()
plt.savefig("mlp_circles_results.png", dpi=100)
print("Plot saved → mlp_circles_results.png")
