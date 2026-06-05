import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(42)

# ============================================================
# Dataset: MNIST handwritten digits (0–9)
# Downloads ~11 MB on first run, cached in ./data afterwards
# ============================================================

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),  # MNIST global mean & std
])

train_dataset = torchvision.datasets.MNIST(
    root="./data", train=True,  download=True, transform=transform
)
val_dataset = torchvision.datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_dataset,   batch_size=64, shuffle=False, num_workers=0)

print(f"Train: {len(train_dataset):,} samples  |  Val: {len(val_dataset):,} samples")
print(f"Image shape: {train_dataset[0][0].shape}  (channels × H × W)")
print(f"Classes: {train_dataset.classes}")

# ============================================================
# Model: two conv blocks + fully-connected head
#
#   Input  (N,  1, 28, 28)
#   Conv1  (N, 16, 28, 28)  ← padding=2 keeps same spatial size
#   ReLU
#   Pool   (N, 16, 14, 14)  ← MaxPool halves H and W
#   Conv2  (N, 32, 14, 14)
#   ReLU
#   Pool   (N, 32,  7,  7)
#   Flatten → 32*7*7 = 1568
#   Linear  1568 → 128
#   ReLU
#   Linear  128  → 10       ← one logit per class
# ============================================================

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return self.head(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")

model     = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}\n")

# ============================================================
# Training helpers
# ============================================================

def run_epoch(loader, train):
    model.train() if train else model.eval()
    total_loss = 0.0
    correct    = 0
    total      = 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch)
            loss   = criterion(logits, y_batch)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(y_batch)
            correct    += (logits.argmax(dim=1) == y_batch).sum().item()
            total      += len(y_batch)

    return total_loss / total, correct / total

# ============================================================
# Training loop
# ============================================================

epochs = 5
train_losses, val_losses = [], []
train_accs,   val_accs   = [], []

print(f"{'Epoch':>5} | {'Train Loss':>10} | {'Val Loss':>8} | {'Train Acc':>9} | {'Val Acc':>7}")
print("-" * 54)

for epoch in range(1, epochs + 1):
    tr_loss, tr_acc = run_epoch(train_loader, train=True)
    vl_loss, vl_acc = run_epoch(val_loader,   train=False)

    train_losses.append(tr_loss)
    val_losses.append(vl_loss)
    train_accs.append(tr_acc)
    val_accs.append(vl_acc)

    print(f"{epoch:>5} | {tr_loss:>10.4f} | {vl_loss:>8.4f} | "
          f"{tr_acc:>9.4f} | {vl_acc:>7.4f}")

print(f"\nFinal validation accuracy: {val_accs[-1]*100:.2f}%")

# ============================================================
# Save model
# ============================================================

torch.save(model.state_dict(), "cnn_mnist.pth")
print("Model saved → cnn_mnist.pth")

# ============================================================
# Visualisation 1: Sample predictions
# ============================================================

model.eval()
images, labels = next(iter(val_loader))
with torch.no_grad():
    logits = model(images.to(device))
    preds  = logits.argmax(dim=1).cpu()

fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    img = images[i].squeeze().numpy()
    ax.imshow(img, cmap="gray", interpolation="nearest")
    color = "green" if preds[i] == labels[i] else "red"
    ax.set_title(f"pred={preds[i].item()}  true={labels[i].item()}",
                 color=color, fontsize=9)
    ax.axis("off")
plt.suptitle("Sample Predictions (green = correct, red = wrong)", fontsize=11)
plt.tight_layout()
plt.savefig("cnn_predictions.png", dpi=100)
print("Plot saved → cnn_predictions.png")

# ============================================================
# Visualisation 2: Training curves
# ============================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

ax1.plot(range(1, epochs+1), train_losses, "o-", label="Train")
ax1.plot(range(1, epochs+1), val_losses,   "o-", label="Val")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Cross-Entropy Loss")
ax1.set_title("Loss Curves")
ax1.legend()

ax2.plot(range(1, epochs+1), [a*100 for a in train_accs], "o-", label="Train")
ax2.plot(range(1, epochs+1), [a*100 for a in val_accs],   "o-", label="Val")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy (%)")
ax2.set_title("Accuracy Curves")
ax2.legend()

plt.tight_layout()
plt.savefig("cnn_training_curves.png", dpi=100)
print("Plot saved → cnn_training_curves.png")

# ============================================================
# Visualisation 3: Learned conv1 filters
# ============================================================

filters = model.block1[0].weight.detach().cpu().numpy()  # (16, 1, 5, 5)
n_filters = filters.shape[0]

fig, axes = plt.subplots(2, n_filters // 2, figsize=(12, 4))
for i, ax in enumerate(axes.flat):
    fmin, fmax = filters[i, 0].min(), filters[i, 0].max()
    f_norm = (filters[i, 0] - fmin) / (fmax - fmin + 1e-8)
    ax.imshow(f_norm, cmap="viridis", interpolation="nearest")
    ax.set_title(f"Filter {i+1}", fontsize=8)
    ax.axis("off")

plt.suptitle("Learned Conv1 Filters (16 filters, 5×5 each)", fontsize=11)
plt.tight_layout()
plt.savefig("cnn_learned_filters.png", dpi=100)
print("Plot saved → cnn_learned_filters.png")

# ============================================================
# Visualisation 4: Feature maps for one test image
# ============================================================

sample_img = val_dataset[7][0].unsqueeze(0).to(device)  # (1, 1, 28, 28)

model.eval()
with torch.no_grad():
    block1_out = model.block1(sample_img)               # (1, 16, 14, 14)
    block2_out = model.block2(block1_out)               # (1, 32,  7,  7)

b1_maps = block1_out.squeeze(0).cpu().numpy()  # (16, 14, 14)
b2_maps = block2_out.squeeze(0).cpu().numpy()  # (32,  7,  7)

fig, axes = plt.subplots(3, 8, figsize=(16, 6))

# Row 0: original image (top-left) + first 7 block1 feature maps
axes[0, 0].imshow(sample_img.squeeze().cpu().numpy(), cmap="gray")
axes[0, 0].set_title(f"Input\ntrue={val_dataset[7][1]}", fontsize=8)
axes[0, 0].axis("off")
for i in range(1, 8):
    axes[0, i].imshow(b1_maps[i-1], cmap="viridis")
    axes[0, i].set_title(f"Conv1\nmap {i}", fontsize=8)
    axes[0, i].axis("off")

# Row 1: block1 feature maps 8–15
for i in range(8):
    axes[1, i].imshow(b1_maps[8+i], cmap="viridis")
    axes[1, i].set_title(f"Conv1\nmap {9+i}", fontsize=8)
    axes[1, i].axis("off")

# Row 2: first 8 block2 feature maps
for i in range(8):
    axes[2, i].imshow(b2_maps[i], cmap="viridis")
    axes[2, i].set_title(f"Conv2\nmap {i+1}", fontsize=8)
    axes[2, i].axis("off")

plt.suptitle("Feature Maps Through the CNN Layers", fontsize=11)
plt.tight_layout()
plt.savefig("cnn_feature_maps.png", dpi=100)
print("Plot saved → cnn_feature_maps.png")
