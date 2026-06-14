import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset, TensorDataset
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================================
# Dataset: CIFAR-10 (2 000 train, 500 val)
# Images are 32×32; ResNet expects 224×224 — we resize on load.
# ============================================================

transform = T.Compose([
    T.Resize(224),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # ImageNet stats
])

train_full = torchvision.datasets.CIFAR10(
    root="./data", train=True,  download=True, transform=transform
)
val_full = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform
)

rng       = np.random.default_rng(42)
train_idx = rng.choice(len(train_full), 2000, replace=False)
val_idx   = rng.choice(len(val_full),    500, replace=False)

train_subset = Subset(train_full, train_idx)
val_subset   = Subset(val_full,   val_idx)

print(f"Train subset: {len(train_subset)}  |  Val subset: {len(val_subset)}")
print(f"Classes: {train_full.classes}")

# ============================================================
# Backbone: ResNet18 pretrained on ImageNet
# We remove the final fc layer to expose the 512-d feature vector.
# ============================================================

backbone = torchvision.models.resnet18(
    weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1
)
backbone.fc = nn.Identity()   # output: (batch, 512)
backbone = backbone.to(device).eval()

for p in backbone.parameters():
    p.requires_grad = False   # frozen — no gradients through backbone

print(f"\nBackbone: ResNet18 pretrained on ImageNet (backbone frozen)")

# ============================================================
# Precompute features once — much faster than running backbone
# through every training batch.
# ============================================================

def extract_features(dataset, batch_size=64):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    feats, labels = [], []
    with torch.no_grad():
        for X, y in loader:
            feats.append(backbone(X.to(device)).cpu())
            labels.append(y)
    return torch.cat(feats), torch.cat(labels)


print("\nExtracting training features …")
train_feats, train_labels = extract_features(train_subset)
print("Extracting validation features …")
val_feats,   val_labels   = extract_features(val_subset)
print(f"Feature shape: {train_feats.shape}  (samples × 512 ImageNet features)")

# ============================================================
# Linear probe: train only a Linear(512 → 10) on frozen features
# ============================================================

train_ds     = TensorDataset(train_feats, train_labels)
val_ds       = TensorDataset(val_feats,   val_labels)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False)

classifier = nn.Linear(512, 10).to(device)
optimizer  = torch.optim.Adam(classifier.parameters(), lr=0.001)
criterion  = nn.CrossEntropyLoss()

epochs = 50
train_accs, val_accs = [], []

print(f"\n{'Epoch':>5} | {'Train Acc':>9} | {'Val Acc':>7}")
print("-" * 32)

for epoch in range(1, epochs + 1):
    classifier.train()
    correct, total = 0, 0
    for feats, labels in train_loader:
        feats, labels = feats.to(device), labels.to(device)
        out  = classifier(feats)
        loss = criterion(out, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        correct += (out.argmax(1) == labels).sum().item()
        total   += len(labels)
    train_accs.append(correct / total)

    classifier.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for feats, labels in val_loader:
            feats, labels = feats.to(device), labels.to(device)
            correct += (classifier(feats).argmax(1) == labels).sum().item()
            total   += len(labels)
    val_accs.append(correct / total)

    if epoch % 10 == 0:
        print(f"{epoch:>5} | {train_accs[-1]:>9.3f} | {val_accs[-1]:>7.3f}")

print(f"\nFinal val accuracy: {val_accs[-1]*100:.1f}%  (random baseline: 10%)")

# ============================================================
# Save classifier head
# ============================================================

torch.save(classifier.state_dict(), "transfer_classifier.pth")
print("Classifier saved → transfer_classifier.pth")

# ============================================================
# Visualisation 1: accuracy curves
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot([a * 100 for a in train_accs], label="Train")
axes[0].plot([a * 100 for a in val_accs],   label="Val")
axes[0].axhline(10, color="gray", linestyle="--", label="Random (10%)")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy (%)")
axes[0].set_title("Linear Probe on Frozen ResNet18 Features\n(2 000 training images, CIFAR-10)")
axes[0].legend()

# ============================================================
# Visualisation 2: per-class accuracy on val set
# ============================================================

classifier.eval()
all_preds, all_true = [], []
with torch.no_grad():
    for feats, labels in val_loader:
        all_preds.append(classifier(feats.to(device)).argmax(1).cpu())
        all_true.append(labels)
all_preds = torch.cat(all_preds)
all_true  = torch.cat(all_true)

per_class = [
    (all_preds[all_true == c] == c).float().mean().item()
    for c in range(10)
]

axes[1].bar(train_full.classes, per_class, color="steelblue")
axes[1].set_ylim(0, 1)
axes[1].set_ylabel("Accuracy")
axes[1].set_title("Per-class Val Accuracy\n(ResNet18 backbone, linear head)")
axes[1].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.savefig("transfer_learning_results.png", dpi=100)
print("Plot saved → transfer_learning_results.png")

# ============================================================
# Summary
# ============================================================

print("\n" + "=" * 50)
print("Transfer learning result summary")
print("=" * 50)
print(f"  Backbone      : ResNet18 pretrained on ImageNet")
print(f"  Probe         : Linear(512 → 10), trained on CIFAR-10 subset")
print(f"  Training set  : 2 000 samples (vs 50 000 full CIFAR-10)")
print(f"  Final val acc : {val_accs[-1]*100:.1f}%")
print(f"  Random chance : 10.0%")
print(f"\n  ImageNet features generalise to CIFAR-10 without")
print(f"  fine-tuning the backbone at all.")
