# 3.2 CNN on MNIST — PyTorch

## What We're Building

Module 3.1 showed what convolution *is* — a sliding dot product. Here we build an actual convolutional neural network in PyTorch, train it on 60,000 real handwritten digit images, and visualize what the network learns.

Expected result: **~99% accuracy** on the 10,000-image validation set in 5 epochs.

---

## Why MNIST?

MNIST (Modified National Institute of Standards and Technology) is the "hello world" of image classification:

- 70,000 grayscale images of handwritten digits 0–9
- Each image: 28×28 pixels, 1 channel
- Train split: 60,000 images, Val split: 10,000 images
- Labels: integers 0–9

It's simple enough to train quickly (minutes on CPU) but complex enough that a fully-connected network is outclassed by a CNN.

---

## Network Architecture

```
Input:   (N,  1, 28, 28)   ← N images, 1 channel, 28×28 pixels

Block 1:
  Conv2d(1→16, kernel=5, pad=2)   → (N, 16, 28, 28)   16 feature maps
  ReLU                            → (N, 16, 28, 28)   zero out negatives
  MaxPool2d(2×2, stride=2)        → (N, 16, 14, 14)   halve spatial dims

Block 2:
  Conv2d(16→32, kernel=5, pad=2)  → (N, 32, 14, 14)   32 feature maps
  ReLU                            → (N, 32, 14, 14)
  MaxPool2d(2×2, stride=2)        → (N, 32,  7,  7)   halve again

Classifier:
  Flatten                         →  N × 1568          (32 × 7 × 7)
  Linear(1568 → 128)              →  N × 128
  ReLU
  Linear(128  → 10)               →  N × 10            one logit per class
```

Total parameters: ~50,000 — remarkably few for 99% accuracy.

---

## DataLoader: Batched Training

In modules 01 and 02, we passed the entire dataset through each forward pass. At 60,000 images that's impractical.

`DataLoader` handles batching automatically:

```python
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
```

Each iteration yields `(X_batch, y_batch)` where `X_batch.shape = (64, 1, 28, 28)`.

**Why shuffle=True for training?**

If we feed images in order (all 0s, then all 1s, ...), the gradient updates are biased toward the current class. Shuffling ensures each mini-batch contains a random mix of classes, giving more representative gradient estimates.

**Why shuffle=False for validation?**

Order doesn't affect evaluation — we just compute accuracy over the full set. No need to shuffle.

---

## Loss Function: CrossEntropyLoss

For multi-class classification (10 digits) we use cross-entropy loss instead of BCE:

```
CrossEntropyLoss = -log(softmax(logits)[correct_class])
```

The model outputs 10 raw logits (one per class). Cross-entropy:
1. Applies softmax to turn logits into probabilities
2. Takes the log of the correct class probability
3. Negates it (so lower = better)

Predicting the correct class with probability 1.0 → loss = 0.
Predicting it with probability 0.1 → loss = −log(0.1) = 2.3.

PyTorch's `nn.CrossEntropyLoss` fuses softmax + log + negative mean into one numerically stable operation.

---

## Training Loop with model.train() / model.eval()

```python
def run_epoch(loader, train):
    model.train() if train else model.eval()

    with torch.enable_grad() if train else torch.no_grad():
        for X_batch, y_batch in loader:
            logits = model(X_batch)
            loss   = criterion(logits, y_batch)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
```

`torch.no_grad()` during evaluation:
- Disables the computation graph
- Saves memory (no need to store activations for backprop)
- Speeds up inference by ~30%

---

## Feature Maps: What the Network Sees

After training, we can extract intermediate activations and visualize what each conv layer responds to:

```python
with torch.no_grad():
    block1_out = model.block1(sample_img)   # 16 feature maps, 14×14
    block2_out = model.block2(block1_out)   # 32 feature maps,  7×7
```

What you'll observe:
- **Block 1** maps: edge detectors, stroke detectors, simple patterns
- **Block 2** maps: more abstract — combinations of edges, curve detectors

This spatial hierarchy (simple → complex) is the key property of deep convolutional networks.

---

## Learned Filters vs Hand-Designed Kernels

In module 3.1 we manually designed kernels (Sobel, sharpen, blur). In a CNN, those 5×5 filters are random at initialization and *learned from data* during backpropagation.

The CNN figures out which patterns are useful for digit classification on its own. Some learned filters will look like oriented edge detectors — the network independently rediscovers what computer vision researchers hand-coded for decades.

---

## Expected Output

```
Train: 60,000 samples  |  Val: 10,000 samples
Image shape: torch.Size([1, 28, 28])

Epoch | Train Loss |  Val Loss | Train Acc |  Val Acc
------------------------------------------------------
    1 |     0.1823 |    0.0589 |    0.9452 |   0.9811
    2 |     0.0517 |    0.0437 |    0.9843 |   0.9863
    3 |     0.0367 |    0.0310 |    0.9886 |   0.9900
    4 |     0.0275 |    0.0279 |    0.9913 |   0.9908
    5 |     0.0215 |    0.0268 |    0.9933 |   0.9916

Final validation accuracy: 99.16%
```

Output files:
- `cnn_predictions.png` — 10 sample images with predicted vs true labels
- `cnn_training_curves.png` — loss and accuracy over epochs
- `cnn_learned_filters.png` — all 16 learned 5×5 filters from block1
- `cnn_feature_maps.png` — block1 and block2 activations for one test image

---

## Running

```bash
# Install torchvision if not present
pip install torchvision

# Run (downloads MNIST ~11MB on first run)
python cnn_mnist.py
```

---

## What You Learned

- `nn.Conv2d` and how it maps to the manual `conv2d` from module 3.1
- `DataLoader` for efficient batched training with shuffling
- `nn.CrossEntropyLoss` for multi-class classification
- `model.train()` / `model.eval()` and `torch.no_grad()` for proper inference
- How to extract intermediate activations to visualize learned representations
- The spatial hierarchy: early layers detect edges, later layers detect abstract patterns

---

## Big Picture: The CNN Mental Model

```
Fully Connected (MLP):
  Every output neuron ← every input pixel
  Position-blind, too many parameters for images

Convolutional Network:
  Each output ← small local patch
  Same kernel slides across all positions (weight sharing)
  Early layers: local edge detectors
  Later layers: global digit shape detectors
  MaxPool: progressive spatial compression
  FC head: final classification decision
```

The same architecture pattern — conv blocks followed by a linear head — underlies ResNet, VGG, EfficientNet, and most modern vision models. You've now built the foundation.
