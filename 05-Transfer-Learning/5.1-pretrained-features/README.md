# 05.1 — Transfer Learning: Feature Extraction with ResNet18

Use a ResNet18 pretrained on ImageNet as a fixed feature extractor, then train only a single linear layer on CIFAR-10. With only 2 000 training images (vs. the full 50 000), the pretrained backbone gives 70-80% accuracy where training from scratch would barely reach 40%.

## The Core Idea

ImageNet training forces a network to learn visual primitives that are useful for almost any image task: edges, textures, object parts, shapes. Those 512 features from the final ResNet18 layer are not "ImageNet features" — they are general-purpose vision features.

```
CIFAR-10 image (32×32)
       ↓  resize to 224×224
ResNet18 backbone (frozen, pretrained on ImageNet)
       ↓  extract 512-d feature vector
Linear(512 → 10)  ← the only layer we train
       ↓
Class logits → Cross-entropy loss
```

## Two Strategies

| Strategy | What trains | When to use |
|---|---|---|
| **Feature extraction** | Only the new head | Small dataset, fast, works well |
| **Fine-tuning** | Head + some backbone layers | More data, slower, higher ceiling |

This module implements feature extraction. Fine-tuning adds a small learning rate to backbone layers and generally improves accuracy by 3-10% on the cost of longer training.

## Why Precompute Features?

Running the ResNet18 forward pass for every batch in every epoch wastes time when the backbone is frozen. We can run it once, cache the 512-d vectors, and train the linear classifier on those — which is just a matrix multiply per batch.

## Run

```bash
python transfer_learning.py
```

Downloads CIFAR-10 on first run (~170 MB). Feature extraction takes 1-2 minutes on CPU. Linear probe trains in under 10 seconds.

Plot saved as `transfer_learning_results.png`.
