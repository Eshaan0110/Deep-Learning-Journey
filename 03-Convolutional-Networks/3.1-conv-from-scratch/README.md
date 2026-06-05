# 3.1 2D Convolution from Scratch

## Why Convolution?

In modules 01 and 02, every layer was **fully connected**: each output neuron receives input from every input neuron. For a 28×28 image that's 784 inputs. Add a hidden layer of 512 neurons and you already have 784 × 512 = 401,408 weights — just in the first layer.

Worse, fully-connected layers throw away spatial structure. Pixel (0,0) and pixel (1,1) are just two numbers in a flat vector; the network doesn't know they're neighbors.

Convolutional layers fix both problems:
1. **Weight sharing**: the same small kernel slides across the entire image → far fewer parameters
2. **Locality**: each output value depends only on a small spatial region (receptive field)
3. **Translation invariance**: a dog detector trained on dogs in the center also fires on dogs in the corner

---

## What Convolution Actually Does

A kernel (also called a filter) is a small matrix of learnable weights, e.g. 3×3.

```
Image patch (3×3)     Kernel (3×3)         Product (sum = output)
┌───────────────┐   ┌───────────────┐
│  1   2   3    │   │ -1  -1  -1    │
│  0   1   2    │ × │  0   0   0    │  →  sum all 9 values  →  scalar
│  3   0   1    │   │  1   1   1    │
└───────────────┘   └───────────────┘
```

The kernel slides across the image with a step size called **stride**:
- stride=1 → move one pixel at a time (dense output)
- stride=2 → skip every other position (output is half the size)

**Padding** adds zeros around the border to control output dimensions:
- No padding (valid): output shrinks by kernel_size - 1
- Same padding: output size equals input size (for stride=1)

---

## Output Size Formula

```
output_size = (input_size + 2×padding - kernel_size) / stride + 1
```

Examples with a 3×3 kernel on a 32×32 input:
- stride=1, pad=0: (32 + 0 - 3) / 1 + 1 = **30×30**
- stride=1, pad=1: (32 + 2 - 3) / 1 + 1 = **32×32**  ← "same" convolution
- stride=2, pad=1: (32 + 2 - 3) / 2 + 1 = **16×16**  ← halves spatial dims

---

## Kernels Detect Features

Hand-crafted kernels (before deep learning, people designed these manually):

**Horizontal edge detector**
```
-1  -1  -1
 0   0   0
 1   1   1
```
Fires strongly where pixel values change from dark (top) to bright (bottom).

**Vertical edge detector**
```
-1   0   1
-1   0   1
-1   0   1
```
Fires where values change left-to-right.

**Sharpen**
```
 0  -1   0
-1   5  -1
 0  -1   0
```
Amplifies the center pixel relative to its neighbors (enhances fine details).

**Blur (box filter)**
```
1/9  1/9  1/9
1/9  1/9  1/9
1/9  1/9  1/9
```
Averages each pixel with its neighbors (smooths noise).

In CNNs, the network **learns** these kernels from data instead of hand-designing them.

---

## Max Pooling

After convolution + ReLU, we apply pooling to reduce spatial size:

```
Feature map (4×4)       After MaxPool2d (2×2, stride=2) → 2×2
┌───────────────┐          ┌───────┐
│  1   3   2   0 │   max   │  3   2│
│  0   2   1   4 │  ───>   │  3   4│
│  3   1   0   2 │         └───────┘
│  0   0   3   1 │
└───────────────┘
```

Each 2×2 window → one value (the maximum). This:
- Reduces computation in deeper layers
- Makes the network slightly position-invariant (small shifts don't change which feature was detected)
- Retains the strongest activations (most detected features)

---

## The Full Stack: Conv → ReLU → Pool

```
Input: 64×64

conv2d (3×3 kernel, pad=1)  →  64×64 feature map
ReLU                         →  64×64 (zeros out negative activations)
MaxPool2d (2×2, stride=2)   →  32×32

Second conv block            →  32×32
ReLU                         →  32×32
MaxPool2d                    →  16×16

...
Flatten → Fully Connected → Output
```

Each convolution layer learns to detect different features:
- Early layers: edges, corners, color blobs
- Middle layers: textures, shapes
- Deep layers: object parts, semantic concepts

---

## Implementation Notes

The `conv2d` function here uses explicit nested loops to make the operation transparent:

```python
for i in range(h_out):
    for j in range(w_out):
        patch = image[i*stride : i*stride+kh, j*stride : j*stride+kw]
        output[i, j] = np.sum(patch * kernel)
```

PyTorch's `nn.Conv2d` does exactly this, but:
- Vectorized with C++/CUDA (thousands of times faster)
- Handles batches (N images at once)
- Handles multiple input channels (e.g., RGB → 3 channels)
- Handles multiple output channels (e.g., 16 kernels → 16 feature maps)
- Tracks gradients for backpropagation

---

## Running the Code

```bash
python conv2d.py
```

Output files:
- `conv2d_feature_maps.png`: Four kernels applied to a test image, showing the resulting feature maps before and after max pooling
- `conv2d_stride_padding.png`: Same kernel with different stride/padding configurations and their effect on output size

---

## What You Learned

- How 2D cross-correlation (convolution) is computed
- Output dimension formula with stride and padding
- Why edge-detector kernels look the way they do
- What max pooling does and why it's useful
- The intuition behind CNNs: local detection + spatial hierarchy

Next: 3.2 — build a real CNN in PyTorch, train it on MNIST, and visualize the learned filters.
