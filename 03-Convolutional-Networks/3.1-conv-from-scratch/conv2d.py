import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# ============================================================
# Core convolution operations from scratch
# ============================================================

def conv2d(image, kernel, stride=1, padding=0):
    """
    2D cross-correlation.  (Deep learning calls this 'convolution',
    but technically skips the kernel flip that true convolution does.)

    Steps at each position (i, j):
      1. Extract the patch: image[i:i+kH, j:j+kW]
      2. Element-wise multiply patch * kernel
      3. Sum → one scalar output value
    """
    if padding > 0:
        image = np.pad(image, padding, mode='constant')

    h_in, w_in = image.shape
    kh, kw     = kernel.shape
    h_out = (h_in - kh) // stride + 1
    w_out = (w_in - kw) // stride + 1

    output = np.zeros((h_out, w_out))
    for i in range(h_out):
        for j in range(w_out):
            patch = image[i*stride : i*stride+kh,
                          j*stride : j*stride+kw]
            output[i, j] = np.sum(patch * kernel)
    return output


def max_pool2d(feature_map, pool_size=2, stride=2):
    """
    Keeps the maximum activation in each pool_size × pool_size window.
    Reduces spatial dimensions by factor of stride.
    """
    h, w   = feature_map.shape
    h_out  = (h - pool_size) // stride + 1
    w_out  = (w - pool_size) // stride + 1

    output = np.zeros((h_out, w_out))
    for i in range(h_out):
        for j in range(w_out):
            patch = feature_map[i*stride : i*stride+pool_size,
                                 j*stride : j*stride+pool_size]
            output[i, j] = np.max(patch)
    return output


def avg_pool2d(feature_map, pool_size=2, stride=2):
    """Average pooling: takes the mean instead of the max."""
    h, w   = feature_map.shape
    h_out  = (h - pool_size) // stride + 1
    w_out  = (w - pool_size) // stride + 1

    output = np.zeros((h_out, w_out))
    for i in range(h_out):
        for j in range(w_out):
            patch = feature_map[i*stride : i*stride+pool_size,
                                 j*stride : j*stride+pool_size]
            output[i, j] = np.mean(patch)
    return output


def relu(x):
    return np.maximum(0, x)


def output_size(input_size, kernel_size, stride=1, padding=0):
    return (input_size + 2*padding - kernel_size) // stride + 1


# ============================================================
# Classic hand-designed kernels
# ============================================================

KERNELS = {
    "Horizontal edges": np.array([
        [-1, -1, -1],
        [ 0,  0,  0],
        [ 1,  1,  1],
    ], dtype=np.float32),

    "Vertical edges": np.array([
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1],
    ], dtype=np.float32),

    "Sharpen": np.array([
        [ 0, -1,  0],
        [-1,  5, -1],
        [ 0, -1,  0],
    ], dtype=np.float32),

    "Blur": np.ones((3, 3), dtype=np.float32) / 9,
}


# ============================================================
# Synthetic test image
# ============================================================

def make_test_image(size=64):
    img = np.zeros((size, size), dtype=np.float32)
    img[12:52, 12:52] = 1.0          # filled square
    img[20:44, 20:44] = 0.0          # hollow it out
    img[10:54, 30:34] = 0.8          # vertical stripe
    img[30:34, 10:54] = 0.8          # horizontal stripe
    return img


if __name__ == "__main__":
    image = make_test_image(64)

    # ============================================================
    # Step-by-step walkthrough with a tiny example
    # ============================================================

    print("=" * 55)
    print(" Step-by-step 2D convolution walkthrough (4×4 → 2×2)")
    print("=" * 55)

    tiny = np.array([
        [1, 2, 3, 0],
        [0, 1, 2, 3],
        [3, 0, 1, 2],
        [2, 3, 0, 1],
    ], dtype=np.float32)

    edge_k = np.array([
        [-1, -1, -1],
        [ 0,  0,  0],
        [ 1,  1,  1],
    ], dtype=np.float32)

    result = conv2d(tiny, edge_k)
    print("\nInput (4×4):")
    print(tiny.astype(int))
    print("\nHorizontal-edge kernel (3×3):")
    print(edge_k.astype(int))
    print("\nFeature map (stride=1, no padding):")
    print(result)
    print(f"\nDimension formula: ({tiny.shape[0]} - {edge_k.shape[0]}) // 1 + 1 = {result.shape[0]}")

    print("\n--- Position (0,0) computed manually ---")
    patch = tiny[0:3, 0:3]
    print("Patch:\n", patch.astype(int))
    print("kernel:\n", edge_k.astype(int))
    print("Element-wise product:\n", (patch * edge_k).astype(int))
    print(f"Sum (= feature map value at [0,0]): {np.sum(patch * edge_k):.1f}")

    # ============================================================
    # Demonstrate what each kernel detects
    # ============================================================

    print("\n\nDimension changes through a typical conv → relu → pool stack:")
    h, w = image.shape
    print(f"  Input:         {h}×{w}")
    fm   = conv2d(image, KERNELS["Horizontal edges"], padding=1)
    print(f"  After conv2d (3×3 kernel, pad=1): {fm.shape[0]}×{fm.shape[1]}")
    act  = relu(fm)
    pool = max_pool2d(act)
    print(f"  After max_pool2d (2×2, stride=2): {pool.shape[0]}×{pool.shape[1]}")

    # ============================================================
    # Visualisation: kernels and their feature maps
    # ============================================================

    n_kernels = len(KERNELS)
    fig, axes = plt.subplots(3, n_kernels, figsize=(4 * n_kernels, 9))

    for col, (name, kernel) in enumerate(KERNELS.items()):
        # Row 0: kernel weights
        axes[0, col].imshow(kernel, cmap="RdBu_r", vmin=-2, vmax=2)
        axes[0, col].set_title(f'"{name}"\nkernel', fontsize=9)
        for i in range(kernel.shape[0]):
            for j in range(kernel.shape[1]):
                axes[0, col].text(j, i, f"{kernel[i,j]:.0f}",
                                  ha='center', va='center', fontsize=8)
        axes[0, col].axis("off")

        # Row 1: raw feature map
        fm = conv2d(image, kernel, padding=1)
        axes[1, col].imshow(fm, cmap="gray")
        axes[1, col].set_title(f"Feature map\n{fm.shape}", fontsize=9)
        axes[1, col].axis("off")

        # Row 2: after ReLU + MaxPool
        pooled = max_pool2d(relu(fm))
        axes[2, col].imshow(pooled, cmap="gray")
        axes[2, col].set_title(f"ReLU + MaxPool\n{pooled.shape}", fontsize=9)
        axes[2, col].axis("off")

    fig.text(0.01, 0.83, "Kernels",      va='center', rotation=90, fontsize=10, fontweight='bold')
    fig.text(0.01, 0.50, "Feature Maps", va='center', rotation=90, fontsize=10, fontweight='bold')
    fig.text(0.01, 0.17, "After Pool",   va='center', rotation=90, fontsize=10, fontweight='bold')

    plt.suptitle("2D Convolution from Scratch — How Different Kernels Detect Features",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig("conv2d_feature_maps.png", dpi=100, bbox_inches="tight")
    print("\nPlot saved → conv2d_feature_maps.png")

    # ============================================================
    # Visualisation 2: effect of stride and padding
    # ============================================================

    fig2, axes2 = plt.subplots(1, 4, figsize=(16, 4))
    axes2[0].imshow(image, cmap="gray")
    axes2[0].set_title("Original (64×64)")
    axes2[0].axis("off")

    configs = [
        {"stride": 1, "padding": 0, "label": "stride=1, pad=0"},
        {"stride": 1, "padding": 1, "label": "stride=1, pad=1 (same size)"},
        {"stride": 2, "padding": 1, "label": "stride=2, pad=1 (half size)"},
    ]
    kernel = KERNELS["Vertical edges"]
    for i, cfg in enumerate(configs, 1):
        fm = conv2d(image, kernel, stride=cfg["stride"], padding=cfg["padding"])
        axes2[i].imshow(relu(fm), cmap="gray")
        axes2[i].set_title(f"{cfg['label']}\n→ {fm.shape[0]}×{fm.shape[1]}")
        axes2[i].axis("off")

    plt.suptitle("Effect of Stride and Padding on Output Dimensions", fontsize=12)
    plt.tight_layout()
    plt.savefig("conv2d_stride_padding.png", dpi=100)
    print("Plot saved → conv2d_stride_padding.png")
