# Deep Learning Journey

A ground-up progression through deep learning: from raw NumPy math to PyTorch CNNs. Each module builds directly on the previous one — no shortcuts.

## Structure

```
00-Foundation/
  numpy-from-scratch/        Linear regression, MSE loss, gradient descent in NumPy

01-neural-networks-from-scratch/
  01.1-single-neuron/        Logistic regression (sigmoid neuron, BCE loss)
  01.2-mlp-from-scratch/     Multi-layer perceptron, XOR problem, backpropagation

02-Pytorch-Fundamentals/
  2.1-tensor-playground/     Tensors, autograd, computation graph, linear regression
  2.2-mlp-pytorch/           MLP via nn.Module, Adam, train/val split, model saving

03-Convolutional-Networks/
  3.1-conv-from-scratch/     2D convolution and max pooling in NumPy
  3.2-cnn-pytorch/           CNN on MNIST, DataLoader, feature map visualization
```

## How to Use

Each directory has a `README.md` explaining the theory, and a Python script you can run directly.

```bash
pip install -r requirements.txt

# Examples
python 00-Foundation/numpy-from-scratch/linear_regression.py
python 02-Pytorch-Fundamentals/2.2-mlp-pytorch/mlp_pytorch.py
python 03-Convolutional-Networks/3.2-cnn-pytorch/cnn_mnist.py   # downloads MNIST on first run
```

## Progression

| Module | Core concept | Key output |
|--------|-------------|------------|
| 00 | Gradient descent from scratch | Loss curve converging to true slope |
| 01.1 | Sigmoid neuron + BCE | Decision boundary on synthetic data |
| 01.2 | Backprop through layers | MLP solving XOR |
| 2.1 | PyTorch autograd | Computation graph, `loss.backward()` |
| 2.2 | `nn.Module` + Adam | 99% accuracy on two-circles problem |
| 3.1 | 2D convolution math | Edge/blur/sharpen feature maps |
| 3.2 | Full CNN pipeline | 99% accuracy on MNIST in 5 epochs |
