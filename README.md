# Deep Learning Journey

A ground-up progression through deep learning: from raw NumPy math to PyTorch CNNs, recurrent networks, and transfer learning. Each module builds directly on the previous one — no shortcuts.

## Structure

```
00-Foundation/
  numpy-from-scratch/        Linear regression, MSE/MAE/BCE/CCE losses, gradient descent

01-neural-networks-from-scratch/
  01.1-single-neuron/        Logistic regression (sigmoid neuron, BCE loss)
  01.2-mlp-from-scratch/     Multi-layer perceptron, XOR problem, backpropagation

02-Pytorch-Fundamentals/
  2.1-tensor-playground/     Tensors, autograd, computation graph, linear regression
  2.2-mlp-pytorch/           MLP via nn.Module, Adam, train/val split, model saving

03-Convolutional-Networks/
  3.1-conv-from-scratch/     2D convolution and max pooling in NumPy
  3.2-cnn-pytorch/           CNN on MNIST, DataLoader, feature map visualization

04-Sequence-Models/
  4.1-rnn-from-scratch/      Vanilla RNN, BPTT, gradient clipping, sine wave prediction
  4.2-lstm-pytorch/          LSTM with nn.LSTM, gate equations, hidden state analysis

05-Transfer-Learning/
  5.1-pretrained-features/   ResNet18 feature extraction, linear probe on CIFAR-10

tests/                       pytest suite covering loss functions, conv ops, and the RNN
```

## How to Use

Each directory has a `README.md` explaining the theory, and a Python script you can run directly.

```bash
pip install -r requirements.txt

# Examples
python 00-Foundation/numpy-from-scratch/linear_regression.py
python 01-neural-networks-from-scratch/01.2-mlp-from-scratch/mlp.py
python 02-Pytorch-Fundamentals/2.2-mlp-pytorch/mlp_pytorch.py
python 03-Convolutional-Networks/3.2-cnn-pytorch/cnn_mnist.py     # downloads MNIST on first run
python 04-Sequence-Models/4.1-rnn-from-scratch/rnn.py
python 04-Sequence-Models/4.2-lstm-pytorch/lstm_timeseries.py
python 05-Transfer-Learning/5.1-pretrained-features/transfer_learning.py  # downloads CIFAR-10

# Run tests
pytest tests/
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
| 4.1 | Vanilla RNN + BPTT | Sine wave prediction, vanishing gradient intuition |
| 4.2 | LSTM gates + `nn.LSTM` | Same task, gated memory, hidden state heatmap |
| 5.1 | Transfer learning | 70-80% CIFAR-10 accuracy with 2 000 training images |
