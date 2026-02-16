import torch

print(torch.__version__)


#create tensor

# x = torch.tensor([[1, 2, 3],[4, 5, 6]])

# print(x)
# print("data type of x:", x.dtype)
# print("shape of x:", x.shape)
# print("device of x:", x.device)


#require gradient example

w = torch.tensor([2.0], requires_grad=True)
b = torch.tensor([1.0], requires_grad=True)
x = torch.tensor([3.0])

y = w * x + b

print("y:", y)