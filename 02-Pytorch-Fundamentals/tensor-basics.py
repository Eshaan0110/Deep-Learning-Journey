import torch

print(torch.__version__)


#create tensor

# x = torch.tensor([[1, 2, 3],[4, 5, 6]])

# print(x)
# print("data type of x:", x.dtype)
# print("shape of x:", x.shape)
# print("device of x:", x.device)


#require gradient example
'''requires_grad=True allows us to track the gradients of the tensor during backpropagation. 
This is essential for training neural networks, as it enables the computation of gradients with respect to the parameters. so in simple terms, 
it allows us to calculate how much the parameters (like weights and biases)
 should be adjusted to minimize the loss function during training.'''


w = torch.tensor([2.0], requires_grad=True)
b = torch.tensor([1.0], requires_grad=True)
x = torch.tensor([3.0])

# y = w * x + b

# print("y:", y)
# print(w.requires_grad)
# print(x.requires_grad)
# print(y.requires_grad)

#inspect gradients

print("Initial values:")
print("w:", w)
print("b:", b)
print("x:", x)

z = w * x
print("\nAfter multiplication z = w * x:")
print("z:", z)
print("z.grad_fn:", z.grad_fn)


y = z + b
print("\nAfter addition y = z + b:")
print("y:", y)
print("y.grad_fn:", y.grad_fn)

print("\nAfter backward()")
print("Gradient of w:", w.grad)
print("Gradient of b:", b.grad)
