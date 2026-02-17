import torch 
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim

#generate data

torch.manual_seed(42)

X = torch.randn(100,1)
y = 3*X + 2 + 0.5*torch.randn(100,1)

#model parameters

model = nn.Linear(1,1)

'''nn.Linear(1,1) creates a linear layer with 1 input feature and 1 output feature.
This means that the layer will take a single input value and produce a single output value. 
it replaces the need to manually define weights and biases, as it automatically initializes them for us. also the w = random 
and b = 0.0 is replaced by the nn.Linear layer which initializes the weights and biases for us.'''

