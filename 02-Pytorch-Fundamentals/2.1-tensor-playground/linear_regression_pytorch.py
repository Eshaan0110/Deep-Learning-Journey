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

#loss function

criterion = nn.MSELoss()
''' MSELoss = Mean Squared Error Formula:    
 mean((y_pred - y_true)^2). This measures how wrong our predictions are. 
 In the context of linear regression, it calculates the average of the squares of the differences between the predicted values (y_pred) and 
 the actual target values (y_true). The goal of training is to minimize this loss, which means we want our predictions to be as close as possible
to the true values.
'''

#optimizer

optimizer = optim.SGD(model.parameters(), lr=0.01)

'''SGD = Stochastic Gradient Descent. It is an optimization algorithm used to update the parameters of a model during training. 
The "stochastic" part means that it updates the parameters using a single sample (or a small batch) of data at a time, rather than the entire dataset. 
This can lead to faster convergence in some cases, especially for large datasets. 
The "gradient descent" part refers to the method of updating the parameters in the direction that reduces the loss.'''

#training loop

epochs = 100
losses = []

for epoch in range ( epochs):

    #computation of loss
    y_pred = model(X)
    loss = criterion(y_pred, y)
    losses.append(loss.item())


    #backward pass 

    optimizer.zero_grad()
    loss.backward()
    #update parameters
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")
# ==============================
# Visualization
# ==============================

with torch.no_grad():
    predictions = model(X)

plt.scatter(X.numpy(), y.numpy(), label="Data")
plt.plot(X.numpy(), predictions.numpy(), color="red", label="Model")
plt.legend()
