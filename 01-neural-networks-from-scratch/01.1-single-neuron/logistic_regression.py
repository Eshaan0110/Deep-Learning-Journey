
# Linear Model → Activation → Loss → Optimization - The holy flow


import numpy as np
import matplotlib.pyplot as plt 

#Data Generation

np.random.seed(42)

X = np.random.randn(200,1)

y = (X>0).astype(int)

#model parameters

w = np.random.randn()
b= 0.0

#Sigmoid Activation

def sigmoid(z):

    '''
    Sigmoid activation function.

    Transforms any real-valued number into a probability between 0 and 1.

    Formula:
        sigmoid(z) = 1 / (1 + exp(-z))

    Why we use it:
    - Converts linear output (wX + b) into probability
    - Smooth and differentiable (required for gradient descent)
    - Used in binary classification problems
    '''

    return 1/(1+np.exp(-z))

#forward pass

def predict(W,x,b):
    z = w*X + b
    return sigmoid(z)

#loss function (BCE)

def binary_cross_entropy(y_true,y_pred):
    epsilon = 1e-9
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(
        y_true * np.log(y_pred) +
        (1 - y_true) * np.log(1 - y_pred)
    )

#gradient descent

def compute_gradients(X, y_true, y_pred):
    n = len(X)
    dw = (1/n)* np.sum((y_pred - y_true) * X)
    db = (1/n)* np.sum(y_pred - y_true)
    return dw, db

#training loop

learning_rate = 0.1
epochs = 1000
losses = []

for epoch in range(epochs):
    y_pred = predict(X, w, b)

    loss = binary_cross_entropy(y, y_pred)
    losses.append(loss)

    dw, db = compute_gradients(X, y, y_pred)

    w -= learning_rate * dw
    b -= learning_rate * db

    if epoch % 100 == 0:
        print(f"Epoch {epoch} | Loss: {loss:.4f}")

    
#visualization

plt.scatter(X, y, alpha=0.5, label="Data")

# Decision boundary occurs when wX + b = 0
decision_boundary = -b / w
plt.axvline(x=decision_boundary, color='red', label="Decision Boundary")

plt.legend()
plt.show()