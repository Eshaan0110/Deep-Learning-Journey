import numpy as np
import matplotlib.pyplot as plt

#XOR dataset

X= np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])


np.random.seed(42)

W1 = np.random.rand(2,2)
b1 = np.zeros((1,2))

W2 = np.random.randn(2, 1)
b2 = np.zeros((1, 1))

#relu activation

def relu(z):
    return np.maximum(0, z)

#sigmoid activation

def sigmoid(z):
    return 1/(1+np.exp(-z))

#forward pass

def forward_pass(X):
    z1=np.dot(X,W1)+b1
    a1=relu(z1)
    z2=np.dot(a1,W2)+b2
    a2=sigmoid(z2)
    return z1,a1,z2,a2

#bce loss
def binary_cross_entropy(y_true,y_pred):
    epsilon = 1e-9
    A2 = np.clip(A2, epsilon, 1 - epsilon)

    return -np.mean(
        y * np.log(A2) +
        (1 - y) * np.log(1 - A2)
    )

#backpropagation

def backpropagation(X,y,z1,a1,z2,a2):
    global W1,b1,W2,b2
    m = X.shape[0]
    dA2 = -(y/a2) + ((1-y)/(1-a2))
    dZ2 = dA2 * (a2 * (1-a2))
    dW2 = np.dot(a1.T, dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m
    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * (z1 > 0)
    dW1 = np.dot(X.T, dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m

    return dW1, db1, dW2, db2



