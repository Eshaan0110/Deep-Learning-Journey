
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
