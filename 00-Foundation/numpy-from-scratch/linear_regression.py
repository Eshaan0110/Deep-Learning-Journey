import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

#generate data

X= np.random.randn(100, 1)  # 100 samples, 1 feature
noise = np.random.rand(100, 1) * 0.5
y = 3 * X + noise

#Normalize data

X_mean = np.mean(X)
X_std = np.std(X)
X_normalized = (X - X_mean) / X_std

#model parameters

w = np.random.rand(1,1) # weight
b= 0.0

#forward pass

def predict(X,w,b):
    return w*X + b

#loss function

def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)