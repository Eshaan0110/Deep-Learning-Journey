import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

#generate data

X= np.random.randn(100, 1)  # 100 samples, 1 feature
noise = np.random.rand(100, 1) * 0.5
y = 3 * X + noise

#Normalize data - standardization (z-score normalization)-used to scale the features of the data to have a mean of 0 and a standard deviation of 1. This is important for many machine learning algorithms, including linear regression, as it helps to ensure that all features contribute equally to the model and prevents features with larger scales from dominating the learning process.

X_mean = np.mean(X)
X_std = np.std(X)
X_normalized = (X - X_mean) / X_std

#model parameters 
'''In linear regression, the model parameters typically include the weights (coefficients) and the bias (intercept). 
    The weights determine the influence of each feature on the predicted output, 
    while the bias allows the model to fit data that does not pass through the origin. 
    In this case, since we have only one feature, we will have one weight and one bias.'''

w = np.random.rand(1,1) # weight
b= 0.0

#forward pass

def predict(X,w,b):
    return w*X + b

#loss function
'''Mean Squared Error (MSE) is a common loss function used in regression tasks.
    It measures the average of the squares of the errors—that is, 
    the average squared difference between the predicted values and the actual values.'''
    
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

#gradient computation

def compute_gradients(X, y_true, y_pred):
    n = len(y_true)
    dw = (-2/n) * np.sum(X * (y_true - y_pred))
    db = (-2/n) * np.sum(y_true - y_pred)
    return dw, db

#training loop

learning_rate = 0.01
epochs = 1000
losses = []

for epoch in range(epochs):
    y_pred = predict(X_normalized, w, b)
    loss = mse_loss(y, y_pred)
    losses.append(loss)
    
    dw, db = compute_gradients(X_normalized, y, y_pred)
    
    w -= learning_rate * dw
    b -= learning_rate * db

    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss}')

#visualization

plt.scatter(X, y, label="Data")
plt.plot(X, predict(X, w, b), color="red", label="Regression Line")
plt.legend()
plt.show()

plt.plot(losses)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.show()