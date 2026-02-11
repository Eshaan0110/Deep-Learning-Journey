# NumPy From Scratch – Linear Regression

## Overview

This project is the first step in my Deep Learning Journey.

The goal was to understand the mathematical and computational foundations of machine learning by implementing linear regression entirely from scratch using only NumPy.

No machine learning libraries were used.

---

## What This Project Covers

- NumPy array operations  
- Broadcasting  
- Manual gradient descent  
- Loss computation  
- Linear regression from scratch  
- Basic data visualization with Matplotlib  

---

## Why NumPy Instead of Python Lists?

NumPy is used because:

- It stores data in contiguous memory blocks.
- It supports vectorized operations (no explicit loops needed).
- It is significantly faster than Python lists.
- It forms the foundation of deep learning frameworks like PyTorch and TensorFlow.

Understanding NumPy is essential before moving into deep learning libraries.

---

## Problem Setup

We generate synthetic data based on a linear relationship:

y=3x + noise


The model attempts to learn this relationship using gradient descent.

---

## Model Equation

y_pred = w * X + b


Where:

- `w` = weight (slope)
- `b` = bias (intercept)
- `X` = input feature
- `y_pred` = predicted output

---

## Loss Function

We use **Mean Squared Error (MSE)**:

MSE = mean( (y_true - y_pred)^2 )


This loss penalizes larger errors more heavily.

---

## Gradient Descent Update Rule

w = w - learning_rate * dw
b = b - learning_rate * db


Where:

- `dw` is the derivative of loss with respect to `w`
- `db` is the derivative of loss with respect to `b`

This process repeats for multiple epochs until the loss decreases.

---

## Training Process

1. Initialize `w` and `b`
2. Make predictions
3. Compute loss
4. Compute gradients
5. Update parameters
6. Repeat

---

## Results

- Training loss decreases over time.
- The regression line fits the noisy data.
- The learned weight approaches the true slope (~3).

---

## What I Learned

- How vectorized operations work in NumPy  
- How broadcasting simplifies mathematical operations  
- How gradient descent updates parameters  
- Why normalization improves training stability  
- The relationship between loss functions and optimization  

---

## Project Structure

numpy-from-scratch/
│
├── linear_regression.py
├── loss_function.py
├── utils.py
└── README.md

