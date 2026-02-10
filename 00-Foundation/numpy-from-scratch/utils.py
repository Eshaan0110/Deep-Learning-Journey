import numpy as np

#basic array creation

a = np.array([1, 2, 3])
print(a)
b = np.array([[1, 2, 3], [4, 5, 6]])
print(b)

#array attributes
print(a.shape)
print(b.shape)
print(a.dtype)
print(b.dtype)

#random array creation
c = np.random.rand(3, 4)
d = np.random.randn(3, 4)

 # difference btw rand and randn is that rand generates random numbers from a uniform distribution over [0, 1), while randn generates random numbers from a standard normal distribution (mean 0, variance 1) or (-infinity, +infinity)
print(c)
print(d)

