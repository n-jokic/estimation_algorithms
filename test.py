import jax
import jax.numpy as jnp
import numpy as np
from jax import grad, jit, vmap
from jax import random
from jax import jacfwd, jacrev

key = random.PRNGKey(0)
import matplotlib.pyplot as plt
def process_model(x):

    A = jnp.array([[1, 0.1], [0, 1]])
    x = A.dot(x)
    return x

def state_update(x, sigma_p):
    r = np.random.rand()*sigma_p
    x = process_model(x) + np.array([0, 1])*r
    return x

def measurement_model(x):
    C = jnp.array([1, 0])
    a = jnp.power(C.dot(x),3)
    return a

t = np.linspace(0, 1, num = 100)
n = len(t)
x0 = np.array([3, 0.1])
print(x0.shape)
x = np.zeros((2, n))
y = np.zeros((1, n))
xk = x0
for idx, i in enumerate(t):
    x[:, idx] = xk
    y[:, idx] = measurement_model(xk)
    xk = state_update(xk, sigma_p=0.1)

print(np.array(x))
print(np.array(y))

#plt.plot(t, x.T)
#plt.show()

f = lambda m: process_model(m)
h = lambda z: measurement_model(z)
J = jax.jacobian(f)(x0+1)
Jh = jax.jacobian(h)(x0+1)
print(x0.shape)
print(J)
print(Jh)
print(J.shape)


A = np.eye(3)-1
print(A)
x = np.array([1, 2, 3])
print(A.dot(x))