import jax
import jax.numpy as jnp
import numpy as np
from jax import random
import extended_kalman_filter as ekf


key = random.PRNGKey(0)
import matplotlib.pyplot as plt
def process_model(x):

    A = jnp.array([[0.5, 0.1], [0, 1]])
    x = A.dot(x)
    return x

def state_update(x, sigma_p):
    r = np.random.randn()*sigma_p
    x = process_model(x) + np.array([0, 1])*r
    return x

def measurement_model(x):
    C = jnp.array([[1, 0]])
    a = jnp.power(C@x, 1)
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
    xk = state_update(xk, sigma_p=1)

print(np.array(x))
print(np.array(y))

#plt.plot(t, x.T)
#plt.show()

f = lambda m: process_model(m)
h = lambda z: measurement_model(z)
Jf = jax.jacobian(f)
Jh = jax.jacobian(h)
R = 0.1

ekf_1 = ekf.ExtendedKalmanFilter(dim_x=2, dim_z=1)
ekf_1.Q = np.eye(2)*10
ekf_1.dxh = Jh
ekf_1.R = np.eye(1)*R
ekf_1.dxf = Jf
ekf_1.f = process_model
ekf_1.h = measurement_model
ekf_1.P = np.eye(2)*10
ekf_1.x = np.array([12., -1.])

ekf_est = []
for idx, i in enumerate(y):
    y[:, idx] += np.random.randn()*R

ekf_est = np.zeros((2, n))
for idx, i in enumerate(t):
    ekf_1.predict()
    ekf_1.update(z=y[:, idx])
    ekf_est[:, idx] = ekf_1.x

print(ekf_est)

def measurement_model_2(x):
    C = jnp.array([[0, 1]])
    a = jnp.power(C@x, 2)*jnp.sign(C@x)
    return a
h = lambda z: measurement_model_2(z)
Jh_2 = jax.jacobian(h)
xk = np.array([3, 0.1])
y_2 = np.zeros((1, n))
for idx, i in enumerate(t):
    xk = x[:, idx]
    y_2[:, idx] = measurement_model_2(xk) + np.random.randn()*R

    #xk = state_update(xk, sigma_p=1)


ekf_1 = ekf.ExtendedKalmanInformationFilter(dim_x=2, dim_z=1)
ekf_1.Q = np.eye(2)*1
ekf_1.dxh = [Jh, Jh_2]
ekf_1.R_inv = np.zeros((2, 1, 1))
ekf_1.R_inv[0, :, :] = np.eye(1)*1/R
ekf_1.R_inv[1, :, :] = np.eye(1)*1/R
ekf_1.dxf = Jf
ekf_1.f = process_model
ekf_1.h = [measurement_model, measurement_model_2]
ekf_1.P_inv = np.eye(2)*10
ekf_1.x = np.array([3., 0.1])

print(Jh(xk).shape)
ekf_est = np.zeros((2, n))
for idx, i in enumerate(t):
    ekf_1.update(z=np.array([y[:, idx], y_2[:, idx]]), multiple_sensors=True)
    ekf_est[:, idx] = ekf_1.x
    ekf_1.predict()



plt.plot(t, x.T)
plt.plot(t, ekf_est.T, 'k--')
#plt.plot(t, y.T, color='black', linestyle='dotted')
#plt.plot(t, y_2.T, color = 'red', linestyle = 'dotted')
plt.show()
