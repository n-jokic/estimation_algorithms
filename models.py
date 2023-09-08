import jax
import jax.numpy as jnp
import numpy as np
def proces_model(x_state):
    dt = 0.05

    x_new = jnp.array([x_state[0] + (jnp.cos(x_state[6])*x_state[1] - jnp.sin(x_state[6])*x_state[4])*dt,
                       x_state[1] + x_state[2]*dt,
                       x_state[2],
                       x_state[3] + (jnp.cos(x_state[6]) * x_state[4] + jnp.sin(x_state[6]) * x_state[1]) * dt,
                       x_state[4] + x_state[5] * dt,
                       x_state[5],
                       x_state[6]
                       ])
    return x_new


def measurement_model(x_state):
    z = jnp.array([x_state[0],
                   x_state[2],
                   x_state[3],
                   x_state[5],
                   x_state[6]
                   ])
    return z

def process_model_kalman(x_state):
    x_new = x_state
    return x_state
def measurement_model_kalman(x_state):
    return x_state

def proces_model_ukf(x_state):
    num_sigma_states = x_state.shape[0]
    x_new = np.zeros(x_state.shape)
    for idx in range(num_sigma_states):
        x_new[idx, :, :] = proces_model(x_state[idx, :, 0]).reshape((x_state.shape[1], 1))
    return x_new


def measurement_model_ukf(x_state):
    num_sigma_states = x_state.shape[0]
    z_new = np.zeros((x_state.shape[0], x_state.shape[1]-2, x_state.shape[2]))
    for idx in range(num_sigma_states):
        z_new[idx, :, :] = measurement_model(x_state[idx, :, 0]).reshape((x_state.shape[1]-2, 1))

    return z_new







