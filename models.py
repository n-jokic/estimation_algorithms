import jax
import jax.numpy as jnp
import numpy as np
def proces_model(x_state):
    dt = 0.01
    x_new = jnp.array([x_state[0] + (jnp.cos(x_state[6])*x_state[1] - jnp.sin(x_state[6])*x_state[3])*dt,
                       x_state[1] + x_state[2]*dt,
                       x_state[2],
                       x_state[3] + (jnp.cos(x_state[6]) * x_state[3] + jnp.sin(x_state[6]) * x_state[1]) * dt,
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



# Example state
x_example = jnp.array([2.0, 3.0, 1, 1, 1, 1,  0.1*np.pi])

# Compute the Jacobian of the transition model
transition_jacobian = jax.jacfwd(proces_model)(x_example)

# Compute the Jacobian of the measurement model
measurement_jacobian = jax.jacfwd(measurement_model)(x_example)

print("State Transition Jacobian:")
print(transition_jacobian)

print("\nMeasurement Jacobian:")
print(measurement_jacobian)
