# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import numpy as np
import kalman_filter as kalman
import matplotlib.pyplot as plt
def generate_trajectory_1d(num_steps, dt, initial_position, velocity, acceleration=0):
    """
    Generates a 1D trajectory for an object moving with constant or zero acceleration.

    Parameters:
        num_steps (int): Number of time steps in the trajectory.
        dt (float): Time step size.
        initial_position (float): Initial position.
        velocity (float): Constant velocity of the object.
        acceleration (float, optional): Constant acceleration. Default is 0 (constant velocity).

    Returns:
        np.ndarray: Array of shape (num_steps,) representing the 1D trajectory positions.
    """
    position = initial_position
    trajectory = np.zeros(num_steps)

    for t in range(num_steps):
        position += velocity * dt + 0.5 * acceleration * dt**2
        velocity += acceleration * dt
        trajectory[t] = position

    return trajectory

def generate_measurements_1d(trajectory, measurement_noise_std):
    """
    Generates noisy measurements for a given 1D trajectory.

    Parameters:
        trajectory (np.ndarray): Array of shape (num_steps,) representing the 1D trajectory positions.
        measurement_noise_std (float): Standard deviation of measurement noise.

    Returns:
        np.ndarray: Array of shape (num_steps,) representing the noisy measurements.
    """
    num_steps = trajectory.shape[0]
    measurement_noise = np.random.normal(scale=measurement_noise_std, size=num_steps)
    measurements = trajectory + measurement_noise

    return measurements

if __name__ == "__main__":
    # Parameters
    num_steps = 100  # Number of time steps in the trajectory
    dt = 0.1  # Time step size
    initial_position = 0  # Initial position
    velocity = 1  # Constant velocity
    acceleration = 0.1  # Constant acceleration
    measurement_noise_std = 0.5*2  # Standard deviation of measurement noise

    # Generate 1D trajectory and measurements
    trajectory_1d = generate_trajectory_1d(num_steps, dt, initial_position, velocity, acceleration)
    measurements_1d = generate_measurements_1d(trajectory_1d, measurement_noise_std)

    # Print the first few positions and measurements for illustration
    print("True 1D Trajectory:")
    print(trajectory_1d[:5])
    print("\nNoisy Measurements:")
    print(measurements_1d[:5])
    kf = kalman.KalmanFilter(dim_x = 2, dim_z = 1)
    kf.F = np.array([[1, dt],
              [0, 1]])
    kf.R = np.eye(1)*0.25
    kf.H = np.array([[1], [0]]).T
    kf.Q = np.eye(2)*0.01
    kf_est = []
    i = 0
    for sample in measurements_1d:
        kf.predict()
        #print(kf.H)
        kf.update(z=sample)
        kf_est.append(kf.x[0])

    t = np.linspace(0, 10, num_steps)
    plt.plot(t, trajectory_1d)
    plt.plot(t, np.array(kf_est))
    plt.plot(t, measurements_1d, color = 'black', linestyle='dotted')
    plt.show()

    kif = kalman.KalmanInformationFilter(dim_x=2, dim_z=1)
    kif.F = np.array([[1, dt],
                     [0, 1]])
    kif.R_inv = np.eye(1) * 4
    kif.H = np.array([[1], [0]]).T
    kif.Q = np.eye(2) * 0.01
    kif_est = []
    i = 0
    measurements_1d_good = generate_measurements_1d(trajectory_1d, measurement_noise_std/8)
    for sample in measurements_1d:
        kif.predict()
        # print(kf.H)
        print(np.array([[sample], [measurements_1d_good[i]]])[0, :].shape)
        kif.update(z=np.array([[sample], [measurements_1d_good[i]]]).T, R_inv=[1, 16], multiple_sensors=True)
        kif_est.append(kif.x[0])
        i+=1

    t = np.linspace(0, 10, num_steps)
    plt.plot(t, trajectory_1d)
    plt.plot(t, np.array(kif_est))
    plt.plot(t, measurements_1d, color='black', linestyle='dotted')
    plt.plot(t, measurements_1d_good, color = 'red', linestyle = 'dotted')
    plt.show()


def process_model(x):
    x1 = x[0, 0]
    x2 = x[1, 0]

    x[0, 0] = 0.1*x2
    x[1, 0] = x1

    return x

def state_update(x, sigma_p):
    r = np.random.rand()*sigma_p
    x = process_model(x) + np.array([[0.1, 0], [0, 1]])*r
    return x

def measurement_model(x):
    return np.array([x[0, 0]**3])

t = np.linspace(0, 1, num = 100)
n = len(t)
x0 = np.array([[0], [0.1]])
x = np.zeros((2, n))
y = np.zeros((1, n))
xk = x0
for idx, i in enumerate(t):
    x[:, idx] = xk[:, 0]
    y[idx] = measurement_model(xk)[0]
    xk = state_update(xk, sigma_p=0.1)

print(np.array(x))
print(np.array(y))




# See PyCharm help at https://www.jetbrains.com/help/pycharm/
