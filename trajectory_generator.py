import numpy as np
import matplotlib.pyplot as plt
import os

def trajectory_param_init(t):
    _T = t[-1]

    theta = np.sin(2*np.pi/_T*t)*np.pi/2
    acc_x_bf = -np.sin(4*np.pi/_T*t)
    acc_y_bf = np.sin(4*np.pi / _T * t)

    return acc_x_bf, acc_y_bf, theta


def generate_trajectory(t):
    acc_x_bf, acc_y_bf, theta = trajectory_param_init(t)
    NUM_STATES = 7  # 2 positions, speeds and accelerations and 1 tilting angle
    trajectory = np.zeros(((len(t), NUM_STATES)))

    dt = t[1]-t[0]

    for idx, val in enumerate(t):
        trajectory[idx, 2] = acc_x_bf[idx]
        trajectory[idx, 5] = acc_y_bf[idx]
        trajectory[idx, 6] = theta[idx]

        if idx != 0:
            trajectory[idx, 1] = trajectory[idx-1, 1] + trajectory[idx-1, 2]*dt        # update the speed_x
            trajectory[idx, 4] = trajectory[idx - 1, 4] + trajectory[idx - 1, 5] * dt  # update the speed_y
            trajectory[idx, 6] = theta[idx]     # update the tilting angle

            old_theta = theta[idx-1]

            trajectory[idx, 0] = trajectory[idx-1, 0] + (np.cos(old_theta)*trajectory[idx - 1, 1]
                                                         - np.sin(old_theta)*trajectory[idx - 1, 4])*dt    # update pos_x
            trajectory[idx, 3] = trajectory[idx-1, 3] + (np.sin(old_theta) * trajectory[idx - 1, 1]
                                                         + np.cos(old_theta) * trajectory[idx - 1, 4])*dt  # update pos_y

    return trajectory


def measure(x_state):
    sigma_pos = 0.1
    sigma = 0.1
    sigma_theta = 0.1

    z = np.array([x_state[0] + np.random.randn()*sigma_pos,
                  x_state[2] + np.random.randn()*sigma,
                  x_state[3] + np.random.randn()*sigma_pos,
                  x_state[5] + np.random.randn()*sigma,
                  x_state[6] + np.random.randn()*sigma_theta
                  ])
    return z


def measure_full_trajectory(trajectory):
    duration = len(trajectory[:, 0])
    z = np.zeros((duration, 5))

    for idx in range(duration):
        z[idx, :] = measure(trajectory[idx, :])

    return z


def measure_full_trajectory_kalman(trajectory):
    duration = len(trajectory[:, 0])
    z = np.zeros((duration, 2))

    for idx in range(duration):
        tmp = measure(trajectory[idx, :])
        z[idx, :] = np.array([tmp[0], tmp[2]])

    return z

def plot_example(N, dt):
    dt = dt
    N = N
    t_end = (N - 1) * dt
    t = np.linspace(0, t_end, N)
    trajectory = generate_trajectory(t)
    z = measure_full_trajectory(trajectory)
    symb = ['sx [m]', 'ax [ms^-2] ', 'sy [m]', 'ay [ms^-2]', r'theta [rad]']
    traj = [0, 2, 3, 5, 6]


    # Generate a timestamp for unique filenames
    num_meas = 5

    for state_idx in range(num_meas):
        plt.figure(figsize=(8, 4))
        plt.plot(t, z[:, state_idx], 'b--', marker='o', label='zašumljena trajektorija')
        plt.plot(t, trajectory[:, traj[state_idx]], 'k-', label='trajektorija')
        plt.grid(True)
        plt.legend()
        plt.xlabel('t [s]')
        plt.ylabel(symb[state_idx])
        plt.tight_layout()
        nm = os.path.join('figures', f'example_{symb[state_idx]}.png')
        plt.savefig(nm, dpi=600)
