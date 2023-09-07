import numpy as np
import matplotlib.pyplot as plt

def trajectory_param_init(t):
    _T = t[-1]

    theta = np.sin(2*np.pi/_T*t)*np.pi/2
    acc_x_bf = -np.sin(4*np.pi/_T*t)/10
    acc_y_bf = np.sin(4*np.pi / _T * t)/10

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
                                                         - np.sin(old_theta)*trajectory[idx - 1, 4])    # update pos_x
            trajectory[idx, 3] = trajectory[idx-1, 3] + (np.sin(old_theta) * trajectory[idx - 1, 1]
                                                         + np.cos(old_theta) * trajectory[idx - 1, 4])  # update pos_y

    return trajectory

t = np.linspace(0, 10, num=1001)
print(t[1]-t[0])

trajectory = generate_trajectory(t)
plt.figure()


pos_x = trajectory[:, 0]
plt.plot(t, pos_x)
plt.figure()
sp_x = trajectory[:, 1]
plt.plot(t, sp_x)
plt.figure()
acc_x = trajectory[:, 2]
plt.plot(t, acc_x)
plt.figure()

differences = np.diff(acc_x)

# Compute the average difference
average_difference = np.mean(differences)

print("Average difference:", average_difference)

pos_y = trajectory[:, 3]
plt.plot(t, pos_y)
plt.figure()
sp_y = trajectory[:, 4]
plt.plot(t, sp_y)
plt.figure()
acc_y = trajectory[:, 5]
plt.plot(t, acc_y)
plt.figure()

differences = np.diff(acc_y)

# Compute the average difference
average_difference = np.mean(differences)

print("Average difference:", average_difference)

theta = trajectory[:, 6]
plt.plot(t, theta)
plt.show()

differences = np.diff(theta)

# Compute the average difference
average_difference = np.mean(differences)

print("Average difference:", average_difference)


