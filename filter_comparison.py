import matplotlib.pyplot as plt

import extended_kalman_filter as ekf
import trajectory_generator
import unscented_kalman_filter as ukf
import particle_filter as pf
import kalman_filter as kf
import models
import jax
import datetime
jax.config.update('jax_platform_name', 'cpu')
import numpy as np
import os
import time


def plot_results(t, trajectory, ekf_est, ukf_est, pf_est, save_plot=False):
    # Turn interactive plotting off
    plt.ioff()
    num_states = trajectory.shape[1]
    symb = ['sx [m]', 'vx [ms^-1]', 'ax [ms^-2] ', 'sy [m]', 'vy [ms^-1]', 'ay [ms^-2]', r'theta [rad]']
    file_symb = ['sx', 'vx', 'ax ', 'sy', 'vy', 'ay', r'theta']
    # Generate a timestamp for unique filenames
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    for state_idx in range(num_states):
        if state_idx == 0 or state_idx == 3 or state_idx == 6:
            plt.figure(figsize=(8, 4))
            plt.plot(t, trajectory[:, state_idx], 'k-', label='trajektorija')
            plt.plot(t, ekf_est[:, state_idx], 'b--', marker='o', markevery=50, label='prošireni KF')
            plt.grid(True)
            plt.legend()
            plt.xlabel('t [s]')
            plt.ylabel(symb[state_idx])
            plt.tight_layout()

            plt.plot(t, ukf_est[:, state_idx], 'g-.', marker='s', markevery=50, label='neosetljiv KF')
            plt.grid(True)
            plt.legend()
            plt.xlabel('t [s]')
            plt.ylabel(symb[state_idx])
            plt.tight_layout()

            plt.plot(t, pf_est[:, state_idx], 'r:', marker='^', markevery=50, label='čestični filtar')
            plt.grid(True)
            plt.legend()
            plt.xlabel('t [s]')
            plt.ylabel(symb[state_idx])
            plt.tight_layout()
            if save_plot:
                nm = os.path.join('figures', f'all_{file_symb[state_idx]}.png')
                plt.savefig(nm, dpi=600)

        if state_idx == 1 or state_idx == 2:
            th = trajectory[:, 6]
            plt.figure(figsize=(8, 4))
            plt.plot(t, np.cos(th)*trajectory[:, state_idx] - np.sin(th)*trajectory[:, state_idx+3], 'k-', label='trajektorija')
            plt.plot(t, np.cos(th)*ekf_est[:, state_idx] - np.sin(th)*ekf_est[:, state_idx+3], 'b--', marker='o', markevery=50, label='prošireni KF')
            plt.grid(True)
            plt.legend()
            plt.xlabel('t [s]')
            plt.ylabel(symb[state_idx])
            plt.tight_layout()

            plt.plot(t, np.cos(th)*ukf_est[:, state_idx] - np.sin(th)*ukf_est[:, state_idx+3], 'g-.', marker='s', markevery=50, label='neosetljiv KF')
            plt.grid(True)
            plt.legend()
            plt.xlabel('t [s]')
            plt.ylabel(symb[state_idx])
            plt.tight_layout()

            plt.plot(t, np.cos(th)*pf_est[:, state_idx] - np.sin(th)*pf_est[:, state_idx+3], 'r:', marker='^', markevery=50, label='čestični filtar')
            plt.grid(True)
            plt.legend()
            plt.xlabel('t [s]')
            plt.ylabel(symb[state_idx])
            plt.tight_layout()
            if save_plot:
                nm = os.path.join('figures', f'all_{file_symb[state_idx]}.png')
                plt.savefig(nm, dpi=600)

        if state_idx == 4 or state_idx == 5:
            th = trajectory[:, 6]
            plt.figure(figsize=(8, 4))
            plt.plot(t, np.cos(th) * trajectory[:, state_idx] + np.sin(th) * trajectory[:, state_idx - 3], 'k-',
                     label='trajektorija')
            plt.plot(t, np.cos(th) * ekf_est[:, state_idx] + np.sin(th) * ekf_est[:, state_idx - 3], 'b--', marker='o',
                     markevery=50, label='prošireni KF')
            plt.grid(True)
            plt.legend()
            plt.xlabel('t [s]')
            plt.ylabel(symb[state_idx])
            plt.tight_layout()

            plt.plot(t, np.cos(th) * ukf_est[:, state_idx] + np.sin(th) * ukf_est[:, state_idx - 3],
                     'g-.', marker = 's', markevery = 50, label = 'neosetljiv KF')
            plt.grid(True)
            plt.legend()
            plt.xlabel('t [s]')
            plt.ylabel(symb[state_idx])
            plt.tight_layout()


            plt.plot(t, np.cos(th) * pf_est[:, state_idx] + np.sin(th) * pf_est[:, state_idx - 3], 'r:', marker='^',
                     markevery=50, label='čestični filtar')
            plt.grid(True)
            plt.legend()
            plt.xlabel('t [s]')
            plt.ylabel(symb[state_idx])
            plt.tight_layout()
            if save_plot:
                nm = os.path.join('figures', f'all_{file_symb[state_idx]}.png')
                plt.savefig(nm, dpi=600)



    plt.close('all')

def param_init(x0=np.zeros((7, ))):
    dt = 0.05
    N = 101
    t_end = (N-1)*dt
    t = np.linspace(0, t_end, N)

    trajectory = trajectory_generator.generate_trajectory(t)
    z = trajectory_generator.measure_full_trajectory(trajectory)
    x0 = x0

    R = (np.diag([.1, 0.1, .1, 0.1, 0.1])**2)
    Q = np.array([[0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0.1 ** 2 * dt, 0, 0, 0, 0],
                  [0, 0.1 ** 2 * dt, 0.1 ** 2, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0.1 ** 2 * dt, 0],
                  [0, 0, 0, 0, 0.1 ** 2 * dt, 0.1 ** 2, 0],
                  [0, 0, 0, 0, 0, 0, 0.1 ** 2]]
                 )

    P = np.eye(7)

    return t, trajectory, z, x0, R, Q, P

def ekf_init(x0, R, Q, P):
    f = lambda x: models.proces_model(x)
    h = lambda z: models.measurement_model(z)
    Jf = jax.jacobian(f)
    Jh = jax.jacobian(h)

    ekf_1 = ekf.ExtendedKalmanFilter(dim_x=7, dim_z=5)
    ekf_1.Q = Q
    ekf_1.dxh = Jh
    ekf_1.R = R
    ekf_1.dxf = Jf
    ekf_1.f = models.proces_model
    ekf_1.h = models.measurement_model
    ekf_1.P = P
    ekf_1.x = x0

    return ekf_1

def ukf_init(x0, R, Q, P):
    ukf_1 = ukf.UnscentedKalmanFilter(dim_x=7, dim_z=5)
    ukf_1.h = models.measurement_model_ukf
    ukf_1.f = models.proces_model_ukf
    ukf_1.P = P
    ukf_1.x = x0
    ukf_1.R = R
    ukf_1.Q = Q

    return ukf_1

def pf_init(z, t):
    sensor_data = [z[idx, :] for idx in range(len(t))]
    pf_1 = pf.particle_filter(sensor_data=sensor_data, N_particles=20000, algorithm="dynamic_resampling")

    return pf_1


def run_comparison(x0, plot=False, save_plot=False):
    t, trajectory, z, x0, R, Q, P = param_init(x0)
    dt = t[1] - t[0]

    ekf = ekf_init(x0, R, Q, P)
    ekf_est = np.zeros(trajectory.shape)

    start = time.time()
    for idx, i in enumerate(t):
        ekf.update(z=z[idx, :])
        ekf_est[idx, :] = np.array(ekf.x)
        ekf.predict()
    t_ekf = time.time()-start
    ukf = ukf_init(x0, R, Q, P)
    ukf_est = np.zeros(trajectory.shape)

    start = time.time()
    for idx, i in enumerate(t):
        ukf.update(z=z[idx, :])
        ukf_est[idx, :] = np.array(ukf.x)
        ukf.prediction()
    t_ukf = time.time() - start

    pf = pf_init(z, t)
    pf_est = np.zeros(trajectory.shape)
    start = time.time()
    for idx, i in enumerate(t):
        pf_est[idx, :] = pf.particle_filtering()
    t_pf = time.time()-start

    if plot:
        plot_results(t, trajectory, ekf_est, ukf_est, pf_est, save_plot)

    rms_ekf = np.sqrt(np.mean((ekf_est - trajectory) ** 2, axis=0))
    rms_ukf = np.sqrt(np.mean((ukf_est - trajectory) ** 2, axis=0))
    rms_pf = np.sqrt(np.mean((pf_est - trajectory) ** 2, axis=0))

    kf_est = np.zeros((trajectory.shape[0], 4))
    rms_calc = np.zeros((trajectory.shape[0], 4))
    kf_est[:, 0] = z[:, 0]
    kf_est[:, 2] = z[:, 1]
    kf_est[:, 1] = np.cos(trajectory[:, 6]) * z[:, 1] - np.sin(trajectory[:, 6]) * z[:, 3]
    kf_est[:, 3] = np.cos(trajectory[:, 6]) * z[:, 3] + np.sin(trajectory[:, 6]) * z[:, 1]

    rms_calc[:, 0] = trajectory[:, 0]
    rms_calc[:, 1] = np.cos(trajectory[:, 6]) * trajectory[:, 1] - np.sin(trajectory[:, 6]) * trajectory[:, 4]
    rms_calc[:, 2] = trajectory[:, 3]
    rms_calc[:, 3] = np.cos(trajectory[:, 6]) * trajectory[:, 4] + np.sin(trajectory[:, 6]) * trajectory[:, 1]

    rms = np.sqrt(np.mean((kf_est - rms_calc) ** 2, axis=0))

    return rms_ekf, rms_ukf, rms_pf, t_ekf, t_ukf, t_pf, rms


def param_init_kalman(x0):
    dt = 0.05
    N = 101
    t_end = (N - 1) * dt
    t = np.linspace(0, t_end, N)

    trajectory = trajectory_generator.generate_trajectory(t)
    z = trajectory_generator.measure_full_trajectory_kalman(trajectory)

    x0 = x0

    R = np.diag([.1, .1])**2
    Q = np.array([[0, 0.1**2*dt, 0, 0],
                  [0.1**2*dt, 0.1**2, 0, 0],
                  [0, 0, 0, 0.1**2*dt],
                  [0, 0, 0.1**2*dt, 0.1**2]])
    P = np.eye(4)

    return t, trajectory, z, x0, R, Q, P

def kf_init(x0, R, Q, P):
    dt = 0.05
    kf_1 = kf.KalmanFilter(4, 2)
    kf_1.P = P
    kf_1.R = R
    kf_1.Q = Q
    kf_1.x = x0
    kf_1.H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
    kf_1.F = np.array([[1, dt, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 1, dt],
                       [0, 0, 0, 1]])
    return kf_1


def plot_kalman(t, trajectory, kf_est, save_plot=False):
    # Turn interactive plotting off
    plt.ioff()
    num_states = kf_est.shape[1]
    symb = ['sx [m]', 'vx [ms^-1]', 'sy [m]', 'vy [ms^-1]']
    file_symb = ['sx', 'vx', 'sy', 'vy']
    # Generate a timestamp for unique filenames
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    for state_idx in range(num_states):
        if state_idx == 0:
            plt.figure(figsize=(8, 4))
            plt.plot(t, trajectory[:, state_idx], 'k-', label='trajektorija')
            plt.plot(t, kf_est[:, state_idx], 'b--', marker='o', markevery=50, label='KF')
            plt.grid(True)
            plt.legend()
            plt.xlabel('t [s]')
            plt.ylabel(symb[state_idx])
            plt.tight_layout()
        if state_idx==1:
            plt.figure(figsize=(8, 4))
            v = np.cos(trajectory[:, 6])*trajectory[:, 1] - np.sin(trajectory[:, 6])*trajectory[:, 4]
            plt.plot(t, v, 'k-', label='trajektorija')
            plt.plot(t, kf_est[:, state_idx], 'b--', marker='o', markevery=50, label='KF')
            plt.grid(True)
            plt.legend()
            plt.xlabel('t [s]')
            plt.ylabel(symb[state_idx])
            plt.tight_layout()
        if state_idx==3:
            plt.figure(figsize=(8, 4))
            v = np.cos(trajectory[:, 6])*trajectory[:, 4] + np.sin(trajectory[:, 6])*trajectory[:, 1]
            plt.plot(t, v, 'k-', label='trajektorija')
            plt.plot(t, kf_est[:, state_idx], 'b--', marker='o', markevery=50, label='KF')
            plt.grid(True)
            plt.legend()
            plt.xlabel('t [s]')
            plt.ylabel(symb[state_idx])
            plt.tight_layout()
        if state_idx == 2:
            plt.figure(figsize=(8, 4))
            plt.plot(t, trajectory[:, state_idx+1], 'k-', label='trajektorija')
            plt.plot(t, kf_est[:, state_idx], 'b--', marker='o', markevery=50, label='KF')
            plt.grid(True)
            plt.legend()
            plt.xlabel('t [s]')
            plt.ylabel(symb[state_idx])
            plt.tight_layout()


        if save_plot:
            nm = os.path.join('figures', f'kf_{file_symb[state_idx]}.png')
            plt.savefig(nm, dpi=600)



def only_kalman(x0, plot=False, save_plot=False):
    t, trajectory, z, x0, R, Q, P = param_init_kalman(x0)

    kf = kf_init(x0, R, Q, P)
    kf_est = np.zeros((trajectory.shape[0], 4))

    start_time = time.time()
    for idx, i in enumerate(t):
        kf.update(z=z[idx, :])
        kf_est[idx, :] = np.array(kf.x)
        kf.predict()
    runtime = time.time()-start_time
    if plot:
        plot_kalman(t, trajectory, kf_est, save_plot)

    rms_calc = np.zeros((trajectory.shape[0], 4))
    rms_calc[:, 0] = trajectory[:, 0]
    rms_calc[:, 1] = np.cos(trajectory[:, 6])*trajectory[:, 1] - np.sin(trajectory[:, 6])*trajectory[:, 4]
    rms_calc[:, 2] = trajectory[:, 3]
    rms_calc[:, 3] = np.cos(trajectory[:, 6])*trajectory[:, 4] + np.sin(trajectory[:, 6])*trajectory[:, 1]

    rms = np.sqrt(np.mean((kf_est - rms_calc)**2, axis=0))

    kf_est[:, 0] = z[:, 0]
    kf_est[:, 2] = z[:, 1]

    rms_true = np.sqrt(np.mean((kf_est - rms_calc)**2, axis=0))

    return rms, rms_true, runtime
