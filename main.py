# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import numpy as np
import filter_comparison
import matplotlib.pyplot as plt
import os
import trajectory_generator


trajectory_generator.plot_example(101, 0.05)

error = np.array([10, 1, -10, -1])
filter_comparison.only_kalman(np.zeros(4, ) + error, plot=True, save_plot=True)
filter_comparison.only_kalman(np.zeros(4, ), plot=True, save_plot=True)

error = np.array([10, 1, 0.4, -10, -1, -.4, 1])
filter_comparison.run_comparison(np.zeros(7, ) + error, plot=True, save_plot=True)
filter_comparison.run_comparison(np.zeros(7, ), plot=True, save_plot=True)

plt.close('all')

N_mc = 100

kf_rms = np.zeros((N_mc, 4))
ekf_rms = np.zeros((N_mc, 7))
ukf_rms = np.zeros((N_mc, 7))
pf_rms = np.zeros((N_mc, 7))

for i in range(N_mc):
    kf_rms[i, :] = filter_comparison.only_kalman(np.zeros(4, ))
    e_rms, u_rms, p_rms = filter_comparison.run_comparison(np.zeros(7, ))
    ekf_rms[i, :] = e_rms
    ukf_rms[i, :] = u_rms
    pf_rms[i, :] = p_rms
    print(i)



output_folder = "results"
# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)


m_kf = np.mean(kf_rms, axis=0)
v_kf = np.var(kf_rms, axis=0)

variable_name = 'kf'
mean_filename = os.path.join(output_folder, f"{variable_name}_mean.txt")
variance_filename = os.path.join(output_folder, f"{variable_name}_variance.txt")
np.savetxt(mean_filename, m_kf, delimiter=',', fmt='%f')
np.savetxt(variance_filename, v_kf, delimiter=',', fmt='%f')

m_ekf = np.mean(ekf_rms, axis=0)
v_ekf = np.var(ekf_rms, axis=0)

variable_name = 'ekf'
mean_filename = os.path.join(output_folder, f"{variable_name}_mean.txt")
variance_filename = os.path.join(output_folder, f"{variable_name}_variance.txt")
np.savetxt(mean_filename, m_ekf, delimiter=',', fmt='%f')
np.savetxt(variance_filename, v_ekf, delimiter=',', fmt='%f')

m_ukf = np.mean(ukf_rms, axis=0)
v_ukf = np.var(ukf_rms, axis=0)

variable_name = 'ukf'
mean_filename = os.path.join(output_folder, f"{variable_name}_mean.txt")
variance_filename = os.path.join(output_folder, f"{variable_name}_variance.txt")
np.savetxt(mean_filename, m_ukf, delimiter=',', fmt='%f')
np.savetxt(variance_filename, v_ukf, delimiter=',', fmt='%f')

m_pf = np.mean(pf_rms, axis=0)
v_pf = np.var(pf_rms, axis=0)

variable_name = 'pf'
mean_filename = os.path.join(output_folder, f"{variable_name}_mean.txt")
variance_filename = os.path.join(output_folder, f"{variable_name}_variance.txt")
np.savetxt(mean_filename, m_pf, delimiter=',', fmt='%f')
np.savetxt(variance_filename, v_pf, delimiter=',', fmt='%f')


print('done!')



