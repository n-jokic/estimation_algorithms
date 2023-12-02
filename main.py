# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import numpy as np
import filter_comparison
import matplotlib.pyplot as plt
import os
import trajectory_generator


trajectory_generator.plot_example(101, 0.05)
print('done_example')

#error = np.array([10, 1, -10, -1])
#filter_comparison.only_kalman(np.zeros(4, ) + error, plot=True, save_plot=True)
kf_res = filter_comparison.only_kalman(np.zeros(4, ), plot=True, save_plot=True)
print(kf_res[1])
print(kf_res[0])

#error = np.array([10, 1, 0.4, -10, -1, -.4, 1])
#filter_comparison.run_comparison(np.zeros(7, ) + error, plot=True, save_plot=True)
filter_comparison.run_comparison(np.zeros(7, ), plot=True, save_plot=True)

plt.close('all')

print('done, first part')

N_mc = 100

t_kf = []
kf_rms = np.zeros((N_mc, 4))
t_ekf = []
ekf_rms = np.zeros((N_mc, 7))
t_ukf = []
ukf_rms = np.zeros((N_mc, 7))
t_pf = []
pf_rms = np.zeros((N_mc, 7))
no_filt_rms = np.zeros((N_mc, 4))

for i in range(N_mc):
    kf_res = filter_comparison.only_kalman(np.zeros(4, ))
    t_kf.append(kf_res[2])
    kf_rms[i, :] = kf_res[0]
    other_res = filter_comparison.run_comparison(np.zeros(7, ))
    ekf_rms[i, :] = other_res[0]
    ukf_rms[i, :] = other_res[1]
    pf_rms[i, :] = other_res[2]
    t_ekf.append(other_res[3])
    t_ukf.append(other_res[4])
    t_pf.append(other_res[5])
    no_filt_rms[i, :] = other_res[6]
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
t = {'Kalman': t_kf, 'EKF': t_ekf, 'UKF': t_ukf, 'PF': t_pf}
for key in t:
    data = t[key]

    # Compute mean and standard deviation
    mean = np.mean(data)
    std = np.std(data)

    # Create subplots
    fig, ax = plt.subplots()

    # Plot histogram
    ax.hist(data, bins=30, alpha=0.5, color='blue', edgecolor='black')
    ax.axvline(mean, color='red', linestyle='dashed', linewidth=1.5, label='srednja vrednost')

    # Plot bars for 3 standard deviations
    ax.axvspan(mean - 3 * std, mean + 3 * std, color='orange', alpha=0.2, label='+- 3 std')
    # Set labels and title
    ax.set_xlabel('t [s]')
    #ax.set_title('Histogram of : ' + algorithm + ' estmation')
    #ax.set_xlim([0.58, 0.71])
    # Add legend
    ax.legend()
    plt.rcParams['text.usetex'] = False
    # Display the plot
    ax.set_rasterized(True)
    FULL_PATH = r'C:\Users\milos\OneDrive\VIII_semestar\diplomski\code\estimation_algorithms\figures' + key+ '.png'
    plt.savefig(FULL_PATH, format='png', dpi=600)
plt.show()




