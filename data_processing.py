from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import scipy.signal as signal
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler

# set font style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 10
def get_abspath(path):
    '''
    #get data path list
    path: folder path of the '.csv' or 'txt' files
    '''
    file_name = os.listdir(path)
    sorted_file_names = sorted(file_name, key=custom_sort)
    file_list = [os.path.join(path, x) for x in sorted_file_names]
    return file_list
# sort the file due to the time
def custom_sort(filename):
    parts = filename.split('-')
    return (int(parts[0]), int(parts[1]), int(parts[2].split('.')[0]))
def short_time_energy(data, window_size, step_size):
    """
    calculate the short energy using the sliding window method
    :param data: input data
    :param window_size: sliding window size
    :param step_size: step of the sliding window
    :return: finally list short energy
    """
    energies = []
    num_windows = (len(data) - window_size) // step_size + 1

    for i in range(num_windows):
        start = i * step_size
        end = start + window_size
        window_data = data[start:end]
        energy = np.sum(np.square(window_data))
        energies.append(energy)

    return np.array(energies).reshape(-1, 1)  # reshape for GMM input
def apply_gmm(energies, n_components=2,tol=1e-5):
    """
    cluster to 2 classes using the GMM method

    :param energies: the short energy list
    :param n_components: components of the GMM, here is 2
    :return: Result of the 2 label 
    """
    scaler=MinMaxScaler(feature_range=(0,1))
    energies_scaled=scaler.fit_transform(energies)
    gmm = GaussianMixture(n_components=n_components, covariance_type='full', tol=tol,random_state=0)
    gmm.fit(energies_scaled)
    labels = gmm.predict(energies_scaled)
    return labels
def get_keep_indices(data, window_size, step_size):
    """
    Get the indices of the data points to keep.

    :param data: The input data sequence
    :param window_size: The size of the sliding window
    :param step_size: The step size of the sliding window
    :return: Indices of the data points to keep
    """
    # Compute short-term energy
    energy_list = short_time_energy(data, window_size, step_size)

    # Apply GMM for clustering
    labels = apply_gmm(energy_list)

    # Determine the label for low energy class
    # Here we assume the label for low energy class is 0, you might need to adjust based on actual results
    low_energy_label = 1

    # Create a boolean index to mark the data points to keep
    keep_indices = np.ones(len(data), dtype=bool)

    num_windows = (len(data) - window_size) // step_size + 1
    for i in range(num_windows):
        start = i * step_size
        end = start + window_size

        if labels[i] == low_energy_label:
            # Mark all data points in this window for deletion
            keep_indices[start:end] = False

    return keep_indices
# Moving average method for baseline drift correction
def moving_average_baseline_correction(signal, window_size=100):
    baseline_estimate = np.convolve(signal, np.ones(window_size) / window_size, mode='same')
    corrected_signal = signal - baseline_estimate
    return corrected_signal, baseline_estimate
if __name__=='__main__':
    # Get the list of file names
    f_path = r'F:\phd\2024立铣刀数据集\力信号'
    v_path = r'F:\phd\2024立铣刀数据集\振动信号'
    f_list = get_abspath(f_path)
    v_list = get_abspath(v_path)
    
    # Read certain force data
    data = pd.read_csv(f_list[6], sep='\t').values
    print(data.shape)
    
    # Compute short-term energy and perform clustering
    window_size = 500
    step_size = 500
    
    # Get the indices to keep
    keep_indices = get_keep_indices(data[:,1], window_size, step_size)
    print(data.shape)
    print(data[keep_indices].shape)
    '''
    #check the raw data
    plt.figure(figsize=(12,3))
    plt.plot(data[0:1000000, 1], label='Raw signal with invalid data')
    plt.xlabel('Sample data point')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    plt.show()
    plt.figure(figsize=(12, 3))
    plt.plot(data[2000000:3000000,1],label='Raw signal with invalid data')
    plt.xlabel('Sample data point')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    plt.show()
    plt.figure(figsize=(12, 3))
    plt.plot(data[-1000000:, 1], label='Raw signal with invalid data')
    plt.xlabel('Sample data point')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 3))
    plt.plot(data[:, 1], label='Raw signal with invalid data')
    plt.xlabel('Sample data point')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 3))
    plt.plot(data[keep_indices][:, 1], label='Raw signal with invalid data')
    plt.xlabel('Sample data point')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    plt.show()
    '''

    # moving average method to remove the baseline drift in Force_z
    signal=data[keep_indices][:,3]
    corrected_signal_ma, baseline_ma = moving_average_baseline_correction(signal, window_size=100)
    plt.figure(figsize=(12,5))
    plt.plot(signal, label='Original Signal')
    plt.plot(baseline_ma, label='Estimated Baseline Drift', linestyle='--')
    plt.plot(corrected_signal_ma, label='Baseline Corrected Signal')
    plt.xlabel('Sample data point')
    plt.ylabel('Amplitude')
    plt.title('Baseline Drift Removal using Moving Average Method')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
