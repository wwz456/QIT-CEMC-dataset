from sklearn.preprocessing import MinMaxScaler
import pywt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import scipy.signal as signal
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler

# 全局设置字体
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 10
def get_abspath(path):
    '''
    获取振动与声音文件
    path:csv文件所在文件夹
    '''
    file_name = os.listdir(path)
    sorted_file_names = sorted(file_name, key=custom_sort)
    file_list = [os.path.join(path, x) for x in sorted_file_names]
    return file_list
# 定义自定义排序函数
def custom_sort(filename):
    parts = filename.split('-')
    return (int(parts[0]), int(parts[1]), int(parts[2].split('.')[0]))
def short_time_energy(data, window_size, step_size):
    """
    计算数据序列的短时能量

    :param data: 输入的数据序列
    :param window_size: 滑动窗口的大小
    :param step_size: 窗口移动的步长
    :return: 短时能量的列表
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
    应用高斯混合模型进行聚类

    :param energies: 短时能量序列
    :param n_components: GMM的组件数量
    :return: 每个数据点的类别标签
    """
    scaler=MinMaxScaler(feature_range=(0,1))
    energies_scaled=scaler.fit_transform(energies)
    gmm = GaussianMixture(n_components=n_components, covariance_type='full', tol=tol,random_state=0)
    gmm.fit(energies_scaled)
    labels = gmm.predict(energies_scaled)
    return labels
def get_keep_indices(data, window_size, step_size):
    """
    获取保留的数据点的索引

    :param data: 输入的数据序列
    :param window_size: 滑动窗口的大小
    :param step_size: 窗口移动的步长
    :return: 保留的索引
    """
    # 计算短时能量
    energy_list = short_time_energy(data, window_size, step_size)

    # 使用 GMM 进行聚类
    labels = apply_gmm(energy_list)

    # 确定低能量类别的标签
    # 这里假设低能量类别的标签为 0，您可能需要根据实际结果调整
    low_energy_label = 1

    # 创建一个布尔索引，用于标记保留的数据点
    keep_indices = np.ones(len(data), dtype=bool)

    num_windows = (len(data) - window_size) // step_size + 1
    for i in range(num_windows):
        start = i * step_size
        end = start + window_size

        if labels[i] == low_energy_label:
            # 标记该窗口内的所有数据点需要删除
            keep_indices[start:end] = False

    return keep_indices
# 移动平均法校正基线漂移
def moving_average_baseline_correction(signal, window_size=100):
    baseline_estimate = np.convolve(signal, np.ones(window_size) / window_size, mode='same')
    corrected_signal = signal - baseline_estimate
    return corrected_signal, baseline_estimate
if __name__=='__main__':
    # 获取文件名列表
    f_path = r'F:\phd\2024立铣刀数据集\力信号'
    v_path=r'F:\phd\2024立铣刀数据集\振动信号'
    f_list = get_abspath(f_path)
    v_list=get_abspath(v_path)
    #读取某力数据
    data=pd.read_csv(f_list[6],sep='\t').values
    print(data.shape)
    #计算短时能量并聚类
    window_size = 500
    step_size = 500
    # 获取保留的索引
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
