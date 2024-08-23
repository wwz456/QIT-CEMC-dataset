import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pywt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
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
# 获得某把刀的所有文件路径列表
def get_allpath(toll_path):
    path=os.listdir(toll_path)
    sorted_path=sorted(path,key=lambda x:x[4:7])
    # print(sorted_path)
    c_path=[]
    for file in sorted_path:
        c_path.append(toll_path+'\\'+file)
    return c_path
def get_toolpath(tool_num):
    if tool_num==1:
        path=r'D:\minist_test\c1\c1'
    elif tool_num==4:
        path=r'D:\minist_test\c1\c4'
    else:
        path = r'D:\minist_test\c1\c6'
    return path
#截断数据
def trim_milling_data(milling_data):
    # 计算第三四分位数Q3
    Q3 = np.percentile(milling_data, 75)

    # 从数据的开头开始查找第一个大于Q3的值
    start_index = 0
    for i in range(len(milling_data)):
        if milling_data[i] > Q3:
            start_index = i
            break

    # 截断数据，从第一个大于Q3的值开始
    trimmed_data = milling_data[start_index:]

    # 从数据的结尾开始查找第一个大于Q3的值
    end_index = len(trimmed_data) - 1
    for i in range(len(trimmed_data) - 1, -1, -1):
        if trimmed_data[i] > Q3:
            end_index = i
            break

    # 再次截断数据，从最后一个大于Q3的值开始
    final_trimmed_data = trimmed_data[:end_index+1]

    return final_trimmed_data, start_index, end_index
#去除离群值
def replace_outliers_with_mean(signal, threshold=1.5):
    """
    将信号中的离群值替换为就近的平均值。

    参数：
    signal (numpy数组)：包含信号数据的numpy数组。
    threshold (浮点数)：用于定义离群值的IQR阈值倍数。

    返回：
    处理后的信号，离群值替换为就近的平均值。
    """
    # 计算IQR
    Q1 = np.percentile(signal, 25)
    Q3 = np.percentile(signal, 75)
    IQR = Q3 - Q1

    # 定义离群值的下界和上界
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR

    # 处理离群值
    processed_signal = np.copy(signal)
    outliers = (signal < lower_bound) | (signal > upper_bound)
    for i in range(len(signal)):
        if outliers[i]:
            # 找到就近的非离群值的平均值
            nearest_values = signal[~outliers]
            nearest_mean = np.mean(nearest_values)
            processed_signal[i] = nearest_mean

    return processed_signal


def hampel_filter(data, window_size, threshold):
    n = len(data)
    filtered_data = data.copy()

    for i in range(n):
        start = max(0, i - window_size)
        end = min(n, i + window_size + 1)
        window = data[start:end]

        std= np.std(window)
        if data[i]>threshold*std or data[i]<-threshold*std:
            filtered_data[i] = np.mean(window)

    return filtered_data
def hampel_filter_all(data,w_size,thres):
    out=np.empty_like(data)
    for i in range(data.shape[1]):
        out[:,i]=hampel_filter(data[:,i],window_size=w_size,threshold=thres)
    return out

#加速版滤波
def hampel_filter_all_acc(data, window_size, threshold):
    medians = np.median(data, axis=0)
    deviations = np.abs(data - medians)
    med_dev = np.median(deviations, axis=0)
    scale_factor = 1.4826  # Scale factor for MAD to approximate standard deviation

    threshold_array = scale_factor * med_dev * threshold

    lower_bound = medians - threshold_array
    upper_bound = medians + threshold_array

    filtered_data = np.where((data < lower_bound) | (data > upper_bound), medians, data)
    return filtered_data

def standardize(X_train,X_test):
    # 创建MinMaxScaler对象并在训练集上拟合（计算最小值和最大值）
    scaler = MinMaxScaler()
    scaler.fit(X_train)  # X_train是训练集的特征数据

    # 使用训练集上计算的最小值和最大值来标准化训练集和测试集
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)  # X_test是测试集的特征数据
#给定315个(m,n)的数据，每列取其24个特征返回(315,n*24)的数据
def get_feature(signal,simple=0):
    size,b=signal.shape
    features=[]
    for i in range(b):
        data = signal[:, i]
        if simple==1:
            max = np.max(data)
            min=np.min(data)
            mean=np.mean(data)
            f=[max,min,mean]
            features.extend(f)
        else:
            absolute_mean_value = np.sum(np.fabs(data)) / size
            # 峰值
            max = np.max(data)
            # 均方根值
            root_mean_score = np.sqrt(np.sum(np.square(data)) / size)
            # 方根幅值
            Root_amplitude = np.square(np.sum(np.sqrt(np.fabs(data))) / size)
            # 歪度值
            skewness = np.sum(np.power((np.fabs(data) - absolute_mean_value), 3)) / size
            # 峭度值
            Kurtosis_value = np.sum(np.power(data, 4)) / size
            # 波形因子
            shape_factor = root_mean_score / absolute_mean_value
            # 脉冲因子
            pulse_factor = max / absolute_mean_value
            # 歪度因子
            skewness_factor = skewness / np.power(root_mean_score, 3)
            # 峰值因子
            crest_factor = max / root_mean_score
            # 裕度因子
            clearance_factor = max / Root_amplitude
            # 峭度因子
            Kurtosis_factor = Kurtosis_value / np.power(root_mean_score, 4)

            # 频域
            data_fft = np.fft.fft(data)
            Y = np.abs(data_fft)
            freq = np.fft.fftfreq(size, 1 / 50000)
            ps = Y ** 2 / size  # 功率谱密度
            # 重心频率
            FC = np.sum(freq * ps) / np.sum(ps)
            # 均方频率
            MSF = np.sum(ps * np.square(freq)) / np.sum(ps)
            # 均方根频率
            RMSF = np.sqrt(MSF)
            # 频率方差
            VF = np.sum(np.square(freq - FC) * ps) / np.sum(ps)

            # 时频域，小波=db3，模式=对称，最大级别=3
            wp = pywt.WaveletPacket(data, wavelet='db3', mode='symmetric', maxlevel=3)
            aaa = wp['aaa'].data
            aad = wp['aad'].data
            ada = wp['ada'].data
            add = wp['add'].data
            daa = wp['daa'].data
            dad = wp['dad'].data
            dda = wp['dda'].data
            ddd = wp['ddd'].data
            ret1 = np.linalg.norm(aaa, ord=None)
            ret2 = np.linalg.norm(aad, ord=None)
            ret3 = np.linalg.norm(ada, ord=None)
            ret4 = np.linalg.norm(add, ord=None)
            ret5 = np.linalg.norm(daa, ord=None)
            ret6 = np.linalg.norm(dad, ord=None)
            ret7 = np.linalg.norm(dda, ord=None)
            ret8 = np.linalg.norm(ddd, ord=None)

            f = [absolute_mean_value, max, root_mean_score, Root_amplitude, skewness, Kurtosis_value,
                 shape_factor, pulse_factor, skewness_factor, crest_factor, clearance_factor, Kurtosis_factor,
                 FC, MSF, RMSF, VF,
                 ret1, ret2, ret3, ret4, ret5, ret6, ret7, ret8]
            features.extend(f)
    return features
def create_subsequences(signal, num_subsequences):
    signal_length = len(signal)
    window_size = signal_length // num_subsequences
    subsequences = []

    # 计算剩余的不足窗口大小的信号长度
    remaining_length = signal_length % window_size

    # 去除开头和结尾以确保剩下的部分是整数个窗口
    if remaining_length > 0:
        remove_from_start = remaining_length // 2
        remove_from_end = remaining_length - remove_from_start
        signal = signal[remove_from_start:signal_length - remove_from_end]

    for i in range(0, len(signal), window_size):
        subsequence = signal[i:i + window_size]
        subsequences.append(subsequence)

    return subsequences


import numpy as np
import scipy.signal as signal


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

from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler
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


def plot_signals(raw_data, another_data):
    fig, axs = plt.subplots(2, 3, figsize=(12, 5), sharex=True, sharey=True)

    # 第一行子图
    axs[0, 0].plot(raw_data[0:200000, 1])
    # axs[0, 0].set_title('First 800000 samples')
    axs[0, 0].set_xlim(0, 200000)

    axs[0, 1].plot(raw_data[2000000:3000000, 1])
    # axs[0, 1].set_title('800000 to 1500000 samples')
    axs[0, 1].set_xlim(0, 1000000)

    axs[0, 2].plot(raw_data[5000000:, 1])
    # axs[0, 2].set_title('1500000 to 2500000 samples')
    axs[0, 2].set_xlim(0,200000)

    axs[0, 0].set_ylabel('Raw signal')

    # 第二行子图
    axs[1, 0].plot(another_data[0:200000, 1])
    # axs[1, 0].set_title('First 800000 samples after filtering')
    axs[1, 0].set_xlim(0,200000)

    axs[1, 1].plot(another_data[2000000:3000000, 1])
    # axs[1, 1].set_title('800000 to 1500000 samples after filtering')
    axs[1, 1].set_xlim(0,1000000)

    axs[1, 2].plot(another_data[-200000:, 1])
    # axs[1, 2].set_title('1500000 to 2500000 samples after filtering')
    axs[1, 2].set_xlim(0,200000)

    axs[1, 0].set_ylabel('Filtered signal')

    # Adjust layout
    plt.tight_layout()
    plt.show()
# 多项式最小二乘法拟合基线漂移
def polynomial_baseline_correction(signal, degree=4):
    t = len(signal)
    poly_coefficients = np.polyfit(range(t), signal, degree)
    poly_fit = np.poly1d(poly_coefficients)
    baseline_drift_fit = poly_fit(range(t))
    corrected_signal = signal - baseline_drift_fit
    return corrected_signal, baseline_drift_fit

# 差分处理方法校正基线漂移
def difference_baseline_correction(signal):
    diff_signal = np.diff(signal)
    baseline_estimate = np.mean(diff_signal)
    corrected_signal = signal - baseline_estimate * np.arange(len(signal))
    return corrected_signal, None  # 差分处理不需要返回基线漂移

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
    print(len(f_list))
    print(len(v_list))
    #读取力数据
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
    '''
    #去除离群值
    processed_signal=hampel_filter(data[keep_indices][:, 1],5000,1.5)
    plt.figure(figsize=(12, 3))
    plt.plot(data[keep_indices][:, 1], label='processed')
    plt.figure(figsize=(12, 3))
    plt.plot(processed_signal, label='processed data after remove the outliers')
    plt.xlabel('Sample data point')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    plt.show()
    # plt.figure(figsize=(12, 3))
    # plt.plot(data[keep_indices,1],label='Signal after remove the invalid data')
    # plt.show()
    '''
    #零点漂移
    # 移动平均方法结果
    # 使用移动平均方法校正基线漂移
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





    '''
    #绘制滤波前后的图
    path_list = get_allpath(get_toolpath(1))
    data = pd.read_csv(path_list[0]).values
    num_columns = data.shape[1]
    filtered_data=hampel_filter_all_acc(data,10000,3)
    for i in range(num_columns):
        plt.figure(figsize=(10, 4))

        # 绘制原始数据
        plt.subplot(2, 1, 1)
        plt.plot(data[:, i], label=f'Original Data')
        plt.xlabel('Sample',fontsize=12)
        plt.ylabel('Value',fontsize=12)
        plt.legend(fontsize=12)
        plt.tight_layout()
        # 绘制滤波后的数据
        plt.subplot(2, 1, 2)
        plt.plot(filtered_data[:, i], label=f'Filtered Data', color='orange')
        plt.xlabel('Sample',fontsize=12)
        plt.ylabel('Value',fontsize=12)
        plt.legend(fontsize=12)

        plt.suptitle(f'Comparison Result ',fontsize=12)
        plt.tight_layout()
    plt.show()
    '''
    '''
    #1.对原始数据进行前后截断、离群值去除操作
    for tool in [1,4,6]:
        path_list=get_allpath(get_toolpath(tool))
        for i in range(315):
            #获取每个文件的pd
            data=pd.read_csv(path_list[i])
            #获取在截断时表现良好的第3列
            ori=data.iloc[:,2]
            #返回被截断的开始和结束索引位置
            _,st,end=trim_milling_data(ori)
            #对截断后的数据进行hampel 利用加速版
            out=hampel_filter_all_acc(data.iloc[st:end,:].values,10000,3)
            #将out保存到指定位置
            np.save(r'D:\minist_test\data_process_20231011\_1remove_st-end_outliers\no_norm\c{}\c{}_{:03}.npy'.format(tool,tool,i+1),out)
            print(tool,i)
    '''
    #2.取特征
    '''
    #（1）不划分子序列
    for tool in [1,4,6]:
        f=[]
        for j in range(315):
            signal=np.load(r'D:\minist_test\data_process_20231011\_1remove_st-end_outliers\no_norm\c{}\c{}_{:03}.npy'.format(tool,tool,j+1))
            features=get_feature(signal)
            f.append(features)
            print(j)
        c=np.array(f)
        print(c.shape)
        np.save(r'D:\minist_test\data_process_20231011\_1remove_st-end_outliers\data_315_168_no_norm\c{}_315_168_nonorm.npy'.format(tool),c)
        print(tool,j)
    '''
    '''
    # 测试划分子序列的效果
    data=np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,17,18])
    ndata=create_subsequences(data,5)
    print(ndata)#[array([2, 3, 4]), array([5, 6, 7]), array([ 8,  9, 10]), array([11, 12, 13]), array([14, 15, 17])]
    '''
    # (2)划分子序列
    '''
    # <1>取简单特征：20-120个子序列
    for sub in [20,40,60,80,100,120]:#划分子序列的数目
        for tool in [1, 4, 6]:#刀具的编号
            features = []#用于存放最后的数据（315，sub，feature数目）
            for j in range(315):#每把刀中的每个加工文件的处理
                #signal是读取了去除了前后信号以及离群值的每个加工文件数据
                signal = np.load(r'D:\minist_test\data_process_20231011\_1remove_st-end_outliers\no_norm\c{}\c{}_{:03}.npy'.format(tool,tool,j + 1))
                data_sub=create_subsequences(signal,sub)#按照sub数目划分子序列，共得到sub个子序列（sub，7）即sub行7列
                f=[]#用于存放每个文件的21个特征（sub，21）即每列取3个特征：最大值、最小值、平均值，一共7列，共21列特征
                for s in range(sub):#对每个子序列进行循环取特征
                    feature=get_feature(data_sub[s],simple=1)#取得简单特征，即3个特征
                    f.append(feature)#用列表追加的方式添加到f中
                features.append(f)#将每个子序列的存放21列特征的列表以追加的形式存放到features中（315，sub，feature数目）
                print(j)#打印j便于查看处理进程
            arr=np.array(features)#转换为numpy格式便于处理
            print(arr.shape)#进一步检查维度是否正确
            #保存，最后每把刀的315个数据保存为1个文件，共3把刀，保存为3个文件
            np.save(r'D:\minist_test\data_process_20231011\_1remove_st-end_outliers\data_315_sub_3\c{}_315_{}_3.npy'.format(tool,sub),arr)
            print(sub,tool)#打印进程信息
    '''
    '''
    # <2>取复杂特征：20-120个子序列
    for sub in [20,40,60,80,100,120]:#划分子序列的数目
        for tool in [1, 4, 6]:#刀具的编号
            features = []#用于存放最后的数据（315，sub，feature数目）
            for j in range(315):#每把刀中的每个加工文件的处理
                #signal是读取了去除了前后信号以及离群值的每个加工文件数据
                signal = np.load(r'D:\minist_test\data_process_20231011\_1remove_st-end_outliers\no_norm\c{}\c{}_{:03}.npy'.format(tool,tool,j + 1))
                data_sub=create_subsequences(signal,sub)#按照sub数目划分子序列，共得到sub个子序列（sub，7）即sub行7列
                f=[]#用于存放每个文件的21个特征（sub，21）即每列取3个特征：最大值、最小值、平均值，一共7列，共21列特征
                for s in range(sub):#对每个子序列进行循环取特征
                    feature=get_feature(data_sub[s],simple=0)#取得简单特征，即3个特征
                    f.append(feature)#用列表追加的方式添加到f中
                features.append(f)#将每个子序列的存放21列特征的列表以追加的形式存放到features中（315，sub，feature数目）
                print(j)#打印j便于查看处理进程
            arr=np.array(features)#转换为numpy格式便于处理
            print(arr.shape)#进一步检查维度是否正确
            #保存，最后每把刀的315个数据保存为1个文件，共3把刀，保存为3个文件
            np.save(r'D:\minist_test\data_process_20231011\_1remove_st-end_outliers\data_315_sub_24\c{}_315_{}_24.npy'.format(tool,sub),arr)
            print(sub,tool)#打印进程信息
    '''




    # <1>取简单特征-30

    # <1>取简单特征-30

    # <1>取简单特征-30

    # <1>取简单特征-30

    # <1>取简单特征-30
    '''
    #对数据进行标准化处理，此处使用min-max标准化
    # 创建MinMaxScaler对象并在训练集上拟合（计算最小值和最大值）
    scaler = MinMaxScaler()
    scaler.fit(X_train)  # X_train是训练集的特征数据

    # 使用训练集上计算的最小值和最大值来标准化训练集和测试集
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)  # X_test是测试集的特征数据
    '''