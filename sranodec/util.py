import os
import pickle
from sklearn.preprocessing import MinMaxScaler

import numpy as np


def series_filter(values, kernel_size=3):
    """
    过滤时间序列。实际上，计算内核大小内的平均值。
    As math formula, see https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html.
    :param values:
    :param kernel_size:
    :return: The list of filtered average
    """
    filter_values = np.cumsum(values, dtype=float)

    filter_values[kernel_size:] = filter_values[kernel_size:] - filter_values[:-kernel_size]
    filter_values[kernel_size:] = filter_values[kernel_size:] / kernel_size

    for i in range(1, kernel_size):
        filter_values[i] /= i + 1

    return filter_values


def extrapolate_next(values):
    """
    通过将最后一个值与先前值的斜率相加来推断下一个值。
    :param values: a list or numpy array of time-series
    :return: the next value of time-series
    """

    last_value = values[-1]
    slope = [(last_value - v) / i for (i, v) in enumerate(values[::-1])]
    slope[0] = 0
    next_values = last_value + np.cumsum(slope)

    return next_values


def marge_series(values, extend_num=5, forward=5):
    next_value = extrapolate_next(values)[forward]
    extension = [next_value] * extend_num

    if isinstance(values, list):
        marge_values = values + extension
    else:
        marge_values = np.append(values, extension)
    return marge_values


def average_filter(values, n=3):
    """
    计算给定时间序列的滑动窗口平均值。
    Mathematically, res[i] = sum_{j=i-t+1}^{i} values[j] / t, where t = min(n, i+1)
    :param values: list.
        a list of float numbers
    :param n: int, default 3.
        window size.
    :return res: list.
        a list of value after the average_filter process.
    """

    if n >= len(values):
        n = len(values)

    res = np.cumsum(values, dtype=float)
    res[n:] = res[n:] - res[:-n]
    res[n:] = res[n:] / n

    for i in range(1, n):
        res[i] /= (i + 1)

    return res


def get_data(dataset, max_train_size=None, max_test_size=None,
             normalize=False, spec_res=False, train_start=0, test_start=0):
    """
    Get data from pkl files

    return shape: (([train_size, x_dim], [train_size] or None), ([test_size, x_dim], [test_size]))
    Method from OmniAnomaly (https://github.com/NetManAIOps/OmniAnomaly)
    """
    if max_train_size is None:
        train_end = None
    else:
        train_end = train_start + max_train_size

    print("load data of:", dataset)
    print("train: ", train_start, train_end)
    x_dim = get_data_dim(dataset)
    f = open(os.path.join(dataset + "_train.pkl"), "rb")
    train_data = pickle.load(f).reshape((-1, x_dim))[train_start:train_end, :]
    f.close()

    if normalize:
        train_data, scaler = normalize_data(train_data, scaler=None)

    print("train set shape: ", train_data.shape)
    return train_data


def get_data_dim(dataset):
    """
    :param dataset: Name of dataset
    :return: Number of dimensions in data
    """
    if dataset == "SMAP":
        return 25
    elif dataset == "MSL":
        return 55
    elif str(dataset).startswith("machine"):
        return 38
    else:
        raise ValueError("unknown dataset " + str(dataset))


def normalize_data(data, scaler=None):
    data = np.asarray(data, dtype=np.float32)
    if np.any(sum(np.isnan(data))):
        data = np.nan_to_num(data)

    if scaler is None:
        scaler = MinMaxScaler()
        scaler.fit(data)
    data = scaler.transform(data)
    print("Data normalized")

    return data, scaler


def substitute(data, index):
    '''
    将训练集的异常值替换为周围的正常值
    :param data:训练集
    :param index:异常值下标
    :return:data
    '''
    for i in index:
        a = i
        while a >= 0:
            if a not in index:
                break
            a -= 1
        t = a
        a = i
        while a < len(data)-1:
            if a not in index:
                break
            a += 1
        data[i] = (data[a] + data[t]) / 2
    return data
# def substitute(data, index):
#     '''
#     将训练集的异常值替换为周围的正常值
#     :param data:训练集
#     :param index:异常值下标
#     :return:data
#     '''
#     for i in index:
#         data[i] = extrapolate_next(data[i - 5:i])
#     return data
