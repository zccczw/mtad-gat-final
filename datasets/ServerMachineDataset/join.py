import os
import pandas as pd
from os import listdir, makedirs, path
from pickle import dump
import numpy as np
import sranodec as anom

def loop_directory(directory: str):
    '''循环目录中的文件'''
    test = []
    train = []
    label = []
    for filename in os.listdir(directory):
        if filename.split('.')[0].endswith("test"):
            data = pd.read_pickle(path.join(directory, filename))
            test.append(data)
        elif filename.split('.')[0].endswith("train"):
            data = pd.read_pickle(path.join(directory, filename))
            train.append(data)
        else:
            data = pd.read_pickle(path.join(directory, filename))
            label.append(data)
    train = np.vstack(train)    #将数组拼接起，它是垂直（按照行顺序）的把数组给堆叠起来。

    """
    import numpy as np
a=[1,2,3]
b=[4,5,6]
print(np.vstack((a,b)))

输出：
[[1 2 3]
 [4 5 6]]
    """

    # with open(path.join(directory, "train.pkl"), "wb") as file:
    #     dump(train, file)
    # test = np.vstack(test)
    # with open(path.join(directory, "test.pkl"), "wb") as file:
    #     dump(test, file)
    # label = np.hstack(label)
    # with open(path.join(directory, "label.pkl"), "wb") as file:
    #     dump(label, file)
    # less than period
    amp_window_size = 3
    # (maybe) as same as period
    series_window_size = 5
    # a number enough larger than period
    score_window_size = 100
    spec = anom.Silency(amp_window_size, series_window_size, score_window_size)
    # 獲取異常分數
    for i in range(38):
        score = spec.generate_anomaly_score(train[:, i])
        index_changes = np.where(score > np.percentile(score, 90))[i]
        train[:, i] = anom.substitute(train[:, i], index_changes)


if __name__ == '__main__':
    loop_directory('processed/')
