import scipy.io as io
import numpy as np
INPUT_NUMS = 2600


def read_onemat(filename, label):
    data = io.loadmat(filename)
    x = []
    y = []
    for w in data.keys():
        if len(data[w]) > 1000:
            i = 0
            while (i+INPUT_NUMS) < len(data[w]):
                x.append(data[w][i: i + INPUT_NUMS])
                y.append([label])
                i = i + INPUT_NUMS

    return np.array(x, dtype=np.float32), np.array(y, dtype=np.float32)



