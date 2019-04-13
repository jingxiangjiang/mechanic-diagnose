from Model import *
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from keras.utils.np_utils import *


INPUT_NUMS = 2600
CLASS_NAMES = 5

x_data = pd.read_csv('./data/dataset.csv')
y_label = pd.read_csv('./data/ylabel.csv')

x_data = np.asarray(x_data)
y_label = np.asarray(y_label)
x_data = x_data[:, 1:]
y_label = y_label[:, 1]

scaler = StandardScaler().fit(x_data)
scaled_x_data = scaler.transform(x_data)

shuffled = StratifiedShuffleSplit(test_size=0.1, random_state=2019)
x_train = []
y_train = []
x_valid = []
y_valid = []

for train_index, valid_index in shuffled.split(scaled_x_data, y_label):
    x_train, x_valid = scaled_x_data[train_index], scaled_x_data[valid_index]
    y_train, y_valid = y_label[train_index], y_label[valid_index]

x_train = np.reshape(x_train, [len(x_train), INPUT_NUMS, 1])
x_valid = np.reshape(x_valid, [len(x_valid), INPUT_NUMS, 1])
y_train = to_categorical(y_train, CLASS_NAMES)
y_valid = to_categorical(y_valid, CLASS_NAMES)
modelbone(x_train, y_train, x_valid, y_valid)
