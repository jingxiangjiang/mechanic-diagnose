import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import norm
mpl.use('TkAgg')

def get_diagonals(input):
    diagonals = [input[::-1, :].diagonal(i) for i in range(-input.shape[0] + 1, 1)]
    return diagonals

def standardize(data):
    scaler = StandardScaler()
    data = data.reshape(-1, 1)
    data = scaler.fit_transform(data)
    print('data mean %f, data variance %f' % (float(np.mean(data)), float(np.var(data))))
    return data, scaler

def prepare_seq2seq_data(dataset, look_back, look_ahead):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - look_ahead):
        input_seq = dataset[i:(i + look_back)]
        output_seq = dataset[i + look_back:(i + look_back + look_ahead)]
        dataX.append(input_seq)
        dataY.append(output_seq)
    dataX = np.reshape(np.array(dataX), [-1, look_back, 1])
    dataY = np.reshape(np.array(dataY), [-1, look_ahead, 1])
    return dataX, dataY


def load_data(data_folder, look_back, look_ahead):
    print('loading data...')
    train = np.float64(np.load(data_folder + 'train.npy'))
    validation1 = np.float64(np.load(data_folder + "validation1.npy"))
    validation2 = np.float64(np.load(data_folder + "validation2.npy"))

    test = np.float64(np.load(data_folder + "test.npy"))

    train, train_scaler = standardize(train[:, 0])
    validation1 = train_scaler.transform(validation1[:, 0].reshape(-1, 1))
    validation2_labels = validation2[:, 1]
    validation2 = train_scaler.transform(validation2[:, 0].reshape(-1, 1))
    test_labels = test[:, 1]
    test = train_scaler.transform(test[:, 0].reshape(-1, 1))

    X_train, y_train = prepare_seq2seq_data(train, look_back, look_ahead)
    X_validation1, y_validation1 = prepare_seq2seq_data(validation1, look_back, look_ahead)
    X_validation2, y_validation2 = prepare_seq2seq_data(validation2, look_back, look_ahead)
    X_validation2_labels, y_validation2_labels = prepare_seq2seq_data(
        validation2_labels, look_back, look_ahead
    )
    X_test, y_test = prepare_seq2seq_data(test, look_back, look_ahead)
    X_test_labels, y_test_labels = prepare_seq2seq_data(test_labels, look_back, look_ahead)
    return train_scaler, X_train, y_train, X_validation1, y_validation1, \
           X_validation2, y_validation2, y_validation2_labels, X_test, y_test, y_test_labels