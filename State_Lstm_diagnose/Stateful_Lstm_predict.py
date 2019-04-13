import numpy as np
import tensorflow as tf
import random as rn
from keras import backend as K
import Lstm_backbone as lstm
import config as cfg
import matplotlib as mlb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import utils as util
import seaborn as sns
import matplotlib.pylab as pylab
np.random.seed(2019)
rn.seed(2019)
mlb.use('TkAgg')
session_conf = tf.ConfigProto(
    intra_op_parallelism_threads=1,
    inter_op_parallelism_threads=1
)
tf.set_random_seed(2019)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)


def get_predictions(context, model, X, y, train_scaler, batch_size, look_ahead, look_back,
                    epochs, experiment_id):
    predictions = model.predict(X, batch_size=batch_size)
    print(predictions.shape)
    predictions = train_scaler.inverse_transform(predictions)

    y = train_scaler.inverse_transform(y)

    y_true = y[:, 0].flatten()
    print(predictions)
    print(y_true)
    diagonals = util.get_diagonals(predictions)

    for idx, diagonal in enumerate(diagonals):
        diagonal = diagonal.flatten()
        diagonals[idx] = np.hstack((diagonal, np.full(look_ahead - len(diagonal), diagonal[0])))#hstack:[1,2]+[3,4]=[1,2,3,4]
        #np.full(3,e)=[e,e,e]
    predictions_timesteps = np.asarray(diagonals)
    print(predictions)
    print(y_true)
    return predictions_timesteps, y_true


def run():
    experiment_id = cfg.config['experiment_id']
    data_folder = cfg.config['data_folder']
    look_back = cfg.config['look_back']
    look_ahead = cfg.config['look_ahead']
    batch_size = cfg.config['batch_size']
    epochs = cfg.config['n_epochs']
    dropout = cfg.config['dropout']
    layers = cfg.config['layers']
    loss = cfg.config['loss']
    shuffle = cfg.config['shuffle']
    patience = cfg.config['patience']
    validation = cfg.config['validation']
    learning_rate = cfg.config['learning_rate']

    train_scaler, X_train, y_train, X_validation1, y_validation1, X_validation2, y_validation2, validation2_labels, \
    X_test, y_test, test_labels = util.load_data(data_folder, look_back, look_ahead)

    if batch_size > 1:
        n_train_batches = int(len(X_train)/batch_size)
        len_train = n_train_batches * batch_size
        if len_train < len(X_train):
            X_train = X_train[:len_train]
            y_train = y_train[:len_train]
        print(len(X_train))
        n_validation1_batches = int(len(X_validation1)/batch_size)
        len_validation1 = n_validation1_batches * batch_size
        if len_validation1 < len(X_validation1):
            X_validation1 = X_validation1[:len_validation1]
            y_validation1 = y_validation1[:len_validation1]

        n_validation2_batches = int(len(X_validation2) / batch_size)
        len_validation2 = n_validation2_batches * batch_size
        if len_validation2 < len(X_validation2):
            X_validation2 = X_validation2[:len_validation2]
            y_validation2 = y_validation2[:len_validation2]

        n_test_batches = int(len(X_test)/batch_size)
        len_test = n_test_batches * batch_size
        if len_test < len(X_test):
            X_test = X_test[:len_test]
            y_test = y_test[:len_test]

    stateful_lstm = lstm.StatefulMultiStepLSTM(batch_size=batch_size, look_back=look_back, look_ahead=look_ahead,
                                               layers=layers, dropout=dropout, loss=loss, learning_rate=learning_rate)

    model = stateful_lstm.build_model()


    history = lstm.train_stateful_model(model, X_train, y_train, batch_size, epochs,
                                        shuffle, validation, (X_validation1, y_validation1), patience)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    validation2_loss = model.evaluate(X_validation2, y_validation2, batch_size=batch_size, verbose=2)
    print("Validation2 Loss %s" % validation2_loss)
    test_loss = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=2)
    print("Test Loss %s" % test_loss)

    predictions_train, y_true_train = get_predictions("Train", model, X_train, y_train, train_scaler,
                                                        batch_size, look_ahead, look_back, epochs, experiment_id,)

    np.save(data_folder + "train_predictions", predictions_train)
    np.save(data_folder + "train_true", y_true_train)

    predictions_validation1, y_true_validation1 = get_predictions("Validation1", model, X_validation1, y_validation1,
                                                                  train_scaler, batch_size, look_ahead, look_back,
                                                                  epochs, experiment_id,
                                                                  )
    predictions_validation1_scaled = train_scaler.transform(predictions_validation1)
    print("Calculated validation1 loss %f" % (mean_squared_error(
        np.reshape(y_validation1, [len(y_validation1), look_ahead]),
        np.reshape(predictions_validation1_scaled, [len(predictions_validation1_scaled), look_ahead]))))
    np.save(data_folder + "validation1_predictions", predictions_validation1)
    np.save(data_folder + "validation1_true", y_true_validation1)
    np.save(data_folder + "validation1_labels", validation2_labels)

    predictions_validation2, y_true_validation2 = get_predictions("Validation2", model, X_validation2, y_validation2,
                                                                  train_scaler, batch_size, look_ahead, look_back,
                                                                  epochs, experiment_id,
                                                                 )
    predictions_validation2_scaled = train_scaler.transform(predictions_validation2)
    print("Calculated validation2 loss %f"%(mean_squared_error(
        np.reshape(y_validation2, [len(y_validation2), look_ahead]),
        np.reshape(predictions_validation2_scaled, [len(predictions_validation2_scaled), look_ahead]))))
    np.save(data_folder + "validation2_predictions", predictions_validation2)
    np.save(data_folder + "validation2_true", y_true_validation2)
    np.save(data_folder + "validation2_labels", validation2_labels)


    predictions_test, y_true_test = get_predictions("Test", model, X_test, y_test, train_scaler, batch_size, look_ahead,
                                                    look_back, epochs, experiment_id,
                                                   )
    predictions_test_scaled = train_scaler.transform(predictions_test)
    print("Calculated test loss %f" % (mean_squared_error( np.reshape(y_test, [len(y_test),look_ahead]),
                                       np.reshape(predictions_test_scaled, [len(predictions_test_scaled),look_ahead]))))

    np.save(data_folder + "test_predictions", predictions_test)
    np.save(data_folder + "test_true", y_true_test)
    np.save(data_folder + "test_labels", test_labels)


if __name__ == '__main__':
    run()
