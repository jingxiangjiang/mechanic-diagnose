
from keras.layers.core import Dense, Activation, Dropout, RepeatVector
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import Adam, RMSprop
from keras.callbacks import EarlyStopping, Callback, TensorBoard, ModelCheckpoint
import keras.backend as K
import shutil
import os


class ResetStatesCallback(Callback):
    def on_epoch_begin(self, epoch, logs=None):
        print('resetting states before epoch %d' % epoch)
        self.model.reset_states()


class GetStatesCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        print('on epoch end.epoch number: %d' % epoch)
        states = [K.get_value(x) for x, _ in self.model.state_updates]
        print(states)

    def on_batch_end(self, batch, logs=None):
        print('one batch end. batch number: %d' % batch)
        states = [K.get_value(x) for x, _ in self.model.state_updates]
        print(states)


class StatefulMultiStepLSTM(object):
    def __init__(self, batch_size, look_back, look_ahead, layers, dropout, loss, learning_rate):
        self.batch_size = batch_size
        self.look_back = look_back
        self.look_ahead = look_ahead
        self.n_hidden = len(layers) - 2
        self.model = Sequential()
        self.layers = layers
        self.loss = loss
        self.learning_rate = learning_rate
        self.dropout = dropout

    def build_model(self):
        self.model.add(LSTM(
            units=self.layers['hidden1'],
            batch_input_shape=(self.batch_size, self.look_back, self.layers['input']),
            stateful=True,
            unroll=True,
            return_sequences=True if self.n_hidden > 1 else False
        ))
        self.model.add(Dropout(self.dropout))
        for i in range(2, self.n_hidden + 1):
            return_sequence = True
            if i == self.n_hidden:
                return_sequence = False
            self.model.add(LSTM(units=self.layers['hidden' + str(i)], stateful=True, return_sequences=return_sequence))

        self.model.add(Dense(units=self.layers['output']))
        self.model.add(RepeatVector(self.look_ahead))
        self.model.add(Activation('linear'))
        self.model.compile(loss=self.loss, optimizer=Adam(lr=self.learning_rate, decay=.99))
        self.model.summary()
        return self.model


def train_stateful_model(model, x_train, y_train, batch_size, epochs, shuffle, validation,
                         validation_data, patience):
    print('training...')
    try:
        shutil.rmtree('checkpoints')
    except:
        pass
    os.mkdir('checkpoints')
    checkpoint = ModelCheckpoint(monitor='val_acc',
                                 filepath='checkpoints/model_{epoch:02d}_{val_acc:.3f}.h5',
                                 save_best_only=True)
    if validation:
        early_stopping = EarlyStopping(monitor='val_loss', patience=patience)
        history_callback = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
                                     validation_data=validation_data, shuffle=shuffle, verbose=2,
                                     callbacks=[ResetStatesCallback(), early_stopping])
    else:
        history_callback = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
                                      shuffle=shuffle, verbose=2,
                                     callbacks=[ResetStatesCallback()])
    print("Training Loss per epoch: %s" % str(history_callback.history["loss"]))
    if validation:
        print("Validation  Loss per epoch: %s" % str(history_callback.history["val_loss"]))
    print(history_callback.history.keys())
    return history_callback


def get_states(model):
    return [K.get_value(s) for s, _ in model.state_updates]


def set_states(model, states):
    for (d, _), s in zip(model.state_updates, states):
        K.set_value(d, s)

