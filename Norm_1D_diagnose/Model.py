import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from keras.models import Sequential
from keras.layers import Dense,Activation,Flatten,Convolution1D,Dropout,MaxPooling1D
from keras.optimizers import Adam
from keras.utils import np_utils
import os
MODEL_SAVE_PATH = './model'
MODEL_NAME = 'model.h5'
INPUT_NUMS = 2600
CLASS_NAMES = 5
n_epoch = 15
batch_size = 16


def modelbone(x_train, y_train, x_valid, y_valid):
    with tf.name_scope('block1'):
        model = Sequential()
        model.add(Convolution1D(32, 7, padding='same', input_shape=(INPUT_NUMS, 1)))
        model.add(Activation('relu'))
        model.add(MaxPooling1D(2, 2, padding='same'))
        model.add(Dropout(0.25))
    with tf.name_scope('block2'):
        model.add(Convolution1D(64, 5, padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling1D(2, 2, padding='same'))
    with tf.name_scope('block3'):
        model.add(Convolution1D(128, 3, padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling1D(2, 2, padding='same'))
        model.add(Dropout(0.5))
    with tf.name_scope('dense'):
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dense(CLASS_NAMES, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=Adam(0.0001), metrics=['accuracy'])
        model.summary()
        model.fit(x_train, y_train, epochs=n_epoch, validation_data=(x_valid, y_valid),
                  batch_size=batch_size)
        model.save(os.path.join(MODEL_SAVE_PATH, MODEL_NAME))

