import tensorflow as tf
from keras.models import Model
from keras.layers import *
from keras.optimizers import Adam
import os
MODEL_SAVE_PATH = './model'
MODEL_NAME = 'model.h5'
INPUT_NUMS = 2600
CLASS_NAMES = 5
n_epoch = 15
batch_size = 16


def block(inputs, fil_count):
    with tf.name_scope('out_a'):
        out_a1 = Convolution1D(fil_count, 3, padding='SAME',
                             dilation_rate=1, activation='relu')(inputs)
        out_a2 = Convolution1D(fil_count, 5, padding='SAME',
                               dilation_rate=1, activation='relu')(inputs)
        out_a = add([out_a1, out_a2])
    with tf.name_scope('out_b'):
        out_b1 = Convolution1D(fil_count, 5, padding='SAME',
                               dilation_rate=2, activation='relu')(out_a)
        out_b2 = Convolution1D(fil_count, 7, padding='SAME',
                               dilation_rate=2, activation='relu')(out_a)
        out_b = add([out_b1, out_b2])
    if int(inputs.shape[-1]) != int(fil_count):
        inputs = Convolution1D(fil_count, 1, padding='SAME')(inputs)

    output = add([out_b, inputs])

    return output


def modelbone(x_train, y_train, x_valid, y_valid):
    with tf.name_scope('input_data'):
        inputs = Input(shape=(INPUT_NUMS, 1))
    with tf.name_scope('block1'):
        output = block(inputs, 16)
        output = MaxPooling1D(2)(output)
        output = block(output, 16)
        output = MaxPooling1D(2)(output)
    with tf.name_scope('block2'):
        output = block(output, 32)
        output = MaxPooling1D(2)(output)
        output = block(output, 32)
        output = MaxPooling1D(2)(output)
    with tf.name_scope('block3'):
        output = block(output, 64)
        output = MaxPooling1D(2)(output)
        output = block(output, 64)
        output = MaxPooling1D(2)(output)
    with tf.name_scope('block4'):
        output = block(output, 128)
        output = MaxPooling1D(2)(output)
    with tf.name_scope('dense'):
        output = Dropout(0.5)(output)
        output = GlobalAveragePooling1D()(output)
        output = Dense(64, activation='relu')(output)
        output_tensor = Dense(CLASS_NAMES, activation='softmax')(output)
        model = Model(inputs=[inputs], outputs=[output_tensor])
        model.compile(loss='categorical_crossentropy', optimizer=Adam(0.0001), metrics=['accuracy'])
        model.summary()
        model.fit(x_train, y_train, epochs=n_epoch, validation_data=(x_valid, y_valid),
                  batch_size=batch_size)
        model.save(os.path.join(MODEL_SAVE_PATH, MODEL_NAME))

