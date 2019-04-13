import tensorflow as tf


def read_and_decode(filename):
    reader = tf.TFRecordReader()
    filename_queue = tf.train.string_input_producer([filename])
    filename, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example, features={
        'label': tf.FixedLenFeature([], tf.int64),
        'img_raw': tf.FixedLenFeature([], tf.string)
    })
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [299, 299, 3])
    img = tf.cast(img, tf.float32)*(1./255) - 0.5
    label = tf.cast(features['label'], tf.int32)
    return img, label

