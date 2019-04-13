import Inception_Resnet_V2
from Inception_Resnet_V2 import forward
import Read_TFData
import tensorflow as tf
import numpy as np
import os
MODEL_SAVE_PATH = './pic/model'
MODEL_NAME = 'model.ckpt'
config = tf.ConfigProto(allow_soft_placement=True)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
config.gpu_options.allow_growth = True
epoch = 4
batch_size = 10
num_class = 6
filename = './train.tfrecords'
filenamev = './validation.tfrecords'
with tf.Graph().as_default():
    with tf.name_scope('input'):
        xs = tf.placeholder(tf.float32, [batch_size, 299, 299, 3])
        ys = tf.placeholder(tf.int32, [batch_size])
        keep = tf.placeholder(tf.float32)
    with tf.name_scope('forward'):
        y_ = forward(xs, num_class, keep)
    with tf.name_scope('loss'):
        entropy_cross = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=y_, labels=ys)
        loss = tf.reduce_mean(entropy_cross)
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        global_step = tf.Variable(initial_value=0, trainable=False,
                                  name='global_Step', dtype=tf.int64)
        train_op = optimizer.minimize(loss, global_step)
    with tf.name_scope('eval'):
        eval = tf.nn.in_top_k(y_, ys, k=1)
    saver = tf.train.Saver()
    with tf.Session(config=config) as sess:
        img, label = Read_TFData.read_and_decode(filename)
        img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                        batch_size,
                                                        num_threads=64,
                                                        min_after_dequeue=1000,
                                                        capacity=1000 + 3 * batch_size)
        img_val, label_val = Read_TFData.read_and_decode(filenamev)
        img_valid, label_valid = tf.train.shuffle_batch([img_val, label_val],
                                                        batch_size,
                                                        num_threads=64,
                                                        min_after_dequeue=1000,
                                                        capacity=1000 + 3 * batch_size)
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        batch_idx = int(2314/batch_size)
        for i in range(epoch):
            for j in range(batch_idx):
                x, y = sess.run([img_batch, label_batch])
                _, curr_loss, accuracy = sess.run([train_op, loss, eval], feed_dict={xs: x, ys: y, keep: 0.6})
                curr_step = sess.run(global_step)
                curr_accuracy = np.sum(accuracy) / batch_size
                print('epoch:[%d][%d/%f], accuracy:[%f]'% (i, curr_step, curr_loss, curr_accuracy))

                x_val, y_val = sess.run([img_valid, label_valid])
                curr_loss, accuracy = sess.run([loss, eval], feed_dict={xs: x_val, ys: y_val, keep: 1})
                accuracy = np.sum(accuracy) / batch_size
                print('validation accuracy:[.%8f]' % (accuracy))
        saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME))
        coord.request_stop()
        coord.join(threads)




