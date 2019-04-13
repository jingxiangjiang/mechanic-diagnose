import tensorflow as tf
import tensorflow.contrib.slim as slim


def stem(inputs):
    with slim.arg_scope([slim.conv2d], stride=1, padding='SAME'):
        with tf.name_scope('stem_part'):
            out = slim.conv2d(inputs, 32, [3, 3], stride=2, padding='VALID')
            out = slim.conv2d(out, 32, [3, 3], padding='VALID')
            out = slim.conv2d(out, 64, [3, 3])

            out_branch1 = slim.max_pool2d(out, [3, 3], stride=2)
            out_branch2 = slim.conv2d(out, 96, [3, 3], stride=2, padding='VALID')
            out2 = tf.concat(values=[out_branch1, out_branch2], axis=3)

            out2_branch1 = slim.conv2d(out2, 64, [1, 1])
            out2_branch1 = slim.conv2d(out2_branch1, 96, [3, 3], padding='VALID')
            out2_branch2 = slim.conv2d(out2, 64, [1, 1])
            out2_branch2 = slim.conv2d(out2_branch2, 64, [7, 1])
            out2_branch2 = slim.conv2d(out2_branch2, 64, [1, 7])
            out2_branch2 = slim.conv2d(out2_branch2, 96, [3, 3], padding='VALID')
            out3 = tf.concat(values=[out2_branch1, out2_branch2], axis=3)

            out3_branch1 = slim.conv2d(out3, 192, [3, 3], stride=2, padding='VALID')
            out3_branch2 = slim.max_pool2d(out3, [3, 3], stride=2)
            output = tf.concat(values=[out3_branch1, out3_branch2], axis=3)

            return tf.nn.relu(output)


def inception_resnet_a(inputs):
    with slim.arg_scope([slim.conv2d], stride=1, padding='SAME'):
        with tf.name_scope('Inc_Res_A'):
            with tf.name_scope('out_res_a'):
                out_res_a = tf.identity(inputs)
            with tf.name_scope('out_a'):
                out_a = slim.conv2d(inputs, 32, [1, 1])
                out_a = slim.conv2d(out_a, 384, [1, 1], activation_fn=None)
            with tf.name_scope('out_b'):
                out_b = slim.conv2d(inputs, 32, [1, 1])
                out_b = slim.conv2d(out_b, 32, [3, 3])
                out_b = slim.conv2d(out_b, 384, [1, 1], activation_fn=None)
            with tf.name_scope('out_c'):
                out_c = slim.conv2d(inputs, 32, [1, 1])
                out_c = slim.conv2d(out_c, 48, [3, 3])
                out_c = slim.conv2d(out_c, 64, [3, 3])
                out_c = slim.conv2d(out_c, 384, [1, 1], activation_fn=None)
            with tf.name_scope('out_res_b'):
                out_res_b = tf.add_n([out_a, out_b, out_c])
                out_res_b = tf.multiply(out_res_b, 0.15)
            with tf.name_scope('output'):
                output = tf.add_n([out_res_a, out_res_b])

            return tf.nn.relu(output)


def reduction_a(inputs):
    with slim.arg_scope([slim.conv2d], stride=1, padding='SAME'):
        with tf.name_scope('Red_A'):
            with tf.name_scope('out_a'):
                out_a = slim.max_pool2d(inputs, [3, 3], stride=2, padding='VALID')
            with tf.name_scope('out_b'):
                out_b = slim.conv2d(inputs, 384, [3, 3], stride=2, padding='VALID')
            with tf.name_scope('out_c'):
                out_c = slim.conv2d(inputs, 256, [1, 1])
                out_c = slim.conv2d(out_c, 256, [3, 3])
                out_c = slim.conv2d(out_c, 384, [3, 3], stride=2, padding='VALID')
            with tf.name_scope('output'):
                output = tf.concat(values=[out_a, out_b, out_c], axis=3)

            return tf.nn.relu(output)


def inception_resnet_b(inputs):
    with slim.arg_scope([slim.conv2d], stride=1, padding='SAME'):
        with tf.name_scope('inc_rec_B'):
            with tf.name_scope('out_res_a'):
                out_res_a = tf.identity(inputs)
            with tf.name_scope('out_a'):
                out_a = slim.conv2d(inputs, 192, [1, 1])
                out_a = slim.conv2d(out_a, 1152, [1, 1], activation_fn=None)
            with tf.name_scope('out_b'):
                out_b = slim.conv2d(inputs, 128, [1, 1])
                out_b = slim.conv2d(out_b, 160, [1, 7])
                out_b = slim.conv2d(out_b, 192, [7, 1])
                out_b = slim.conv2d(out_b, 1152, [1, 1], activation_fn=None)
            with tf.name_scope('out_res_b'):
                out_res_b = tf.add_n([out_a, out_b])
                out_res_b = tf.multiply(out_res_b, 0.15)
            with tf.name_scope('output'):
                output = tf.add_n([out_res_a, out_res_b])

            return tf.nn.relu(output)


def reduction_b(inputs):
    with slim.arg_scope([slim.conv2d], stride=1, padding='SAME'):
        with tf.name_scope('Red_B'):
            with tf.name_scope('out_a'):
                out_a = slim.max_pool2d(inputs, [3, 3], stride=2, padding='VALID')
            with tf.name_scope('out_b'):
                out_b = slim.conv2d(inputs, 256, [1, 1])
                out_b = slim.conv2d(out_b, 384, [3, 3], stride=2, padding='VALID')
            with tf.name_scope('out_c'):
                out_c = slim.conv2d(inputs, 256, [1, 1])
                out_c = slim.conv2d(out_c, 288, [3, 3], stride=2, padding='VALID')
            with tf.name_scope('out_d'):
                out_d = slim.conv2d(inputs, 256, [1, 1])
                out_d = slim.conv2d(out_d, 288, [3, 3])
                out_d = slim.conv2d(out_d, 320, [3, 3], stride=2, padding='VALID')
            with tf.name_scope('output'):
                output = tf.concat(values=[out_a, out_b, out_c, out_d], axis=3)

            return tf.nn.relu(output)


def inception_resnet_c(inputs):
    with slim.arg_scope([slim.conv2d], stride=1, padding='SAME'):
        with tf.name_scope('inc_rec_C'):
            with tf.name_scope('out_res_a'):
                out_res_a = tf.identity(inputs)
            with tf.name_scope('out_a'):
                out_a = slim.conv2d(inputs, 192, [1, 1])
                out_a = slim.conv2d(out_a, 2144, [1, 1], activation_fn=None)
            with tf.name_scope('out_b'):
                out_b = slim.conv2d(inputs, 192, [1, 1])
                out_b = slim.conv2d(out_b, 224, [1, 3])
                out_b = slim.conv2d(out_b, 256, [3, 1])
                out_b = slim.conv2d(out_b, 2144, [1, 1], activation_fn=None)
            with tf.name_scope('out_res_b'):
                out_res_b = tf.add_n([out_a, out_b])
                out_res_b = tf.multiply(out_res_b, 0.15)
            with tf.name_scope('output'):
                output = tf.add_n([out_res_a, out_res_b])

            return tf.nn.relu(output)


def average_pooling(inputs):
    with slim.arg_scope([slim.conv2d], stride=1, padding='SAME'):
        with tf.name_scope('average_pooling'):
            output = slim.avg_pool2d(inputs, [8, 8])
            return output


def dropout(inputs, keep=0.8):
    output = slim.dropout(inputs, keep_prob=keep)
    return output


def forward(inputs, num_classes, keep):
    with slim.arg_scope([slim.conv2d],
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                        activation_fn=tf.nn.relu,
                        normalizer_fn=slim.batch_norm):
        with tf.name_scope('Stem'):
            output = stem(inputs)
        with tf.name_scope('5xInception-ResNet-A'):
            for i in range(5):
                output = inception_resnet_a(output)
        with tf.name_scope('reduction-A'):
            output = reduction_a(output)
        with tf.name_scope('10xInception-resnet-B'):
            for i in range(10):
                output = inception_resnet_b(output)
        with tf.name_scope('Reduction-B'):
            output = reduction_b(output)
        with tf.name_scope('5xInception-ResNet-c'):
            for i in range(5):
                output = inception_resnet_c(output)
        with tf.name_scope('average_pooling'):
            output = average_pooling(output)
        with tf.name_scope('Dropout0.8'):
            output = dropout(output, keep)
            output = slim.flatten(output)
        with tf.name_scope('fc'):
            output = slim.fully_connected(output, num_classes)

        return output




