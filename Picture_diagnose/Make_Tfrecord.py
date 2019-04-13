import os
import tensorflow as tf
from PIL import Image



cwd1 = './pic/train/'
cwd2 = './pic/validation/'
for cwd in [cwd1,cwd2]:
    main_dir = './pic/'
    label_file = os.path.join(main_dir, 'label.txt')

    if os.path.exists(label_file) is False:
        all_category = os.listdir(cwd)
        with open(label_file, 'w') as f:
            for label_name in all_category:
                f.write(label_name + '\n')

    classes = [v.strip() for v in tf.gfile.FastGFile(label_file, 'r').readlines()]
    if cwd == cwd1:
        writer = tf.python_io.TFRecordWriter('./train.tfrecords')
    else:
        writer = tf.python_io.TFRecordWriter('./validation.tfrecords')

    for index, name in enumerate(classes):
        data_path = cwd + name + '/'
        for img_name in os.listdir(data_path):
            img_path = data_path + img_name
            img = Image.open(img_path)
            img = img.resize((299, 299))
            img_raw = img.tobytes()
            example = tf.train.Example(features=tf.train.Features(feature={
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            }))
            writer.write(example.SerializeToString())
    writer.close()



