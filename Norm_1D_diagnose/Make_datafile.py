import os
import tensorflow as tf
import pandas as pd
from Read_from_mat import read_onemat
INPUT_NUMS = 2600
cwd = './data/'


label_file = os.path.join(cwd, 'label.txt')
x_data_file = os.path.join(cwd, 'dataset.csv')
y_label_file = os.path.join(cwd, 'ylabel.csv')
if os.path.exists(label_file) is False:
    all_entries=os.listdir(cwd)
    with open(label_file, 'w') as f:
        for dirname in all_entries:
            f.write(dirname+'\n')
classes = [l.strip() for l in tf.gfile.FastGFile(label_file, 'r').readlines()]


for index, name in enumerate(classes):
    class_path = cwd+name+'/'
    for data_file in os.listdir(class_path):
        data_file = class_path+data_file
        data, label = read_onemat(data_file, index)
        data = data.reshape(len(data), INPUT_NUMS)
        data = pd.DataFrame(data)
        ylabel = pd.DataFrame(label)
        if os.path.exists(x_data_file) is False:
            data.to_csv(x_data_file, header=False)
        else:
            data.to_csv(x_data_file, mode='a', header=False)
        if os.path.exists(y_label_file) is False:
            ylabel.to_csv(y_label_file, header=False)
        else:
            ylabel.to_csv(y_label_file, mode='a', header=False)



