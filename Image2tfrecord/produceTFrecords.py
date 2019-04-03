# ！/usr/bin/env python
# -*- coding: utf-8 -*-
# Author:Carton  Time=2019/3/28

import numpy as np
import cv2
import tensorflow as tf

# Control
resize_height = 128  # 存储图片高度
resize_width = 128  # 存储图片宽度
train_file_root = './wafer/'
train_file = train_file_root + 'wafer.txt'  # train_file是txt文件存放的目录


def _int64_feature(value):  # 将输入value转化成int64字节属性
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):  # 将输入value转化成bytes属性
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


# 加载txt文件，返回example：存储图像path的矩阵 ；
def load_file(examples_list_file):
    #  type (object) -> object
    lines = np.genfromtxt(examples_list_file, delimiter=" ", dtype=[('col1', 'S120'), ('col2', 'i8')])
    examples = []
    labels = []
    for example, label in lines:
        examples.append(example)
        labels.append(label)
    return np.asarray(examples), np.asarray(labels), len(lines)


# 从输入的filename处提取image并进行预处理
def extract_image(filename, resize_height, resize_width):
    filename = filename.decode(encoding='utf-8')  # 把byte类型的filename转为str
    image = cv2.imread(filename)
    image = cv2.resize(image, (resize_height, resize_width))
    b, g, r = cv2.split(image)
    rgb_image = cv2.merge([r, g, b])  # cv2读取是BGR格式，需要转换成RGB格式
    rgb_image = rgb_image / 255.
    rgb_image = rgb_image.astype(np.float32)
    return rgb_image


examples, labels, examples_num = load_file(train_file)

# 定义 TFRecordWriter
writer = tf.python_io.TFRecordWriter('./wafer.tfrecords')

for i, [example, label] in enumerate(zip(examples, labels)):
    print('No.%d' % (i))
    root = examples[i]
    image = extract_image(root, resize_height, resize_width)
    a = image.shape
    print(root)
    print('shape: %d, %d, %d, label: %d' % (image.shape[0], image.shape[1], image.shape[2], label))
    image_raw = image.tostring()  # 将Image转化成字符
    example = tf.train.Example(features=tf.train.Features(feature={
        'image_raw': _bytes_feature(image_raw),
        'height': _int64_feature(image.shape[0]),
        'width': _int64_feature(image.shape[1]),
        'depth': _int64_feature(image.shape[2]),
        'label': _int64_feature(label)
    }))
    writer.write(example.SerializeToString())
writer.close()