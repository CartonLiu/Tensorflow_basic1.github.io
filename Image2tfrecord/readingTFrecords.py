# ！/usr/bin/env python
# -*- coding: utf-8 -*-
# Author:Carton  Time=2019/3/31


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

tfrecord_list_file = './wafer.tfrecords'


def read_and_decode(filename_queue, shuffle_batch=True):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(serialized_example, features={
        'image_raw': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.int64)
    })

    image = tf.decode_raw(features['image_raw'], tf.float32)
    image = tf.reshape(image, [128, 128, 3])
    image = image * 255.0

    labels = features['label']

    if shuffle_batch:
        images, labels = tf.train.shuffle_batch(
            [image, labels],
            batch_size=32,  # 生成的batchsize大小
            capacity=8000,
            num_threads=4,
            min_after_dequeue=2000)
    else:
        images, labels = tf.train.batch([image, labels],
                                        batch_size=4,
                                        capacity=8000,
                                        num_threads=4)
    return images, labels


def test_run(tfrecord_filename):
    """Output strings (e.g. filenames) to a queue for an input pipeline."""
    filename_queue = tf.train.string_input_producer([tfrecord_filename],
                                                    num_epochs=3)
    images, labs = read_and_decode(filename_queue)

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    # meanfile = sio.loadmat(root_path + 'mats/mean300.mat')
    # meanvalue = meanfile['mean']               #如果在制作数据时减去的均值，则需要加上来

    with tf.Session() as sess:
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for i in range(1):
            imgs, labs = sess.run([images, labs])
            print
            'batch' + str(i) + ': '
            # print type(imgs[0])

            for j in range(8):
                print
                str(labs[j])
                img = np.uint8(imgs[j])
                plt.subplot(4, 2, j + 1)
                plt.imshow(img)
            plt.show()

        coord.request_stop()
        coord.join(threads)  # 注意，要关闭文件


test_run('./wafer.tfrecords')
print("has done")