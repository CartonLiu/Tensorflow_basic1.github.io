    # ！/usr/bin/env python
# -*- coding: utf-8 -*-
# Author:Carton  Time=2019/4/1
# 测试调用为GPU or CPU

import tensorflow as tf

a = tf.constant([1.,2.,3.,4.,5.,6.], shape=[2,3], name='a')
b = tf.constant([1.,2.,3.,4.,5.,6.], shape=[3,2], name='b')
c = tf.matmul(a,b)

with tf.Session(config= tf.ConfigProto(log_device_placement=True)) as sess:
    print(sess.run(c))
    
# 结果中有GPU 信息 即为用GPU计算
