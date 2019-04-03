# ！/usr/bin/env python
# -*- coding: utf-8 -*-
# Author:Carton  Time=2019/4/1


import random
_NUM_VALIDATION = 350
_RANDOM_SEED = 0

list_path = './wafer/wafer.txt'
train_list_path = 'list_train.txt'
val_list_path = 'list_val.txt'

fd = open(list_path)
lines = fd.readlines()
fd.close()

random.seed(_RANDOM_SEED)
random.shuffle(lines)
# 写入train_list
fd = open(train_list_path, 'w')
for line in lines[_NUM_VALIDATION:]:
    fd.write(line)
fd.close()

# 写入 val_list
fd = open(val_list_path, 'w')
for line in lines[:_NUM_VALIDATION]:
    fd.write(line)
fd.close()