# ！/usr/bin/env python
# -*- coding: utf-8 -*-
# Author:Carton  Time=2019/3/31


import os

for i in range(9):
    img_name = os.listdir('./wafer/data%d' % (i+1))
    for j,temp in enumerate(img_name):
        new_name ='%d_' %(i+1) +str(j).zfill(3)+ '.jpg'
        os.rename('./wafer/data%d' % (i+1) + '/'+temp, './wafer/data%d' % (i+1) + '/' + new_name)
