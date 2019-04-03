# ！/usr/bin/env python
# -*- coding: utf-8 -*-
# Author:Carton  Time=2019/3/31


import os


def generate(root, dir, label):
    files = os.listdir(dir)
    files.sort()  # 进行排序
    listText = open('all_list.txt', 'a')  # 'a':打开一个文件用于追加
    for file in files:
        fileType = os.path.split(file)
        if fileType[1] == '.txt':
            continue
        name = root + file + ' ' + str(int(label)) + '\n'  # 写入txt的内容
        listText.write(name)
    listText.close()


outer_path = './wafer'  # 这里是你的图片的目录
if __name__ == '__main__':
    i = 0
    folderlist = os.listdir(outer_path)  # 列举文件夹
    for folder in folderlist:
        root = outer_path + '/' + folder+'/'
        generate(root, os.path.join(outer_path, folder), i+1)
        i += 1