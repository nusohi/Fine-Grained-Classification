'''
Description: 通过 train_test_split.txt 将 images.txt 分为 train.txt 和 test.txt
Author: nuso
LastEditors: nuso
Date: 2021-03-12 18:48:01
LastEditTime: 2021-03-12 19:02:35
'''
import os
import argparse


parser = argparse.ArgumentParser(description='prepare')
parser.add_argument('--path', default='F:\code\细粒度图像识别\data\CUB\CUB_200_2011', type=str)
opt = parser.parse_args()


flags = []
with open(os.path.join(opt.path, 'train_test_split.txt'), 'r') as f:
    for line in f.readlines():
        flags.append(line.strip().split(' ')[1])

train_list = []
test_list = []
with open(os.path.join(opt.path, 'images.txt'), 'r') as f:
    for line in f.readlines():
        index, path = line.split(' ')
        index = int(index)
        if(flags[index-1] == '0'):
            train_list.append(path)
        else:
            test_list.append(path)

with open(os.path.join(opt.path, 'train.txt'), 'w+') as f:
    f.writelines(train_list)

with open(os.path.join(opt.path, 'test.txt'), 'w+') as f:
    f.writelines(test_list)
