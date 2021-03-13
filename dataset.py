'''
Description: 
Author: nuso
LastEditors: nuso
Date: 2021-03-12 16:10:04
LastEditTime: 2021-03-12 21:48:24
'''

import os
import torch
from PIL import Image
from torch.utils.data import Dataset


class cub_dataset(Dataset):

    def __init__(self, path, train_test='train', transform=None):
        imgs = []
        with open(os.path.join(path, f'{train_test}.txt'), 'r') as f:
            for line in f.readlines():
                line = line.strip('\n').rstrip()
                words = line.split('.')
                imgs.append((line, int(words[0])-1))

        self.imgs = imgs
        self.transform = transform
        self.path = path

    def __getitem__(self, index):
        fn, label = self.imgs[index]

        img = Image.open(os.path.join(self.path, 'images',  fn))
        img = img.convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.imgs)
