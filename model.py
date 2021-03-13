'''
Description:
Author: nuso
LastEditors: nuso
Date: 2021-03-12 15:55:30
LastEditTime: 2021-03-12 19:55:08
'''
import torch
import torch.nn as nn
# from torch.nn import init
from torchvision import models


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.features = nn.Sequential(
            models.resnet50().conv1,
            models.resnet50().bn1,
            models.resnet50().relu,
            models.resnet50().maxpool,
            models.resnet50().layer1,
            models.resnet50().layer2,
            models.resnet50().layer3,
            models.resnet50().layer4)
        self.classifier = nn.Sequential(
            nn.Linear(2048 ** 2, 200))

    def forward(self, x):
        x = self.features(x)
        batch_size = x.size(0)
        x = x.view(batch_size, 2048, x.size(2) ** 2)
        x = (torch.bmm(x, torch.transpose(x, 1, 2)) /
             28 ** 2).view(batch_size, -1)
        x = torch.nn.functional.normalize(
            torch.sign(x) * torch.sqrt(torch.abs(x) + 1e-10))

        x = self.classifier(x)
        return x
