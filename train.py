'''
Description: 
Author: nuso
LastEditors: Please set LastEditors
Date: 2021-03-12 16:27:43
LastEditTime: 2021-03-13 13:28:20
'''
import os
import time
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms

from model import Net
from dataset import cub_dataset
import argparse


parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--name', default='resnet50', type=str)
parser.add_argument('--save_path', default='F:\code\细粒度图像识别\models', type=str)
parser.add_argument(
    '--dataset_path', default='F:\code\细粒度图像识别\data\CUB\CUB_200_2011\CUB_200_2011', type=str)

parser.add_argument('--num_workers', default=8, type=int)
parser.add_argument('--num_epochs', default=20, type=int)
parser.add_argument('--batchsize', default=8, type=int)
parser.add_argument('--warm_epoch', default=0, type=int)
opt = parser.parse_args()


# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_ids = [0]

torch.backends.cudnn.benchmark = True

model = Net().to(device)
# model = nn.DataParallel(model, device_ids=device_ids).to(device)

criterion = nn.CrossEntropyLoss()
criterion.to(device)

optimizer = optim.SGD([
    {'params': model.features.parameters(), 'lr': 0.1*opt.lr},
    {'params': model.classifier.parameters(), 'lr': opt.lr},
], weight_decay=5e-4)

transform = {
    'train': transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.RandomCrop([200, 200]),
        transforms.ToTensor(),
    ]),
    'test': transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.ToTensor(),
    ])
}
# dataset = cub_dataset(opt.dataset_path, 'train', transform)
# dataloader = torch.utils.data.DataLoader(
#     dataset, batch_size=opt.batchsize, shuffle=True, num_workers=opt.num_workers, pin_memory=True)


datasets = {
    x: cub_dataset(opt.dataset_path, x, transform[x]) for x in ['train', 'test']
}
dataloaders = {
    x: torch.utils.data.DataLoader(
        datasets[x], batch_size=opt.batchsize, shuffle=True, num_workers=opt.num_workers, pin_memory=True)
    for x in ['train', 'test']
}

since = time.time()
num_epochs = opt.num_epochs
for epoch in range(num_epochs):
    torch.cuda.empty_cache()

    model.zero_grad()
    optimizer.zero_grad()

    eval_acc = 0.0
    for phase in ['train', 'test']:
        if phase == 'train':
            model.train(True)
        elif phase == 'test':
            model.eval()

        for i, data in enumerate(dataloaders[phase], 0):
            image, label = data
            image, label = image.to(device), label.to(device)

            # forward & loss
            if phase == 'test':
                with torch.no_grad():
                    pred = model(image)
            else:
                pred = model(image)
            loss = criterion(pred, label)

            if phase == 'train':
                loss.backward()
                optimizer.step()
            elif phase == 'test':
                prediction = torch.max(pred, 1)[1]
                num_correct = (prediction == label).sum()
                eval_acc += num_correct

            if (i+1) % 100 == 0:
                print(
                    f'Loss 【{loss.item()}】, Epoch {epoch+1}/{num_epochs}, Step {i+1}/total_step Phase f{phase}')

    # every epoch evaluate
    print('='*30+'>Accuracy of epoch{} : 【{:.6f}】'.format(epoch,
                                                        (eval_acc.float()) / (len(dataloaders['test']))))

    # save every 5 epochs
    if (epoch + 1) % 5 == 0:
        model.eval()
        checkpoint = {
            'state_dict': model.state_dict(),           # model.module.state_dict()
            'opt_state_dict': optimizer.state_dict(),
            'epoch': epoch
        }
        path = os.path.join(opt.save_path, f'{opt.name}_{epoch+1}.pt')
        torch.save(checkpoint, path)
        print(f'checkpoint save: {path}')

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print(f"============================= Epoch {epoch+1} done!\n")


path = os.path.join(opt.save_path, f'{opt.name}_END.pth')
torch.save(model.state_dict(), path)
print(f'model save: {path}')
