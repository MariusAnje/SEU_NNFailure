from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import torch
import argparse
import Pytorch_VGG

from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import tqdm
import time
import numpy as np

def test():
    totalItems = 0
    correct = 0
    with torch.no_grad():
        for data in tqdm.tqdm(val_loader, leave = False):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            totalItems += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = 100. * float(correct) / totalItems
    print(acc, correct, totalItems)
    return acc


if __name__=='__main__':
    # prepare the options
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', action='store_true',
            help='set if only CPU is available')
    parser.add_argument('--data', action='store', default='./data/',
            help='dataset path')
    parser.add_argument('--arch', action='store', default='nin',
            help='the architecture for the network: nin')
    parser.add_argument('--lr', action='store', default='0.01',
            help='the intial learning rate')
    parser.add_argument('--pretrained', action='store', default='nin.best.pth.tar',
            help='the path to the pretrained model')
    parser.add_argument('--evaluate', action='store_true', default=True,
            help='evaluate the model')
    parser.add_argument('--verbose', action='store_true', default=False,
            help='display more information')
    parser.add_argument('--device', action='store', default='cuda:0',
            help='input the device you want to use')
    args = parser.parse_args()
    if args.verbose:
        print('==> Options:',args)
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # set the seed
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)


    valdir = '/home/data/yanzy/val'
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=64, shuffle=False,
        num_workers=32, pin_memory=True)
    
    model = Pytorch_VGG.vgg16(pretrained = True)
    model.to(device)
    model.eval()
    acc = test()
            
