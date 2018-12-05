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

def test(i, key, shape, rand = False, randFactor = None, testFactor = None):
    global best_acc
    if rand == False:
        model = Pytorch_VGG.vgg16(pretrained = True)
        model.to(device)
        model.eval()
        state_dict = model.state_dict()
    

    if len(shape) == 4:
        size1 = shape[1]
        size2 = shape[2]
        size3 = shape[3]
        if rand:
            if (int(i/(size2*size3))%int(size1)) == torch.randint(0,size1-1,[1]):
                try:
                    flag = int(int(i)%randFactor) == torch.randint(0,randFactor-1,[1])
                except:
                    flag = True
                if (flag):
                    model = Pytorch_VGG.vgg16(pretrained = True)
                    model.to(device)
                    model.eval()
                    state_dict = model.state_dict()
                    (state_dict[key][int(i/size1/size2/size3)][int(i/size2/size3%size1)][int(i/size3%size2)][int(i%size3)]).mul_(-1)
                else:
                    return 100
            else:
                return 100
        else:
            (state_dict[key][int(i/size1/size2/size3)][int(i/size2/size3%size1)][int(i/size3%size2)][int(i%size3)]).mul_(-1)

    if len(shape) == 1:
        if rand:
            if (int(int(i)%randFactor) == torch.randint(0,randFactor-1,[1])):
                model = Pytorch_VGG.vgg16(pretrained = True)
                model.to(device)
                model.eval()
                state_dict[key][i].mul_(-1)
            else:
                return 100
        else:
            state_dict[key][i].mul_(-1)

    if len(shape) == 2:
        size = state_dict[key].shape[1]
        if rand:
            if (int(int(i)%randFactor) == torch.randint(0,randFactor-1,[1])):
                model = Pytorch_VGG.vgg16(pretrained = True)
                model.to(device)
                model.eval()
                (state_dict[key][int(i/size)][i%size]).mul_(-1)
            else:
                return 100
        else:
            (state_dict[key][int(i/size)][i%size]).mul_(-1)
            
    theIter = 0
    correct = 0
    totalItems = 0
    with torch.no_grad():
        for data in tqdm.tqdm(val_loader, leave = False):
            if theIter%testFactor == 0:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                totalItems += labels.size(0)
                correct += (predicted == labels).sum().item()
                theIter += 1
    acc = float(correct) / float(totalItems) * 100
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
    
    if not args.cpu:
        model.to(device)

    if args.verbose:
        print(model)

    rand = True
    randFactor = 1
    testFactor = 5
    count = 0
    tLoss = 0
    lMax = 0
    lAvg = 0
    bestAcc = 90.625
    save = []

    find_key = "features.0.weight"
    print(find_key)
    state_dict = model.state_dict()

    for key in state_dict.keys():
        if key.find(find_key) != -1:
            total = 1
            shape = state_dict[key].shape
            use_key = key
            for t in range(len(state_dict[key].shape)):
                total *= state_dict[key].shape[t]  

    with tqdm.tqdm(range(total)) as Loader:
        start = time.time()
        for i in Loader:
            acc = test(i, use_key, shape = shape, rand = rand, randFactor=randFactor, testFactor = testFactor)
            loss = bestAcc - acc

            if (acc != 100):
                count += 1
                tLoss += loss
                lAvg  = tLoss / float(count)
                save.append((i,loss))
                Loader.set_description("T: %d, Av: %.2f%%, M: %.2f%%"%(count, lAvg, lMax))

            if (loss > lMax):
                lMax = loss

            end = time.time()
            if (end - start > 300):
                np.save(find_key+'_tmp',save)
                start = end

    np.save(find_key+'.neg', save)
    print ("lAvg = %f%%, Max = %f%%"%(lAvg, lMax))
    exit()
