from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import torch
import argparse
import models

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import tqdm
import time
import numpy as np

def test(i, key, shape, rand = False, bypass = False, randFactor = None, memoryData = None):
    global best_acc
    if rand == False:
        state_dict = torch.load(args.pretrained)
        model = models.nin()
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        state_dict = model.state_dict()
    

    if len(shape) == 4:
        size1 = shape[1]
        size2 = shape[2]
        size3 = shape[3]
        if rand:
            if ((int(i/(size2*size3))%int(size1)) == torch.randint(0,size1-1,[1]) or bypass):
                try:
                    flag = int(int(i)%randFactor) == torch.randint(0,randFactor-1,[1])
                except:
                    flag = True
                if (flag):
                    state_dict = torch.load(args.pretrained)
                    model = models.nin()
                    model.load_state_dict(state_dict)
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
                state_dict = torch.load(args.pretrained)
                model = models.nin()
                model.load_state_dict(state_dict)
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
                state_dict = torch.load(args.pretrained)
                model = models.nin()
                model.load_state_dict(state_dict)
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
        model.eval()
        for data in memoryData:
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
    parser.add_argument('--pretrained', action='store', default='good.pt',
            help='the path to the pretrained model')
    parser.add_argument('--verbose', action='store_true', default=False,
            help='display more information')
    parser.add_argument('--device', action='store', default='cuda:1',
            help='input the device you want to use')
    args = parser.parse_args()
    if args.verbose:
        print('==> Options:',args)
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # set the seed
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    #indices = np.load("subset.npy")

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    testset = torchvision.datasets.CIFAR10(root='../CIFAR_10/data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=256,
                                             shuffle=False, num_workers=2, pin_memory=True)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    state_dict = torch.load(args.pretrained)
    model = models.nin()
    model.load_state_dict(state_dict)
    
    if not args.cpu:
        model.to(device)

    if args.verbose:
        print(model)

    rand = False
    bypass = True
    randFactor =4
    testFactor = 1
    count = 0
    tLoss = 0
    lMax = 0
    lAvg = 0
    bestAcc = 88.51
    save = []
    memoryData = []

    find_key = "classifier.20.weight"
    print(find_key)
    state_dict = model.state_dict()

    for key in state_dict.keys():
        if key.find(find_key) != -1:
            total = 1
            shape = state_dict[key].shape
            use_key = key
            for t in range(len(state_dict[key].shape)):
                total *= state_dict[key].shape[t]
    

    for data in tqdm.tqdm(testloader, leave = False):
        memoryData += [data]

    with tqdm.tqdm(range(total)) as Loader:
        start = time.time()
        for i in Loader:
            acc = test(i, use_key, shape = shape, rand = rand, bypass = bypass, randFactor=randFactor, memoryData = memoryData)
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
