from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import torch
import argparse
import torch.nn as nn
import torch.optim as optim

from models import resnet
import torchvision
import torchvision.transforms as transforms
import tqdm
import time
import numpy as np

def load_pretrained(filePath, same):
    if args.arch == '56':
        model = resnet.resnet56_cifar(num_classes=100)
    if args.arch == '110':
        model = resnet.resnet110_cifar(num_classes=100)
        
    pretrained_model = torch.load(filePath)
    useState_dict = model.state_dict()
    preState_dict = pretrained_model['state_dict']
    best_acc = pretrained_model['best_acc']
    if same:
        useState_dict = preState_dict
    else:
        useKeys = useState_dict.keys()
        preKeys = preState_dict.keys()
        j = 0
        for key in useKeys:
            useState_dict[key].data = preState_dict[preKeys[j]].data
            j +=1
    model.load_state_dict(useState_dict)
    model.to(device)
    #if args.device == 'cuda:0':
    #    model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])
    return model, best_acc

def test(i, key, shape, rand = False, bypass = False, randFactor = None, memoryData = None, same = False):
    global best_acc
    test_loss = 0
    correct = 0
    if (not rand) or (len(shape) != 4):
        model, best_acc = load_pretrained(args.pretrained, same)
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
                    model, best_acc = load_pretrained(args.pretrained, same)
                    model.to(device)
                    model.eval()
                    state_dict = model.state_dict()
                    (state_dict[key][int(i/size1/size2/size3)][int(i/size2/size3%size1)][int(i/size3%size2)][int(i%size3)]).mul_(0)
                else:
                    return 100
            else:
                return 100
        else:
            (state_dict[key][int(i/size1/size2/size3)][int(i/size2/size3%size1)][int(i/size3%size2)][int(i%size3)]).mul_(0)

    if len(shape) == 1:
        if rand:
            if (int(int(i)%randFactor) == torch.randint(0,randFactor-1,[1])):
                model, best_acc = load_pretrained(args.pretrained, same)
                model.to(device)
                model.eval()
                state_dict[key][i].mul_(0)
            else:
                return 100
        else:
            state_dict[key][i].mul_(-1)

    if len(shape) == 2:
        size = state_dict[key].shape[1]
        if rand:
            if (int(int(i)%randFactor) == torch.randint(0,randFactor-1,[1])):
                model, best_acc = load_pretrained(args.pretrained, same)
                model.to(device)
                model.eval()
                (state_dict[key][int(i/size)][i%size]).mul_(0)
            else:
                return 100
        else:
            (state_dict[key][int(i/size)][i%size]).mul_(0)
            
    with torch.no_grad():
        for data, target in memoryData:
            data, target = data.to(device), target.to(device)

            output = model(data)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    acc = 100. * float(correct) / float(len(testloader.dataset))
    #print (acc,correct,len(testloader.dataset))
    return acc


if __name__=='__main__':
    # prepare the options
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', action='store', default='./data/',
            help='dataset path')
    parser.add_argument('--pretrained', action='store', default='models/res56.best.pth.tar',
            help='the path to the pretrained model')
    parser.add_argument('--evaluate', action='store_true', default=True,
            help='evaluate the model')
    parser.add_argument('--verbose', action='store_true', default=False,
            help='display more information')
    parser.add_argument('--device', action='store', default='cuda:0',
            help='input the device you want to use')
    parser.add_argument('--different', action='store_true', default=False,
            help='if using the original model')
    parser.add_argument('--filename', action='store', default='',
            help='The filename we use')
    parser.add_argument('--arch', action='store', default='56',
            help='resnet numbers')
    parser.add_argument('--batch_size', action='store', default='1024',
            help='batch size')
    args = parser.parse_args()
    if args.verbose:
        print('==> Options:',args)
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # set the seed
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    # prepare the data
    normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
    test_dataset = torchvision.datasets.CIFAR100(
        root=args.data,
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))

    testloader = torch.utils.data.DataLoader(test_dataset,
                                 batch_size=int(args.batch_size), shuffle=False, num_workers=4)

    # define the model
    same = not args.different
    model, best_acc = load_pretrained(args.pretrained, same)
    model.to(device)

    # define solver and criterion
    param_dict = dict(model.named_parameters())
    params = []

    # do the evaluation if specified
    if args.evaluate:

        bestAcc = best_acc
        memoryData = []

        for data in tqdm.tqdm(testloader, leave = False):
            memoryData += [data]
        
        state_dict = model.state_dict()
        keyList = []
        for key in state_dict.keys():
            if key.find('conv1') != -1:
                keyList += [key]
                break
        
        keyList = ['conv1.weight', 'layer1.0.conv1.weight', 'layer1.1.conv1.weight', 'layer1.2.conv1.weight', 'layer1.7.conv1.weight', 'layer1.8.conv1.weight', 'layer2.0.conv1.weight', 'layer2.8.conv1.weight', 'layer3.0.conv1.weight', 'layer3.8.conv1.weight', 'fc.weight']
        print(keyList)

        for find_key in keyList:
            rand = False
            bypass = True
            randFactor = 4
            count = 0
            tLoss = 0
            lMax = 0
            lAvg = 0
            save = []
            for key in state_dict.keys():
                if key.find(find_key) != -1:
                    total = 1
                    shape = state_dict[key].shape
                    use_key = key
                    for t in range(len(state_dict[key].shape)):
                        total *= state_dict[key].shape[t]
                    break

            print(use_key, bestAcc)



            with tqdm.tqdm(range(total)) as Loader:
                start = time.time()
                for i in Loader:
                    acc = test(i, use_key, shape = shape, rand = rand, bypass = bypass, randFactor=randFactor, memoryData = memoryData, same=same)
                    loss = bestAcc - acc

                    if (acc != 100):
                        count += 1
                        lAvg  = tLoss / float(count)
                        tLoss += loss
                        save.append((i,loss))
                        Loader.set_description("T: %d, Av: %.2f%%, M: %.2f%%"%(count, lAvg, lMax))

                    if (loss > lMax):
                        lMax = loss

                    end = time.time()
                    if (end - start > 300):
                        np.save(find_key+'.'+args.arch+args.filename+'zero_tmp',save)
                        start = end

            np.save(find_key+'.'+args.arch+'.'+args.filename+'.zero', save)
            print ("lAvg = %f%%, Max = %f%%"%(lAvg, lMax))
        exit()
